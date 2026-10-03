"""The two code-evolution tasks, PyEvo and RustEvo: data, prompts, code extraction and the judge.

A task sample is an API change together with a programming task that needs the changed API:
  * the *repair pair* (`question`, `answer`): the API, its new signature and the rephrased task, answered by
    the reference code; this is what a method is given;
  * the *test prompt* (`test_prompt`): the original task, answered by the model after the repair and judged
    by executing the test program of the sample.
`api_knowledge=True` adds the documentation and the source of the changed API to the test prompt (used by the
`proxy` protocol to find the tokens that need this knowledge).
"""
from __future__ import annotations

import io
import json
import os
import re
import tokenize
from typing import List, Tuple

import numpy as np

from gap.utils.settings import PROJECT_ROOT

from .chat import answer_suffix, format_prompt

DATA_DIR = PROJECT_ROOT / "data"
TASKS = {"pyevo": ("PyEvo", "python"), "rustevo": ("RustEvo", "rust")}


# ====================================================================== metrics
def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Pass@k of `c` correct samples among `n` (in [0, 1]): 1 - C(n-c, k) / C(n, k).
    """
    if n == 0:
        return 0.0
    if c > n:
        return 1.0
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


# ====================================================================== data
def strip_code_comments(code: str) -> str:
    """
    Strip comments and docstrings from Python code using tokenize + ast.
    Keeps code logic to reduce answer token count for model editing.
    """
    if not code or not code.strip():
        return code
    try:
        import ast as _ast

        docstring_lines: set = set()
        try:
            tree = _ast.parse(code)
            for node in _ast.walk(tree):
                if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef,
                                     _ast.ClassDef, _ast.Module)):
                    if (node.body and
                            isinstance(node.body[0], _ast.Expr) and
                            isinstance(node.body[0].value, _ast.Constant) and
                            isinstance(node.body[0].value.value, str)):
                        ds = node.body[0]
                        for ln in range(ds.lineno, ds.end_lineno + 1):
                            docstring_lines.add(ln)
        except SyntaxError:
            pass  # Continue anyway, just skip docstrings

        result_lines = []
        for lineno, line in enumerate(code.splitlines(), start=1):
            if lineno in docstring_lines:
                continue
            stripped = _strip_inline_comment(line)
            result_lines.append(stripped)

        cleaned: list = []
        prev_blank = False
        for line in result_lines:
            if line.strip() == '':
                if not prev_blank:
                    cleaned.append('')
                prev_blank = True
            else:
                cleaned.append(line)
                prev_blank = False

        return '\n'.join(cleaned).strip()
    except Exception:
        return code


def _strip_inline_comment(line: str) -> str:
    """Strip trailing # comments from a line, correctly handling # inside strings."""
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(line).readline))
        for tok_type, _, tok_start, tok_end, _ in tokens:
            if tok_type == tokenize.COMMENT:
                return line[:tok_start[1]].rstrip()
        return line.rstrip()
    except tokenize.TokenError:
        return line.rstrip()


def _is_rust_crate(module: str) -> bool:
    return not any((module or "").startswith(prefix) for prefix in ("std::", "core::", "alloc::"))


def _repair_question(sample: dict, language: str) -> str:
    name, module = sample.get("name", ""), sample.get("module", "")
    query = sample.get("rephrased_query") or sample.get("query", "")
    if not (name and module and query):
        separator = "::" if language == "rust" else "."
        return query or f"Implement using {module}{separator}{name}"
    if language == "rust":
        api_info = f"{module}::{name}"
        signature = sample.get("signature", "") or api_info
    else:
        api_info = f"{module}.{name}" if "." not in name else name
        signature = sample.get("signature", "")
    function_signature = sample.get("function_signature", "").strip()
    return "\n".join([
        f"API Information: {api_info}",
        f"API Signature: {signature}",
        f"Task: {query}",
        "Function Signature:",
        f"```{language}\n{function_signature}\n```",
    ])


def _repair_views(sample: dict, language: str) -> List[str]:
    """Other renderings of the information of the repair prompt (same fields: API, signature, rephrased task,
    function signature); used by methods that ask for several views of one repair pair."""
    name, module = sample.get("name", ""), sample.get("module", "")
    query = sample.get("rephrased_query") or sample.get("query", "")
    if not (name and module and query):
        return []
    sep = "::" if language == "rust" else "."
    api = f"{module}{sep}{name}" if language == "rust" or "." not in name else name
    signature = sample.get("signature", "") or api
    function_signature = sample.get("function_signature", "").strip()
    # ordered by how different they are from the original: sectioned (markdown), free text, sentence, code first
    return [
        (f"### Task\n{query}\n\n### API\n{api}\n{signature}\n\n### Function signature\n{function_signature}\n\n"
         f"### Language\n{language}"),
        (f"I need {language} code. {query} The solution has to call `{api}` ({signature}). "
         f"Define it as `{function_signature}`."),
        (f"Write a {language} function for the task below. It must use the API `{api}`, whose signature is "
         f"`{signature}`.\n\nTask: {query}\n\nThe function must have exactly this signature:\n"
         f"```{language}\n{function_signature}\n```"),
        (f"```{language}\n{function_signature}\n```\nComplete the function above. {query}\n"
         f"Use the API {api}: {signature}"),
    ]


def _answer(code: str, language: str) -> str:
    """The reference answer of the repair pair, written as the chat models answer: the code in a fenced block, so that a
    repair has to teach the code and not the format (a repaired model that is taught bare code stops closing its
    answer)."""
    return f"```{language}\n{code.rstrip()}\n```"


def load_task(task: str, model_name: str, limit: int = None) -> List[dict]:
    folder, language = TASKS[task]
    path = DATA_DIR / folder / f"{folder}.json"
    if not path.is_file():
        raise FileNotFoundError(f"{folder} data file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # GAP_SPLIT=dev|test: every 8th sample (dev) or all the others (test), so that the technique is developed on a
    # subset spread over the change types and reported on the rest. GAP_SHARD=k/n: the samples whose index is k modulo
    # n, so that a run can be extended later with the other shards. --limit then spreads N samples evenly.
    indexed = list(enumerate(raw))
    split = os.environ.get("GAP_SPLIT", "")
    if split in ("dev", "test"):
        indexed = [(i, x) for i, x in indexed if (i % 8 == 0) == (split == "dev")]
    shard = os.environ.get("GAP_SHARD", "")
    if shard:
        k, n = (int(v) for v in shard.split("/"))
        indexed = [(i, x) for i, x in indexed if i % n == k]
    if limit and limit < len(indexed):
        indexed = [indexed[round(i * (len(indexed) - 1) / max(limit - 1, 1))] for i in range(limit)]

    samples = []
    for index, sample in indexed:
        name = sample.get("name", "")
        code = sample.get("code", "") or ""
        if language == "python":
            # comments and docstrings only add answer tokens
            code = strip_code_comments(code)
        question = _repair_question(sample, language)
        formatted = format_prompt(model_name, question)
        samples.append({
            **sample,
            "index": index,
            "id": f"{name}_{sample.get('from_version', 'unknown')}_to_{sample.get('to_version', 'unknown')}"
                  if name else "unknown",
            "task": task,
            "language": language,
            "code": code,
            "question": formatted if formatted is not None else question,
            "views": [format_prompt(model_name, v) or v for v in _repair_views(sample, language)],
            "answer": _answer(code, language) + answer_suffix(model_name),
        })
    return samples


# ====================================================================== prompts
def test_prompt(sample: dict, api_knowledge: bool = False, phrasing: str = "original") -> str:
    """The code-generation prompt of a sample (before the chat template). `phrasing` 'spec' asks the same task as a
    terse specification list (a wording that no repair view uses): the robustness check of the views."""
    query = sample.get("query", "")
    name, module = sample.get("name", ""), sample.get("module", "")
    function_signature = sample.get("function_signature", "").strip()
    rust = sample["language"] == "rust"
    crate = rust and _is_rust_crate(module)

    info = []
    if crate:
        info.append(f"- Crate Name: {module.split('::')[0] if '::' in module else module}")
    info += [f"- API Name: {name}", f"- API Module: {module}"]
    if api_knowledge:
        info += [
            f"- API Signature: {sample.get('signature', '')}",
            f"- API Documentation: {sample.get('documentation', '')}",
        ]
        if not rust:
            info.append(f"- API Old Source Code: {sample.get('old_source_code', '')}")
        info += [
            f"- API {'' if rust else 'New '}Source Code: {sample.get('source_code', '')}",
            f"- API Changed From Version: {sample.get('from_version', '')}",
            f"- API Changed To Version: {sample.get('to_version', '')}",
        ]
    info = "\n".join(info)

    if phrasing == "spec":
        # a terse specification list: no sections, no free text and no code-first layout, and none of the repair views
        language = "Rust" if rust else "Python"
        return (f"Implement a {language} function.\n- Goal: {query}\n- Required API: {name} (module {module})\n"
                f"- Signature: {function_signature}\nOutput only the code block.")

    if rust:
        compile_note = "Compile with Rust 1.84.0. " if crate else ""
        return f"""You are an expert Rust programmer. Implement the following Rust function:

API Information:
{info}

Task: {query}
Function Signature:
```rust
{function_signature}
```

Requirements:
1. Implement ONLY the Rust function with the signature given above.
2. Your implementation MUST use the specified API: {name}
3. {compile_note}Do not include tests or any extra comments.

Respond with ONLY the Rust function implementation."""

    return f"""You are an expert Python programmer. Implement the following Python function.

API Information:
{info}

Task: {query}

Function Signature:
```python
{function_signature}
```

Requirements:
1. Implement ONLY the function with the signature given above.
2. Your implementation MUST use the specified API: {name}
3. Do not include tests or any extra comments.

Respond with ONLY the Python function implementation."""


# ====================================================================== code extraction
def _unterminated_fence(text: str) -> str:
    """The text after an opening code fence that is never closed ('' if there is none)."""
    match = re.search(r"```[A-Za-z0-9_+#-]*[ \t]*\n([\s\S]*)$", text)
    if not match or '```' in match.group(1):
        return ""
    return match.group(1).strip()


def extract_rust_code(text: str) -> str:
    """Extract Rust code block from LLM output."""
    # Strategy 1: fenced code block with language tag
    lang_blocks = re.findall(r"```(?:rust|rs|Rust)\b[^\n]*\n([\s\S]*?)```", text)
    if lang_blocks:
        valid = [m.strip() for m in lang_blocks if m.strip()]
        if valid:
            return max(valid, key=len)
    # Strategy 2: bare fenced code block
    bare_blocks = re.findall(r"```\s*\n([\s\S]*?)```", text)
    if bare_blocks:
        valid = [m.strip() for m in bare_blocks if m.strip()]
        if valid:
            return max(valid, key=len)
    # Strategy 3: an opening fence without a closing one (a repaired model ends its turn right after the code)
    open_block = _unterminated_fence(text)
    if open_block:
        return open_block
    # Strategy 4: trailing fence only
    candidate = re.sub(r"```\s*$", "", text.strip())
    if candidate and '```' not in candidate:
        return candidate.strip()
    return text.strip("` \n\t")


def extract_python_code(text: str) -> str:
    """Extract Python code from LLM output."""
    code = None

    lang_blocks = re.findall(r"```(?:[Pp]ython[23]?|py)\b[^\n]*\n([\s\S]*?)```", text)
    if lang_blocks:
        code = max(lang_blocks, key=len).strip()

    if code is None:
        bare_blocks = re.findall(r"```\s*\n([\s\S]*?)```", text)
        if bare_blocks:
            code_blocks = [m.strip() for m in bare_blocks
                           if m.strip() and ('def ' in m or 'import ' in m)]
            if code_blocks:
                code = max(code_blocks, key=len)
            else:
                valid = [m.strip() for m in bare_blocks if m.strip()]
                if valid:
                    code = max(valid, key=len)

    if code is None and '[PYTHON]' in text and '[/PYTHON]' in text:
        start = text.find('[PYTHON]') + len('[PYTHON]')
        end = text.find('[/PYTHON]')
        if end > start:
            code = text[start:end].strip()

    if code is None:
        code = _unterminated_fence(text) or None

    if code is None:
        candidate = re.sub(r"```\s*$", "", text.strip())
        if candidate and '```' not in candidate:
            code = candidate.strip()

    if code is None:
        return text.strip()

    # Filter out test functions and test calls
    lines = code.split('\n')
    result_lines = []
    in_test_func = False
    main_func_indent = None

    for line in lines:
        stripped = line.strip()

        # Skip blank lines (preserve within functions)
        if not stripped:
            if result_lines and not in_test_func:
                result_lines.append(line)
            continue

        # Detect test function definitions
        if stripped.startswith('def test_') or stripped.startswith('async def test_'):
            in_test_func = True
            continue

        # Inside test function body
        if in_test_func:
            if line.startswith((' ', '\t')):
                continue
            else:
                in_test_func = False

        # Skip test calls
        if stripped.startswith('test_') and '(' in stripped:
            continue

        if stripped.startswith('#') and 'test' in stripped.lower():
            continue

        # Collect imports
        if stripped.startswith(('import ', 'from ')):
            result_lines.append(line)
            continue

        # Collect main function definition
        is_func_def = stripped.startswith('def ') or stripped.startswith('async def ')
        is_test_def = stripped.startswith('def test_') or stripped.startswith('async def test_')
        if is_func_def and not is_test_def:
            main_func_indent = len(line) - len(line.lstrip())
            result_lines.append(line)
            continue

        # Collect function body
        if main_func_indent is not None:
            current_indent = len(line) - len(line.lstrip()) if stripped else 0
            if current_indent > main_func_indent or not stripped:
                result_lines.append(line)
            elif (stripped.startswith('def ') or stripped.startswith('async def ')) and not (stripped.startswith('def test_') or stripped.startswith('async def test_')):
                main_func_indent = len(line) - len(line.lstrip())
                result_lines.append(line)
            elif current_indent == main_func_indent and not (stripped.startswith('def test_') or stripped.startswith('async def test_')) and not (stripped.startswith('test_') and '(' in stripped):
                result_lines.append(line)

    # Trim trailing blank lines
    while result_lines and not result_lines[-1].strip():
        result_lines.pop()

    return '\n'.join(result_lines) if result_lines else text.strip()


def _parse_rust_test_output(stdout: str) -> Tuple[int, int]:
    """Parse the output of a Rust test run, return (passed, total)"""
    if not stdout:
        return 0, 0
    m = re.search(r"test result:.*?(\d+) passed; (\d+) failed", stdout)
    if m:
        p, f = int(m.group(1)), int(m.group(2))
        return p, p + f
    return 0, 0


def _parse_pytest_output(output: str) -> Tuple[int, int]:
    """Parse the output of pytest, return (passed, total)"""
    if not output:
        return 0, 0
    mp = re.search(r"(\d+) passed", output)
    mf = re.search(r"(\d+) failed", output)
    me = re.search(r"(\d+) error", output)
    p = int(mp.group(1)) if mp else 0
    f = (int(mf.group(1)) if mf else 0) + (int(me.group(1)) if me else 0)
    return p, p + f


def _count_rust_tests(test_program: str) -> int:
    return len(re.findall(r"#\[test\]", test_program or ""))


def _count_py_tests(test_program: str) -> int:
    return len(re.findall(r"def test_", test_program or ""))


# ====================================================================== judge
def preflight(task: str, python_bin: str = None) -> None:
    """Fail before the model is loaded if the tests of `task` cannot be executed in this environment."""
    import shutil
    import subprocess

    if TASKS[task][1] == "rust":
        missing = [tool for tool in ("cargo", "rustc") if shutil.which(tool) is None]
        if missing:
            raise RuntimeError(f"RustEvo needs {' and '.join(missing)} on PATH (toolchains 1.71.0-1.91.0 via rustup)")
        return
    from . import pytest_runner

    interpreter = python_bin or pytest_runner.DEFAULT_PYTHON_BIN
    probe = subprocess.run([interpreter, "-m", "pytest", "--version"], capture_output=True, text=True)
    if probe.returncode != 0:
        raise RuntimeError(
            f"PyEvo needs pytest in the interpreter that runs the tests ({interpreter}); "
            f"choose it with --test_python or PYEVO_PYTHON"
        )


def judge(sample: dict, prediction: str, python_bin: str = None) -> dict:
    """Extract the code of a prediction, check signature and API usage, and execute the tests of the sample.

    `test_status`: PASSED, NO_TEST, EXTRACTION_FAILED, SIGNATURE_ERROR, API_ERROR, or the status of the run.
    `python_bin` runs the PyEvo tests.
    """
    rust = sample["language"] == "rust"
    test_program = sample.get("test_program", "") or ""
    n_cases = (_count_rust_tests if rust else _count_py_tests)(test_program)
    code = (extract_rust_code if rust else extract_python_code)(prediction)
    result = {
        "id": sample.get("id"),
        "prediction": prediction,
        "extracted_code": code,
        "test_status": None,
        "test_passed": False,
        "test_cases_passed": 0,
        "test_cases_total": 0,
    }

    def failed(status: str) -> dict:
        result["test_status"] = status
        result["test_cases_total"] = n_cases
        return result

    if not code or len(code.strip()) < 10:
        return failed("EXTRACTION_FAILED")
    if not test_program or test_program == "INCORRECT CODE":
        result["test_status"] = "NO_TEST"
        return result

    name, module = sample.get("name", ""), sample.get("module", "")
    change_type, signature = sample.get("change_type", ""), sample.get("function_signature", "")

    if rust:
        from . import cargo_runner as runner

        if signature and not runner.check_function_signature(code, signature):
            return failed("SIGNATURE_ERROR")
        if name and not runner.check_api_usage(
            code, name, change_type, module, test_program, sample.get("replacement_api", "")
        ):
            return failed("API_ERROR")
        clean_code, clean_test, rust_version, crate_version, _, _ = runner.prepare_rust_test_inputs(
            code, test_program, module, sample.get("to_version", "1.84.0"), dedup_imports=True,
        )
        run = runner.run_rust_test_auto(
            clean_code, clean_test, module, rust_version, crate_version, timeout=runner.DEFAULT_TEST_TIMEOUT,
        )
        result["test_status"] = "PASSED" if run.get("success") else str(run.get("status") or "FAILED")
        result["test_passed"] = bool(run.get("success", False))
        result["error_type"] = run.get("error_type", "")
        result["test_output"] = (run.get("stdout") or "") + (run.get("stderr") or "")
        problem = str(run.get("error") or "") + result["test_output"]
        if "toolchain" in problem and "is not installed" in problem:
            # an environment problem must not be counted as a failure of the model
            raise RuntimeError(f"missing Rust toolchain for {sample.get('id')}: {problem[:300]}")
        passed, _ = _parse_rust_test_output(result["test_output"])
    else:
        from . import pytest_runner as runner

        if signature and not runner.check_function_signature(code, signature):
            return failed("SIGNATURE_ERROR")
        if name and not runner.check_api_usage(code, name, change_type, module):
            return failed("API_ERROR")
        interpreter = python_bin or runner.DEFAULT_PYTHON_BIN
        run = runner.run_python_test(code, test_program, timeout=runner.DEFAULT_TEST_TIMEOUT, python_bin=interpreter)
        result["test_status"] = "PASSED" if run["success"] else "FAILED"
        result["test_passed"] = bool(run["success"])
        result["error_type"] = "" if run["success"] else str(run.get("error", ""))
        result["test_output"] = run.get("output", "") or ""
        passed, _ = _parse_pytest_output(result["test_output"])

    if run.get("error"):
        result["test_error"] = str(run["error"])
    result["test_cases_passed"] = n_cases if (result["test_passed"] and passed == 0) else min(passed, n_cases)
    result["test_cases_total"] = n_cases
    return result


def failure_class(result: dict) -> str:
    """One of: passed, no_test, extraction_error, signature_error, api_error, compilation_failed, timeout,
    test_failed, other_error."""
    if result.get("test_passed"):
        return "passed"
    status = str(result.get("test_status") or "")
    fixed = {
        "NO_TEST": "no_test", "EXTRACTION_FAILED": "extraction_error",
        "SIGNATURE_ERROR": "signature_error", "API_ERROR": "api_error",
    }
    if status in fixed:
        return fixed[status]
    kind = (str(result.get("error_type") or "") + " " + status).lower()
    if "compil" in kind:
        return "compilation_failed"
    if "timeout" in kind:
        return "timeout"
    if "execution" in kind:
        return "other_error"
    if "test" in kind or status == "FAILED":
        return "test_failed"
    return "other_error"
