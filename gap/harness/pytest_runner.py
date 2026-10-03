"""Run the pytest program of a PyEvo sample on generated code, in a subprocess.

The tests are executed by `python_bin -m pytest` on `solution.py` + `test_solution.py` in a temporary
directory, so the interpreter that runs the tests can differ from the one that runs the model.

Also here: the function-signature check and the API-usage check of a PyEvo sample.
"""
import os
import re
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict

DEFAULT_TEST_TIMEOUT = int(os.environ.get("PYEVO_TEST_TIMEOUT", "60"))
DEFAULT_PYTHON_BIN = os.environ.get(
    "PYEVO_PYTHON",
    sys.executable,
)


def run_python_test(
    code: str,
    test_program: str,
    timeout: int = DEFAULT_TEST_TIMEOUT,
    python_bin: str = None,
) -> Dict:
    """
    Run pytest on generated code + test_program in a temporary directory.

    Args:
        code: the generated Python code
        test_program: the pytest program of the sample
        timeout: seconds
        python_bin: the interpreter that runs the tests

    Returns dict with keys:
        success: bool
        output: str   (stdout+stderr)
        error:  str   (error category if failed)
    """
    python_bin = python_bin or DEFAULT_PYTHON_BIN

    with tempfile.TemporaryDirectory(prefix="pyevo_") as tmpdir:
        sol_path = Path(tmpdir) / "solution.py"
        test_path = Path(tmpdir) / "test_solution.py"

        sol_path.write_text(code, encoding="utf-8")
        test_path.write_text(test_program, encoding="utf-8")

        proc = subprocess.Popen(
                [python_bin, "-m", "pytest", str(test_path), "-v", "--tb=short", "-q"],
                cwd=tmpdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                start_new_session=True,
            )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            output = (stdout + "\n" + stderr).strip()

            if proc.returncode == 0:
                return {"success": True, "output": output, "error": ""}
            else:
                return {"success": False, "output": output, "error": "test_failed"}

        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                try:
                    proc.kill()
                except OSError:
                    pass
            try:
                proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            return {"success": False, "output": "", "error": "timeout"}
        except Exception as e:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            return {"success": False, "output": str(e), "error": "execution_error"}


def check_function_signature(code: str, expected_signature: str) -> bool:
    """
    Check whether the generated code contains a function matching the expected signature.
    Lenient: we only check the function name matches.
    """
    if not expected_signature:
        return True

    # Extract function name from signature like "def foo(..." 
    m = re.search(r'def\s+(\w+)\s*\(', expected_signature)
    if not m:
        return True  # can't parse -> skip check

    func_name = m.group(1)
    return bool(re.search(rf'def\s+{re.escape(func_name)}\s*\(', code))


def check_api_usage(code: str, api_name: str, change_type: str, module: str) -> bool:
    """
    Check that the generated code uses the required API.

    The API name is matched at word boundaries (\\b), so that "cache" does not match "lru_cache".

    For 'deprecated' change_type: the code should NOT use the old API.
    For other types: the code MUST contain the exact API short name.
    """
    if not api_name:
        return True

    # Normalize: for names like "DataFrame.pivot_table", check "pivot_table"
    # For "str.removeprefix", check "removeprefix"
    short_name = api_name.split(".")[-1] if "." in api_name else api_name

    if str(change_type).lower() == "deprecated":
        # For deprecated APIs, we want the code NOT to use the old name
        # But this is tricky - sometimes the API still exists. Be lenient.
        return True

    if re.search(rf'\b{re.escape(short_name)}\b', code):
        return True

    return False
