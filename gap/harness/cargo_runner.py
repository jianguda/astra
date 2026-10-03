"""Run the test program of a RustEvo sample on generated code, with rustc or cargo.

Features:
1. third-party crate dependencies are detected from the `use` lines
2. crate features are configured per crate (`CRATE_FEATURES`)
3. the toolchain of the target version of the sample is used (`rustup run`)
4. errors are classified (compilation, failed test, timeout)

Test strategy:
- samples that use only the standard library: `rustc --test`
- samples that use third-party crates: `cargo test`

Also here: the function-signature check and the API-usage check of a RustEvo sample.
"""
import os
import subprocess
import tempfile
import re
from pathlib import Path
from typing import Dict, Set, Optional, Tuple


CRATE_FEATURES = {
    'libc': [],
    'rustix': ['fs', 'net', 'use-libc-auxv'],
    'time': ['formatting', 'macros', 'parsing'],
    'chrono': [],
    'rand_core': [],
    'nix': ['net', 'socket', 'user'],
    'socket2': [],
    'serde': ['derive'],
    'serde_json': [],
    'regex': [],
    'itertools': [],
    'once_cell': [],
    'anyhow': [],
    'rand': [],
    'thiserror': [],
    'log': ['kv'],
    'bytes': [],
    'bitflags': [],
    'textwrap': [],
    'tempfile': [],
    'hyper': ['server'],
    'hyper-util': ['client-legacy', 'http1', 'http2', 'tokio'],
    'http-body': [],
    'http-body-util': [],
    'tokio': ['full'],
    'futures-util': [],
}

CRATE_MAX_VERSIONS_PRE_185 = {
    'time': '<0.3.45',
    'tempfile': '<3.25',
}

INSTALLED_RUST_MIN = 71  # 1.71.0
INSTALLED_RUST_MAX = 91  # 1.91.0
DEFAULT_RUST_VERSION = "1.84.0"
DEFAULT_TEST_TIMEOUT = int(os.environ.get("RUSTEVO_TEST_TIMEOUT", "120"))
STD_MODULES = {'std', 'core', 'alloc', 'super', 'self', 'crate', 'test'}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CARGO_HOME_DEFAULT = PROJECT_ROOT / ".cargo"
CARGO_TARGET_DIR_DEFAULT = PROJECT_ROOT / ".cargo_target"


def get_cargo_env(temp_dir: Optional[str] = None) -> Dict[str, str]:
    """Build a stable cargo env to reuse caches across runs.
    
    Args:
        temp_dir: If provided, creates an isolated CARGO_TARGET_DIR for this build.
                  This avoids conflicts when multiple tests run concurrently.
    """
    env = os.environ.copy()
    
    if "CARGO_HOME" not in env:
        CARGO_HOME_DEFAULT.mkdir(parents=True, exist_ok=True)
        env["CARGO_HOME"] = str(CARGO_HOME_DEFAULT)
    
    if "CARGO_TARGET_DIR" not in env:
        if temp_dir:
            isolated_target = os.path.join(temp_dir, "target")
            os.makedirs(isolated_target, exist_ok=True)
            env["CARGO_TARGET_DIR"] = isolated_target
        else:
            CARGO_TARGET_DIR_DEFAULT.mkdir(parents=True, exist_ok=True)
            env["CARGO_TARGET_DIR"] = str(CARGO_TARGET_DIR_DEFAULT)
    
    return env


def is_crate_version(version: str) -> bool:
    """Whether `version` is the version of a third-party crate (0.x.x)."""
    return version.startswith('0.') if version else False


def is_rust_version(version: str) -> bool:
    """Whether `version` is a Rust toolchain version (1.xx.x)."""
    if not version or not version.startswith('1.'):
        return False
    parts = version.split('.')
    if len(parts) < 2:
        return False
    try:
        minor = int(parts[1])
        return 0 <= minor <= 99
    except ValueError:
        return False


def normalize_rust_version(version: str) -> str:
    """
    Map a Rust version to the range of installed toolchains (1.71.0-1.91.0).
    """
    if not is_rust_version(version):
        return DEFAULT_RUST_VERSION
    
    parts = version.split('.')
    if len(parts) >= 2:
        try:
            minor = int(parts[1])
            minor = max(INSTALLED_RUST_MIN, min(minor, INSTALLED_RUST_MAX))
            return f"1.{minor}.0"
        except ValueError:
            return DEFAULT_RUST_VERSION
    return DEFAULT_RUST_VERSION


def extract_main_crate(module: str, code: str) -> str:
    """
    The crate that the API of a sample belongs to.
    
    Args:
        module: module of the sample, e.g. "rustix::fs", "net::sockopt", "core::str"
        code: code of the sample; its `use` lines are searched when the module does not name the crate
    
    Returns:
        the crate name, e.g. "rustix", "hyper", "std"
    """
    if module.startswith(('std::', 'core::', 'alloc::')):
        return 'std'
    
    if '::' in module:
        potential_crate = module.split('::')[0]
        if potential_crate.replace('_', '-') in CRATE_FEATURES:
            return potential_crate.replace('_', '-')
    
    for line in code.split('\n'):
        line = line.strip()
        if line.startswith('use '):
            parts = line.split()
            if len(parts) >= 2:
                crate_path = parts[1].rstrip(';')
                crate_name = crate_path.split('::')[0]
                if crate_name not in STD_MODULES:
                    return crate_name.replace('_', '-')
    
    return 'std'


def extract_crate_names(code: str, test_code: str = "") -> Set[str]:
    """Names of the third-party crates that Rust code uses (the standard library is left out)."""
    all_code = code + '\n' + test_code
    crate_names = set()
    lines = all_code.split('\n')
    
    for line in lines:
        line = line.strip()
        if line.startswith('use '):
            parts = line.split()
            if len(parts) >= 2:
                crate_path = parts[1].rstrip(';')
                crate_name = crate_path.split('::')[0]
                if crate_name not in STD_MODULES:
                    crate_name = crate_name.replace('_', '-')
                    crate_names.add(crate_name)
        elif line.startswith('extern crate '):
            parts = line.split()
            if len(parts) >= 3:
                crate_name = parts[2].rstrip(';')
                if crate_name not in STD_MODULES:
                    crate_name = crate_name.replace('_', '-')
                    crate_names.add(crate_name)
    
    return crate_names


def prepare_rust_test_inputs(
    code: str,
    test_code: str,
    module: str,
    to_version: str,
    dedup_imports: bool = True,
) -> Tuple[str, str, str, Optional[str], bool, Set[str]]:
    """Normalize inputs and decide rust/crate versions for tests."""
    if dedup_imports:
        code, test_code = deduplicate_imports(code, test_code)

    crate_names = extract_crate_names(code, test_code)
    has_third_party = len(crate_names) > 0

    if has_third_party:
        rust_version = validate_rust_version(DEFAULT_RUST_VERSION)
        crate_version = to_version if is_crate_version(to_version) else None
    else:
        rust_version = validate_rust_version(to_version)
        crate_version = None

    return code, test_code, rust_version, crate_version, has_third_party, crate_names


def deduplicate_imports(code: str, test_code: str) -> Tuple[str, str]:
    """Drop the `use` lines of the test code that the code already has."""
    code_imports = set()
    for line in code.split('\n'):
        line = line.strip()
        if line.startswith('use ') and line.endswith(';'):
            code_imports.add(line)
    
    if '#[cfg(test)]' in test_code:
        cfg_test_idx = test_code.find('#[cfg(test)]')
        before_cfg = test_code[:cfg_test_idx]
        after_cfg = test_code[cfg_test_idx:]
        
        filtered_lines = []
        for line in before_cfg.split('\n'):
            stripped = line.strip()
            if stripped.startswith('use ') and stripped.endswith(';'):
                if stripped in code_imports:
                    continue
            filtered_lines.append(line)
        
        cleaned_test_code = '\n'.join(filtered_lines) + after_cfg
        return code, cleaned_test_code
    
    return code, test_code


def create_test_file(code: str, test_program: str) -> str:
    """One source file: the code followed by its tests."""
    if "#[cfg(test)]" not in test_program:
        test_program = f"""
#[cfg(test)]
mod tests {{
    use super::*;
    
    {test_program}
}}
"""
    return f"{code}\n\n{test_program}"


def run_cargo_test(
    code: str,
    test_code: str,
    main_crate: str = None,
    rust_version: str = "1.84.0",
    crate_version: str = None,
    timeout: int = 120
) -> Dict:
    """Run the tests with `cargo test` (third-party crates are supported).
    
    Args:
        code: Rust code
        test_code: test code
        main_crate: crate of the API (see `extract_main_crate`)
        rust_version: Rust version
        crate_version: crate version (0.x.x) of `main_crate`
        timeout: seconds
        
    Returns:
        Dict: Test result
    """
    crate_names = extract_crate_names(code, test_code)
    
    code, test_code = deduplicate_imports(code, test_code)
    
    test_file_content = create_test_file(code, test_code)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            cargo_toml_content = "[package]\n"
            cargo_toml_content += "name = \"rust_test\"\n"
            cargo_toml_content += "version = \"0.1.0\"\n"
            cargo_toml_content += "edition = \"2021\"\n\n"
            cargo_toml_content += "[dependencies]\n"
            
            valid_crates = {c for c in crate_names if c in CRATE_FEATURES}
            
            try:
                _minor = int(rust_version.split('.')[1])
            except (IndexError, ValueError):
                _minor = 84
            _pre185 = _minor < 85
            
            for crate_name in sorted(valid_crates):
                features = CRATE_FEATURES.get(crate_name, [])
                
                if crate_name == main_crate and crate_version:
                    version = f"={crate_version}"
                elif _pre185 and crate_name in CRATE_MAX_VERSIONS_PRE_185:
                    version = CRATE_MAX_VERSIONS_PRE_185[crate_name]
                else:
                    version = "*"
                
                if features:
                    features_str = ', '.join(f'"{f}"' for f in features)
                    cargo_toml_content += f'{crate_name} = {{ version = "{version}", features = [{features_str}] }}\n'
                else:
                    cargo_toml_content += f'{crate_name} = "{version}"\n'
            
            cargo_toml_path = os.path.join(temp_dir, "Cargo.toml")
            with open(cargo_toml_path, 'w', encoding='utf-8') as f:
                f.write(cargo_toml_content)
            
            src_dir = os.path.join(temp_dir, "src")
            os.makedirs(src_dir, exist_ok=True)
            lib_rs_path = os.path.join(src_dir, "lib.rs")
            with open(lib_rs_path, 'w', encoding='utf-8') as f:
                f.write(test_file_content)
            
            uses_nightly_features = '#![feature' in test_file_content
            toolchain = 'nightly' if uses_nightly_features else rust_version
            
            test_cmd = f'cd "{temp_dir}" && rustup run {toolchain} cargo test --quiet'
            env = get_cargo_env(temp_dir=temp_dir)
            env["RUST_TEST_THREADS"] = "1"  # avoid data races on global panic hooks
            result = subprocess.run(
                test_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
            
            if result.returncode != 0:
                stderr = result.stderr if result.stderr else result.stdout
                error_type = 'compilation' if 'error[E' in stderr else 'test_failed'
                
                return {
                    'success': False,
                    'status': 'FAILED',
                    'error': stderr,
                    'error_type': error_type,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
            
            return {
                'success': True,
                'status': 'PASSED',
                'error': None,
                'error_type': None,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'status': 'TIMEOUT',
                'error': f'Test execution timeout after {timeout} seconds',
                'error_type': 'timeout',
                'stdout': '',
                'stderr': ''
            }
        except Exception as e:
            return {
                'success': False,
                'status': 'ERROR',
                'error': f'Error running cargo test: {str(e)}',
                'error_type': 'other',
                'stdout': '',
                'stderr': ''
            }


def run_rustc_test(
    code: str,
    test_code: str,
    rust_version: str = "1.84.0",
    timeout: int = 60
) -> Dict:
    """Run the tests with `rustc --test` (standard library only).
    
    Args:
        code: Rust code
        test_code: test code
        rust_version: Rust version
        timeout: seconds
        
    Returns:
        Dict: Test result
    """
    code, test_code = deduplicate_imports(code, test_code)
    
    test_file_content = create_test_file(code, test_code)
    
    with tempfile.NamedTemporaryFile(suffix='.rs', delete=False) as temp_file:
        temp_file_path = temp_file.name
        temp_file.write(test_file_content.encode('utf-8'))
    
    try:
        base_path = os.path.splitext(temp_file_path)[0]
        output_path = f"{base_path}.exe" if os.name == 'nt' else base_path
        
        uses_nightly_features = '#![feature' in test_file_content
        toolchain = 'nightly' if uses_nightly_features else rust_version
        
        compile_cmd = f'rustup run {toolchain} rustc --test "{temp_file_path}" -o "{output_path}"'
        compile_result = subprocess.run(
            compile_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if compile_result.returncode != 0:
            return {
                'success': False,
                'status': 'FAILED',
                'error': compile_result.stderr,
                'error_type': 'compilation',
                'stdout': compile_result.stdout,
                'stderr': compile_result.stderr
            }
        
        test_cmd = f'"{output_path}"'
        test_env = get_cargo_env()
        test_env["RUST_TEST_THREADS"] = "1"  # serialize libtest to avoid hook races
        test_result = subprocess.run(
            test_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=test_env,
        )
        
        if test_result.returncode != 0:
            return {
                'success': False,
                'status': 'FAILED',
                'error': test_result.stderr,
                'error_type': 'test_failed',
                'stdout': test_result.stdout,
                'stderr': test_result.stderr
            }
        
        return {
            'success': True,
            'status': 'PASSED',
            'error': None,
            'error_type': None,
            'stdout': test_result.stdout,
            'stderr': test_result.stderr
        }
        
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'status': 'TIMEOUT',
            'error': f'Test execution timeout after {timeout} seconds',
            'error_type': 'timeout',
            'stdout': '',
            'stderr': ''
        }
    except Exception as e:
        return {
            'success': False,
            'status': 'ERROR',
            'error': f'Error running rustc test: {str(e)}',
            'error_type': 'other',
            'stdout': '',
            'stderr': ''
        }
    finally:
        try:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            if os.path.exists(output_path):
                os.remove(output_path)
        except:
            pass


def validate_rust_version(version: str) -> str:
    """A Rust version inside the range of installed toolchains."""
    return normalize_rust_version(version or DEFAULT_RUST_VERSION)


def run_rust_test_auto(
    code: str,
    test_code: str,
    module: str = "",
    rust_version: str = "1.84.0",
    crate_version: str = None,
    timeout: int = 180
) -> Dict:
    """Run the tests with cargo (third-party crates) or rustc (standard library only).
    
    Args:
        code: Rust code
        test_code: test code
        module: module of the sample (names the crate)
        rust_version: Rust version
        crate_version: crate version (0.x.x)
        timeout: seconds
        
    Returns:
        Dict: Test result
    """
    rust_version = validate_rust_version(rust_version)
    
    crate_names = extract_crate_names(code, test_code)
    has_third_party = len(crate_names) > 0
    
    if has_third_party:
        main_crate = extract_main_crate(module, code)
        return run_cargo_test(code, test_code, main_crate, rust_version, crate_version, timeout)
    else:
        return run_rustc_test(code, test_code, rust_version, timeout)


def check_function_signature(code: str, signature: str) -> bool:
    """Whether the generated code defines the function of the signature (by name)."""
    if not signature or not code:
        return False
    
    clean_signature = re.sub(r'#\[.*?\]', '', signature)
    clean_signature = re.sub(r'///.*?\n', '', clean_signature)
    clean_signature = re.sub(r'//.*?\n', '', clean_signature)
    clean_signature = re.sub(r'pub\s+', '', clean_signature).strip()
    
    fn_match = re.search(r'fn\s+(\w+)', clean_signature)
    if not fn_match:
        return False
    
    fn_name = fn_match.group(1)
    
    if not re.search(r'fn\s+' + re.escape(fn_name) + r'\s*[(<]', code):
        return False
    
    return True


def check_api_usage(code: str, api_name: str, change_type: str = "", api_module: str = "", 
                    test_code: str = "", replacement_api: str = "") -> bool:
    """Whether the generated code uses the required API.
    
    - deprecated API with a `replacement_api`: the code must use the replacement and must not import the old API
    - any other API: the code must use `api_name`
    
    Recognised uses:
    1. a call: api_name()
    2. a trait: impl ApiName, T: ApiName
    3. an import: use module::ApiName
    4. snake_case <-> PascalCase variants (as_fd <-> AsFd)
    
    Args:
        code: the generated code
        api_name: name of the API (for a deprecated API: the old one)
        change_type: type of the API change (e.g. "deprecated")
        api_module: module of the API
        test_code: test code (not searched)
        replacement_api: name of the replacement API (deprecated APIs)
    
    Returns:
        bool: True if the code uses the API
    """
    if not code:
        return False

    all_code = code

    is_deprecated = str(change_type).lower() == "deprecated"

    if is_deprecated and replacement_api:
        target_api = replacement_api
        base_api_name = target_api.split('::')[-1] if '::' in target_api else target_api
    else:
        target_api = api_name
        base_api_name = api_name.split('::')[-1] if '::' in api_name else api_name
    
    if not target_api:
        return False
    
    if is_deprecated and replacement_api and api_module:
        old_api_name = api_name.split('::')[-1] if '::' in api_name else api_name
        old_use_patterns = [
            # use module::OldApiName;
            r'use\s+' + re.escape(api_module) + r'::' + re.escape(old_api_name) + r'\b',
            # use module::{..., OldApiName, ...};
            r'use\s+' + re.escape(api_module) + r'::\{[^}]*\b' + re.escape(old_api_name) + r'\b[^}]*\}',
        ]
        
        for pattern in old_use_patterns:
            if re.search(pattern, all_code):
                return False
    
    def to_pascal_case(s: str) -> str:
        """snake_case or lowercase to PascalCase: as_fd -> AsFd, clone -> Clone"""
        if '_' in s:
            return ''.join(word.capitalize() for word in s.split('_'))
        else:
            return s.capitalize()
    
    def to_snake_case(s: str) -> str:
        """PascalCase to snake_case: AsFd -> as_fd"""
        result = re.sub(r'([A-Z])', r'_\1', s).lower()
        return result.lstrip('_')
    
    name_variants = {base_api_name}
    pascal_variant = to_pascal_case(base_api_name)
    if pascal_variant != base_api_name and '_' in base_api_name:
        name_variants.add(pascal_variant)
    if any(c.isupper() for c in base_api_name):
        snake_variant = to_snake_case(base_api_name)
        if snake_variant != base_api_name:
            name_variants.add(snake_variant)

    full_names = [target_api]
    if api_module and '::' not in target_api:
        full_names.append(f"{api_module}::{target_api}")

    full_match = any(re.search(r'\b' + re.escape(name) + r'\b', all_code) for name in full_names)
    
    base_match = any(re.search(r'\b' + re.escape(variant) + r'\b', all_code) for variant in name_variants)
    
    trait_patterns = []
    for variant in name_variants:
        trait_patterns.extend([
            r'impl\s+' + re.escape(variant) + r'\b',  # impl AsFd
            r':\s*' + re.escape(variant) + r'\b',     # T: Clone
            r'dyn\s+' + re.escape(variant) + r'\b',   # dyn Trait
        ])
    trait_match = any(re.search(pattern, all_code) for pattern in trait_patterns)
    
    use_patterns = []
    for variant in name_variants:
        use_patterns.extend([
            r'use\s+.*::' + re.escape(variant) + r'\b',       # use std::os::fd::AsFd
            r'use\s+.*::\{[^}]*' + re.escape(variant),        # use std::os::fd::{AsFd, ...}
        ])
    use_match = any(re.search(pattern, all_code) for pattern in use_patterns)

    return bool(full_match or base_match or trait_match or use_match)
