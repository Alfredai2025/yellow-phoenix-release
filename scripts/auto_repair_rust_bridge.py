#!/usr/bin/env python3
"""Auto-generate missing RustBridge bindings from yp_engine.py call sites.

Scans yp_engine.py for self.rust.XXX(...) and self._bridge.XXX(...) calls,
maps them to dylib symbols (yp_XXX or special cases), and injects simple
*args wrapper methods into RustBridge in yp_bridge.py.
"""
import re
import ctypes
from pathlib import Path

ROOT = Path('/Users/mac/yellow_phoenix')
ENGINE_PATH = ROOT / 'yp_engine.py'
BRIDGE_PATH = ROOT / 'yp_bridge.py'
DYLIB_PATH = ROOT / 'target' / 'release' / 'libpams.dylib'

# Special cases where Python method name does not map to yp_<name>
NAME_MAP = {
    'spectral_512_build': 'yp_tensor_spectral_512_build',
    'spectral_512_query': 'yp_tensor_spectral_512_query',
}


def find_calls(src: str) -> set:
    """Find all self.rust.NAME( or self._bridge.NAME( call names."""
    calls = set(re.findall(r'self\.(?:rust|_bridge)\.([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', src))
    # Drop obvious Python builtins / attributes
    builtin_like = {
        'print', 'len', 'range', 'str', 'int', 'float', 'bool', 'list', 'dict', 'set', 'tuple',
        'open', 'close', 'read', 'write', 'join', 'append', 'extend', 'pop', 'get', 'items',
        'keys', 'values', 'encode', 'decode', 'strip', 'split', 'replace', 'format', 'time',
        'sleep', 'os', 'sys', 'path', 'exists', 'isfile', 'isdir', 'mkdir', 'makedirs',
        'remove', 'rename', 'copy', 'move', 'shutil', 'hashlib', 'sha256', 'hexdigest',
        'random', 'choice', 'shuffle', 'seed', 'np', 'numpy', 'array', 'zeros', 'ones',
        'empty', 'arange', 'linspace', 'reshape', 'dot', 'matmul', 'transpose', 'mean',
        'std', 'sum', 'max', 'min', 'argmax', 'argmin', 'argsort', 'sort', 'searchsorted',
        'clip', 'where', 'any', 'all', 'isin', 'unique', 'concatenate', 'stack', 'vstack',
        'hstack', 'dstack', 'split', 'hsplit', 'vsplit', 'dsplit', 'tile', 'repeat', 'pad',
        'resize', 'flatten', 'ravel', 'squeeze', 'expand_dims', 'swapaxes', 'moveaxis',
        'rollaxis', 'broadcast_to', 'broadcast_arrays', 'meshgrid', 'ogrid', 'mgrid', 'ix_',
        'fill_diagonal', 'diag', 'diagflat', 'tri', 'tril', 'triu', 'vander', 'histogram',
        'histogram2d', 'histogramdd', 'bincount', 'digitize', 'piecewise', 'select', 'interp',
        'griddata', 'map_coordinates', 'distance_transform_edt', 'label', 'find_objects',
        'count_nonzero', 'nonzero', 'flatnonzero', 'argwhere', 'extract', 'place', 'json',
        'loads', 'dumps', 'pickle', 'torch', 'nn', 'Variable', 'cuda', 'device', 'cpu',
        'load_state_dict', 'state_dict', 'eval', 'no_grad', 'forward', 'backward', 'parameters',
        'named_parameters', 'modules', 'named_modules', 'children', 'named_children', 'train',
        'to', 'type', 'size', 'shape', 'view', 'unsqueeze', 'squeeze', 'detach', 'numpy',
        'item', 'tolist', 'clone', 'copy', 'zero_grad', 'step', 'optimizer', 'scheduler',
        'criterion', 'loss', 'backward', 'optimizer_step', 'load', 'save', 'dump', 'loads',
        'dumps', 'packbits', 'unpackbits', 'frombuffer', 'tobytes', 'astype', 'ctypes',
        'data_as', 'cumsum', 'count', 'bit_count', 'bit_length', 'hex', 'fromhex', 'upper',
        'lower', 'startswith', 'endswith', 'find', 'index', 'count', 'isdigit', 'isalpha',
        'isalnum', 'isspace', 'islower', 'isupper', 'istitle', 'capitalize', 'title',
        'lstrip', 'rstrip', 'center', 'ljust', 'rjust', 'zfill', 'expandtabs', 'translate',
        'maketrans', 'partition', 'rpartition', 'rsplit', 'splitlines', 'join', 'format_map',
        'removeprefix', 'removesuffix', '__init__', 'super', 'hasattr', 'getattr', 'setattr',
        'delattr', 'isinstance', 'issubclass', 'type', 'id', 'repr', 'dir', 'vars', 'locals',
        'globals', 'eval', 'exec', 'compile', '__import__', 'breakpoint', 'help', 'input',
        'quit', 'exit', 'copyright', 'credits', 'license',
    }
    return calls - builtin_like


def rust_symbol(call_name: str) -> str:
    return NAME_MAP.get(call_name, f'yp_{call_name}')


def find_existing_methods(src: str) -> set:
    """Find methods already defined in RustBridge class."""
    methods = set()
    # Simple class parser: find class RustBridge and collect def names until next class
    in_class = False
    for line in src.splitlines():
        if line.startswith('class RustBridge'):
            in_class = True
            continue
        if in_class and line.startswith('class '):
            break
        if in_class:
            m = re.match(r'    def ([a-zA-Z_][a-zA-Z0-9_]*)\(', line)
            if m:
                methods.add(m.group(1))
    return methods


def generate_method(call_name: str, symbol: str, present: bool) -> str:
    if not present:
        return (
            f"    def {call_name}(self, *args, **kwargs):\n"
            f"        raise RuntimeError(\"Rust symbol '{symbol}' not found in libpams.dylib\")\n"
        )
    return (
        f"    def {call_name}(self, *args):\n"
        f"        return getattr(self.lib, '{symbol}')(*args)\n"
    )


def main():
    engine_src = ENGINE_PATH.read_text()
    bridge_src = BRIDGE_PATH.read_text()

    calls = find_calls(engine_src)
    existing = find_existing_methods(bridge_src)
    missing = sorted(calls - existing)

    print(f'Calls in yp_engine.py: {len(calls)}')
    print(f'Already in RustBridge: {len(existing & calls)}')
    print(f'Missing from RustBridge: {len(missing)}')
    for c in missing:
        print(f'  - {c}')

    if not missing:
        print('Nothing to patch.')
        return

    # Load dylib to check symbol presence
    try:
        lib = ctypes.CDLL(str(DYLIB_PATH))
    except OSError as e:
        print(f'Cannot load dylib: {e}')
        return

    methods = []
    for call_name in missing:
        symbol = rust_symbol(call_name)
        present = hasattr(lib, symbol)
        methods.append(generate_method(call_name, symbol, present))
        status = 'OK' if present else 'MISSING'
        print(f'  [{status}] {call_name} -> {symbol}')

    block = '\n'.join(methods)
    marker = '\n    # ── Geometric Structure: HolographicCascade FFI ──\n'

    # Insert generated block right before the holographic cascade section,
    # or before _stub_results if that section is absent.
    if marker in bridge_src:
        new_src = bridge_src.replace(marker, '\n    # ── Auto-generated FFI wrappers from yp_engine.py ──\n' + block + '\n' + marker)
    else:
        # Fallback: insert before _stub_results
        new_src = bridge_src.replace(
            '    def _stub_results(self, query: str, k: int) -> List[dict]:',
            '    # ── Auto-generated FFI wrappers from yp_engine.py ──\n' + block + '\n    def _stub_results(self, query: str, k: int) -> List[dict]:'
        )

    # Backup
    backup = BRIDGE_PATH.with_suffix('.py.bak.autobind')
    backup.write_text(bridge_src)
    print(f'Backup written to {backup}')

    BRIDGE_PATH.write_text(new_src)
    print(f'Patched {BRIDGE_PATH}')

    # Quick compile check
    import py_compile
    py_compile.compile(str(BRIDGE_PATH), doraise=True)
    print('Compile check OK')


if __name__ == '__main__':
    main()
