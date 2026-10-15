from pathlib import Path
import site
import sys


def share_dir():
    source = Path(__file__).resolve().parents[1]
    if (source / 'examples' / 'cases.json').is_file():
        return source
    try:
        from ament_index_python.packages import get_package_share_directory
        return Path(get_package_share_directory('naro'))
    except (ImportError, LookupError):
        pass
    for prefix in (sys.prefix, site.USER_BASE):
        installed = Path(prefix) / 'share' / 'naro'
        if (installed / 'examples' / 'cases.json').is_file():
            return installed
    raise FileNotFoundError('naro example data was not installed')


def load_case(name, root=None):
    import json
    root = Path(root) if root is not None else share_dir() / 'examples'
    cases = json.loads((root / 'cases.json').read_text(encoding='utf-8'))['cases']
    try:
        return root, next(case for case in cases if case['id'] == name)
    except StopIteration:
        raise ValueError('unknown case; choose ' + ', '.join(case['id'] for case in cases)) from None
