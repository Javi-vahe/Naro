import os

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '4')

__version__ = '0.1.0'


def localize(*args, **kwargs):
    from .api import localize as run
    return run(*args, **kwargs)


def prepare(*args, **kwargs):
    from .api import prepare as run
    return run(*args, **kwargs)
