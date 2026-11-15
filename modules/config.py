from types import MappingProxyType

def make_params(e=1.0, lam=4.0, v=1.0, n=1):
    return MappingProxyType({'e': e, 'lam': lam, 'v': v, 'n': n})

def update_params(params, **new):
    return MappingProxyType({**params, **new})

VORTEX_DEFAULT = make_params()
