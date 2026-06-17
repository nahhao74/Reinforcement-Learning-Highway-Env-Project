# highway_env/envs/__init__.py

# Import available env classes in a robust way (compatible with partial/modified modules)
from importlib import import_module

_envs = {}
try:
    mod = import_module("highway_env.envs.highway_env")
    HighwayEnv = getattr(mod, "HighwayEnv")
    _envs["HighwayEnv"] = HighwayEnv
    HighwayEnvFast = getattr(mod, "HighwayEnvFast", HighwayEnv)  # fallback to HighwayEnv if missing
    _envs["HighwayEnvFast"] = HighwayEnvFast
except Exception:
    # If module cannot be imported, leave names undefined; errors will appear later when used
    pass

# Optionally expose names
__all__ = list(_envs.keys())
globals().update(_envs)