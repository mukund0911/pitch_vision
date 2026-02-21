"""
YAML config loader with attribute-style access.
"""

import yaml
from pathlib import Path


class Config(dict):
    """Dict subclass with attribute access for nested config values."""

    def __getattr__(self, key):
        try:
            val = self[key]
            if isinstance(val, dict):
                return Config(val)
            return val
        except KeyError:
            raise AttributeError(f"Config has no key '{key}'")

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


def load_config(path: str) -> Config:
    """Load a YAML config file and return a Config object."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        data = yaml.safe_load(f)
    return Config(data)
