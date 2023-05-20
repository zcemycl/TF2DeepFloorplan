import argparse
from argparse import Namespace

from dynaconf import Dynaconf


def overwrite_args_with_toml(config: argparse.Namespace) -> argparse.Namespace:
    if config.tomlfile is None:
        return config
    settings = Dynaconf(
        envvar_prefix="DYNACONF", settings_files=[config.tomlfile]
    )
    settings = dict((k.lower(), v) for k, v in settings.as_dict().items())
    return Namespace(**settings)
