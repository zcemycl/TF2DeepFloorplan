import argparse

from .model import Model
from .view import View


class Controller:
    def __init__(self, config: argparse.Namespace):
        self.config = config
        self.model = Model(config)
        self.view = View(config)
