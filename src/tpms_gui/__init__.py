"""TPMS App."""

from .gui import Gui

def main():
    Gui().run()

__all__ = ["Gui"]