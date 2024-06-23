"""Test the GUI."""

from tpms_gui import Gui


def test_gui_run() -> None:
    """Test that the GUI can be run."""
    gui = Gui()
    gui.run(open_browser=False, timeout=1)
