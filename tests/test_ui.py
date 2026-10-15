"""The welcome page and data-clearing control must work without loading AI models."""
from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_welcome_and_repeated_clear_do_not_reload_previous_upload():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
    app.run(timeout=30)
    assert not app.exception
    assert "Understand your report" in app.title[0].value
    for expected_version in (1, 2):
        app.button[0].click().run(timeout=30)
        assert not app.exception
        assert app.session_state["uploader_version"] == expected_version
        assert "document" not in app.session_state
