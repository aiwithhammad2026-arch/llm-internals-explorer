"""Pytest configuration and session fixtures.
By Hammad, AI Engineer.

Ensures clean separation between offline mocked unit tests and tests requiring the real local model.
"""

import pytest
from app.model_loader import model_manager

@pytest.fixture(autouse=True)
def load_real_model_only_if_marked(request):
    """Automatically ensures the model is loaded ONLY when a test is marked with @pytest.mark.real_model.
    Tests without this marker run completely offline without downloading or initializing model weights.
    """
    if request.node.get_closest_marker("real_model"):
        if not model_manager.is_loaded:
            model_manager.load_model("distilgpt2")
