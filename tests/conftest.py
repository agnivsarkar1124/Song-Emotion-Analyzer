import pytest

import semantic


@pytest.fixture(autouse=True)
def keyword_search_by_default():
    """Keep rankings deterministic: semantic search is off unless a test installs an encoder."""
    semantic.set_encoder(False)
    yield
    semantic.set_encoder(None)
