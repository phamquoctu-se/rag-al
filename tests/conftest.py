import pytest


# Ensure asyncio_mode is set globally
def pytest_configure(config):
    config.addinivalue_line("markers", "asyncio: mark test as async")
