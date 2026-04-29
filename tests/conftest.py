"""pytest configuration and shared fixtures."""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if item.get_closest_marker("known_fail"):
            item.add_marker(pytest.mark.xfail(strict=False, reason="known documented issue"))
