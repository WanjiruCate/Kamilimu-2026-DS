"""API tests use a copy of the artifacts so online updates do not touch the trained file."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "service"))

SOURCE = ROOT / "artifacts"
DEST = ROOT / "work" / "test_artifacts"


def pytest_configure(config):
    if not (SOURCE / "model.joblib").exists():
        return
    if DEST.exists():
        shutil.rmtree(DEST)
    shutil.copytree(SOURCE, DEST)
    os.environ["ARTIFACT_DIR"] = str(DEST)


@pytest.fixture(scope="session")
def client():
    if not (SOURCE / "model.joblib").exists():
        pytest.skip("model artifacts are not built yet")
    from fastapi.testclient import TestClient

    import api

    with TestClient(api.app) as test_client:
        yield test_client
