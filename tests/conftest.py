"""Pytest configuration.

Redirects the modeling workspace to a throwaway temp directory BEFORE any
``modeling.*`` import so tests never read or write the real ``src/artifacts``.
"""
import os
import pathlib
import tempfile

_WORKSPACE = pathlib.Path(__file__).resolve().parent.parent

os.environ.setdefault("RISK_DATA_DIR", str(_WORKSPACE / "src" / "dataset"))
os.environ["RISK_ARTIFACT_DIR"] = tempfile.mkdtemp(prefix="continual-test-")
