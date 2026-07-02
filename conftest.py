"""Pytest configuration — ensures src/ is on the path and prevents
the ComfyUI entry-point __init__.py from being imported during tests.
"""
import sys
import os

# Add src/ to path so hand_analyzer package is importable
src_path = os.path.join(os.path.dirname(__file__), "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Add the project root so nodes/ is importable as a package
project_root = os.path.dirname(__file__)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Collect tests only from the tests/ directory
collect_ignore = ["__init__.py", "nodes", "src", "docs", "workflows", ".venv"]