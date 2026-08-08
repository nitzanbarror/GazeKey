"""Pytest bootstrap: put the repository root on sys.path.

Keeps `import gaze.*`, `import vision.*` working when pytest is invoked from
anywhere inside the project.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Qt widget tests render into pixmaps; never pop a real window during a test run.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
