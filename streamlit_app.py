# -*- coding: utf-8 -*-
"""Streamlit Cloud entry point.

Deploy settings:
  Main file: streamlit_app.py
  Requirements: requirements.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Dashboard UI runs on import (app/dashboard.py)
import app.dashboard  # noqa: F401
