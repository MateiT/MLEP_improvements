"""Put src/ on sys.path so the tests run without `pip install -e .`.

Both runners rely on this: `pytest tests/` picks conftest.py up automatically,
and the plain-script path (`python tests/test_x.py`, which is what ./run.sh test
uses) imports it explicitly at the top of each test file.
"""
import os
import sys

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)
