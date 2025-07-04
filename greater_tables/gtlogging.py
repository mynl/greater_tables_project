"""
Manage logger setup for CLI. Users have the option of
calling this if they want similar logging.
"""

import logging
import sys

def setup_logging(level=logging.INFO):
    # Disable log propagation to prevent duplicates
    # logger.propagate = False
    root = logging.getLogger()
    if root.hasHandlers():
        root.handlers.clear()
    root.setLevel(level)
    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)

