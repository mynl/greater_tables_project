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


# # Disable log propagation to prevent duplicates
# logger.propagate = False
# if logger.hasHandlers():
#     # Clear existing handlers
#     logger.handlers.clear()
# # SET DEGBUGGER LEVEL
# LEVEL = logging.INFO    # DEBUG or INFO, WARNING, ERROR, CRITICAL
# logger.setLevel(LEVEL)
# handler = logging.StreamHandler(sys.stderr)
# handler.setLevel(LEVEL)
# formatter = logging.Formatter(
#     '%(asctime)s | %(levelname)s |  %(funcName)-15s | %(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)
# logger.info(f'Logger Setup; {__name__} module recompiled.')
