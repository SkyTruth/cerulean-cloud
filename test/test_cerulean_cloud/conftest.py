"""
pytest configuration for the cerulean_cloud package.
"""

import os
import sys

# add stack path to enable relative imports from stack
sys.path.append(os.path.join(os.path.abspath("."), "cerulean_cloud"))
