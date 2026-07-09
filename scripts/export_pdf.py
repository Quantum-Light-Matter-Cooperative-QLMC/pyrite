"""notebooks/analysis_app.py -> results/<stem>.html -- thin shim to cxr_mc.export.

Kept in scripts/ for muscle-memory ``python scripts/export_pdf.py [stem]``; prefer
the installed CLI ``cxr export [stem]``. The real logic lives in cxr_mc/export.py
(now a ``marimo export html`` of the analysis app -- the old nbconvert-PDF path
died with analysis.ipynb in the marimo migration).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from cxr_mc.export import main

if __name__ == "__main__":
    main()
