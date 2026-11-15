import os
import sys

from ledgerlight.common import ROOT

os.execv(sys.executable, [sys.executable, str(ROOT / "scripts/site.py"), *sys.argv[1:]])
