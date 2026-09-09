"""Allow `python -m tests` to run."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from tests.run_all import main
sys.exit(main())