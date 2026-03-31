"""
Decides whether to trigger a model retraining run.
Used by the weekly GitHub Action to check before spending compute time.

Exit code 0 + stdout "true"  → retrain needed
Exit code 0 + stdout "false" → skip retraining
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from monitoring.drift_detector import DriftDetector


def main():
    detector = DriftDetector()
    should, reason = detector.should_retrain()
    print("true" if should else "false")
    if should:
        import sys
        print(f"Reason: {reason}", file=sys.stderr)


if __name__ == "__main__":
    main()
