import tempfile
import unittest
from pathlib import Path

from tracker.base_tracker import Base3DTracker


class RequiredMotionCheckpointTest(unittest.TestCase):
    def test_missing_required_checkpoint_fails_before_tracker_initializes(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing_motion.pt"
            with self.assertRaisesRegex(FileNotFoundError, "REQUIRE_CHECKPOINT"):
                Base3DTracker({
                    "FILTER_MODE": "mamba_multihead_closure",
                    "MAMBA": {
                        "CHECKPOINT_PATH": str(missing),
                        "REQUIRE_CHECKPOINT": True,
                    },
                })


if __name__ == "__main__":
    unittest.main()
