#!/usr/bin/env python3
"""Isolate NumPy ABI failures without importing extensions in the parent process."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys


PROBES = {
    "numpy": "import numpy as np; np.testing.assert_array_equal(np.arange(3), [0, 1, 2])",
    "torch": "import numpy as np, torch; a=np.arange(4, dtype=np.float32); t=torch.from_numpy(a); np.testing.assert_array_equal(t.numpy(), a)",
    "scipy": "import numpy as np; from scipy.linalg import solve; np.testing.assert_allclose(solve(np.eye(2), np.ones(2)), [1, 1])",
    "cv2": "import numpy as np, cv2; assert cv2.Rodrigues(np.zeros(3))[0].shape == (3, 3)",
    "lap": "import numpy as np, lap; assert lap.lapjv(np.eye(2))[0] == 0",
    "pandas": "import numpy as np, pandas as pd; assert pd.DataFrame(np.ones((2,2))).to_numpy().shape == (2,2)",
    "matplotlib": "import matplotlib; matplotlib.use('Agg'); from matplotlib.figure import Figure; from matplotlib.backends.backend_agg import FigureCanvasAgg; f=Figure(); f.subplots().plot([0,1]); FigureCanvasAgg(f).draw()",
    "numba": "import numba; assert numba.njit(lambda x: x + 1)(1) == 2",
    "nuscenes": "from nuscenes.nuscenes import NuScenes; from nuscenes.eval.tracking.evaluate import TrackingEval",
    "motmetrics": "import numpy as np, motmetrics as mm; a=np.array([[0.,0.,1.,1.]]); np.testing.assert_allclose(mm.distances.iou_matrix(a,a), [[0.]]); np.testing.assert_allclose(mm.distances.norm2squared_matrix(a,a), [[0.]])",
    "mamba_ssm": "from mamba_ssm import Mamba2",
}
PACKAGES = ("numpy", "torch", "scipy", "opencv-python", "opencv-python-headless",
            "lap", "pandas", "matplotlib", "numba", "nuscenes-devkit", "motmetrics",
            "mamba-ssm", "causal-conv1d")
ABI_MARKERS = ("compiled using NumPy 1.x", "_ARRAY_API not found", "numpy.dtype size changed",
               "numpy.core.multiarray failed to import", "Failed to initialize NumPy",
               "Numpy is not available", "compiled against API version")


def classify_result(returncode, stdout, stderr):
    combined = stdout + "\n" + stderr
    if any(marker.lower() in combined.lower() for marker in ABI_MARKERS):
        return "numpy_abi_failure"
    return "ok" if returncode == 0 else "failed"


def run_probe(name, timeout):
    try:
        process = subprocess.run([sys.executable, "-c", PROBES[name]],
                                 capture_output=True, text=True, timeout=timeout)
        return {"module": name, "status": classify_result(process.returncode, process.stdout, process.stderr),
                "returncode": process.returncode, "stdout": process.stdout, "stderr": process.stderr}
    except subprocess.TimeoutExpired:
        return {"module": name, "status": "timeout", "timeout_seconds": timeout}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modules", nargs="+", choices=list(PROBES), default=list(PROBES))
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--output", help="Optional JSON report, including full extension tracebacks")
    args = parser.parse_args(argv)
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    report = {"executable": sys.executable, "python": sys.version, "versions": versions, "probes": []}
    print(f"Python: {sys.executable}\n{sys.version}\nVersions: {json.dumps(versions)}", flush=True)
    for name in args.modules:
        result = run_probe(name, args.timeout)
        report["probes"].append(result)
        print(f"[{result['status']}] {name}", flush=True)
        if result["status"] != "ok":
            print(result.get("stdout", "") + result.get("stderr", ""), flush=True)
    try:
        check = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True,
                               text=True, timeout=args.timeout)
        report["pip_check"] = {"returncode": check.returncode, "stdout": check.stdout, "stderr": check.stderr}
    except subprocess.TimeoutExpired:
        report["pip_check"] = {"returncode": -1, "stderr": "pip check timed out"}
    print("pip check:", report["pip_check"], flush=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return int(any(row["status"] != "ok" for row in report["probes"])
               or report["pip_check"]["returncode"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
