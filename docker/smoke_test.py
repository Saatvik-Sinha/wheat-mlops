"""Container smoke test: proves the image has a working Python environment.

Runs inside the container (docker run, and the Kubernetes Job). Exit code 0 = healthy.
"""

import importlib
import os
import platform
import sys

MODULES = ["wheat_mlops", "numpy", "pandas", "cv2", "torch", "ultralytics"]


def main() -> int:
    failed = []
    for name in MODULES:
        try:
            module = importlib.import_module(name)
            print(f"ok    {name} {getattr(module, '__version__', '')}".rstrip())
        except Exception as exc:  # noqa: BLE001 - we want to report any import problem
            print(f"FAIL  {name}: {exc!r}")
            failed.append(name)

    if "torch" not in failed:
        import torch

        cuda = torch.version.cuda
        avail = torch.cuda.is_available()
        print(f"info  torch cuda build={cuda} cuda_available={avail}")

    if not os.path.isfile("params.yaml"):
        print(f"FAIL  params.yaml not found in {os.getcwd()}")
        failed.append("params.yaml")

    print(f"info  python={platform.python_version()} uid={os.getuid()} cwd={os.getcwd()}")
    if failed:
        print(f"SMOKE TEST FAILED: {failed}")
        return 1
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
