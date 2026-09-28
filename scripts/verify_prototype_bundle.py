#!/usr/bin/env python3
"""Verify an AQUIRE-Med prototype bundle before serving it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aquire_preprocessing.prototype_bundle import BundleError, verify_bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--allow-uncalibrated", action="store_true")
    args = parser.parse_args()
    try:
        bundle = verify_bundle(args.bundle, allow_uncalibrated=args.allow_uncalibrated)
    except BundleError as error:
        print(json.dumps({"valid": False, "error": str(error)}, indent=2))
        return 1
    print(
        json.dumps(
            {
                "valid": True,
                "model_version": bundle.model_version,
                "calibrated": bundle.calibrated,
                "artifact_count": len(bundle.artifacts),
                "root": str(bundle.root),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
