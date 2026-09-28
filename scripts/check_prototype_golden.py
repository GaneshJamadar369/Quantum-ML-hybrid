#!/usr/bin/env python3
"""Verify a frozen bundle and execute its signed end-to-end golden fixtures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aquire_preprocessing.production_bundle import FrozenHybridBundle
from aquire_preprocessing.prototype_bundle import BundleError, verify_bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--tolerance", type=float, default=2e-4)
    args = parser.parse_args()
    try:
        bundle = verify_bundle(args.bundle)
        result = FrozenHybridBundle(bundle).golden_self_test(tolerance=args.tolerance)
    except (BundleError, RuntimeError, ValueError) as error:
        print(json.dumps({"valid": False, "error": str(error)}, indent=2))
        return 1
    print(
        json.dumps(
            {
                "valid": True,
                "model_version": bundle.model_version,
                **result,
                "root": str(bundle.root),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
