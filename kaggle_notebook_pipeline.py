"""Thin Kaggle driver.

All scientific logic lives in ``aquire_preprocessing``. Set the dataset roots
in environment variables when Kaggle slugs differ from the defaults:

    AQUIRE_PTBXL_ROOT=/kaggle/input/.../ptb-xl-1.0.3
    AQUIRE_PTBXLP_ROOT=/kaggle/input/.../ptb-xl-plus-1.0.1

This file intentionally contains no alternative label, QC, filter, morphology,
or persistence implementation.
"""

from run_pipeline import main

if __name__ == "__main__":
    main()
