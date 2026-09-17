# PTB-XL+ feature boundary

| Source | Role | Available for a new submitted ECG? |
|---|---|---|
| Local WFDB / SciPy / NeuroKit2 extractor | Production candidate | Yes |
| PTB-XL+ ECGDeli measurements | Open reference and agreement study | Only if the same local extraction dependency is deployed; database values themselves are not inference inputs |
| PTB-XL+ 12SL | Oracle/reference benchmark | No; commercial closed extractor |
| PTB-XL+ Uni-G | Oracle/reference benchmark | No; commercial closed extractor |

The join accepts only canonical fields listed by `feature_description.csv`. Labels, statements, report text, SCP codes, infarction stage, validation status and post-diagnosis outputs are denied. Imputation, scaling, redundancy filtering and feature selection must be fitted inside each development training fold.
