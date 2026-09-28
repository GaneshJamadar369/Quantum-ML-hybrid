# Demonstration ECG inputs

These files are de-identified 10-second, 12-lead, 100 Hz ECG records extracted
from the signed Fold-9 golden fixture for local prototype verification.

| File | Reference label | Expected prototype probability |
|---|---|---:|
| `sample_1_mi_pattern.csv` | MI pattern | about 70.02% |
| `sample_2_non_mi.csv` | Non-MI pattern | about 0.87% |

Each CSV has 1,000 rows and the canonical lead columns `I`, `II`, `III`,
`aVR`, `aVL`, `aVF`, `V1`--`V6`, with voltage values in mV. The samples are
for software demonstration and are not independent test evidence or clinical
cases.

Source: Wagner et al., [PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/)
([dataset paper](https://doi.org/10.1038/s41597-020-0495-6)), released under the
[Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/).
Preserve the source attribution and license terms when redistributing these
derived files.
