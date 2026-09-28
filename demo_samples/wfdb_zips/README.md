# WFDB ZIP demonstration inputs

Each ZIP contains one matching WFDB `.hea`/`.dat` pair with a de-identified,
10-second, 12-lead ECG sampled at 100 Hz in millivolts. The files can be
uploaded directly in the AQUIRE-Med clinician interface.

| File | Reference label | Expected screen | Approx. probability |
|---|---|---|---:|
| `demo_01_mi_pattern.zip` | MI pattern | MI pattern | 70.02% |
| `demo_02_non_mi_pattern.zip` | Non-MI pattern | Non-MI pattern | 0.87% |
| `demo_03_borderline_non_mi.zip` | Non-MI pattern | Non-MI pattern | 39.68% |
| `demo_04_false_positive_challenge.zip` | Non-MI pattern | MI pattern | 70.47% |

The fourth case is intentionally included to demonstrate a false-positive
screen and why clinician review remains necessary. Probabilities are golden
fixture expectations for software verification, not independent performance
evidence.

Source: PTB-XL v1.0.3, released under CC BY 4.0. Preserve source attribution
and license terms when redistributing these derived records.
