# NPU B factorial results

Audit status: **completed**.

Positive contrasts mean lower EPE. Endpoint is fixed at 30,000 updates.

| Seed | Variant | EPE | bad3 (%) |
| --- | --- | ---: | ---: |
| 1234 | linear-identity | 2.200676 | 10.3280 |
| 1234 | signed-surrogate | 2.559591 | 10.6305 |
| 1234 | linear-surrogate | 2.690989 | 10.5594 |
| 1234 | signed-identity | 2.252443 | 10.3025 |
| 2345 | linear-identity | 2.380936 | 10.6830 |
| 2345 | signed-surrogate | 2.092714 | 10.6642 |
| 2345 | linear-surrogate | 2.211694 | 10.5898 |
| 2345 | signed-identity | 2.315693 | 10.0954 |

| Contrast | Seed 1234 | Seed 2345 | Screen |
| --- | ---: | ---: | --- |
| hybrid_package | -0.358915 | 0.288222 | inconclusive / no consistent practical gain |
| forward_at_identity | -0.051767 | 0.065243 | inconclusive / no consistent practical gain |
| forward_at_surrogate | 0.131398 | 0.118980 | positive in both (≥0.05 px) |
| surrogate_at_linear | -0.490313 | 0.169242 | inconclusive / no consistent practical gain |
| surrogate_at_signed | -0.307148 | 0.222979 | inconclusive / no consistent practical gain |
| interaction | 0.183165 | 0.053737 | positive in both (≥0.05 px) |

These are two-seed training-intervention effects, not proof of uniquely discrete-spike benefit.
Inspect per-scene deltas and bad3 before drawing a conclusion. Audit failures invalidate interpretation.
