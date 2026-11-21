# Evaluation summary

72 seeded datasets (4 noise types x 6 rates x 3 seeds), 420 generated points plus planted near-duplicates each, planted ground truth.

## Noise-type classification accuracy

| noise type | n | fingerprinter | matrix baseline |
|---|---|---|---|
| uniform | 18 | 0.722 | 0.000 |
| class_dependent | 18 | 0.778 | 1.000 |
| instance_dependent | 18 | 0.667 | 0.000 |
| boundary | 18 | 0.778 | 0.000 |
| **overall** | 72 | **0.736** | **0.250** |

## Rate estimation

Mean absolute error of the estimated noise rate vs the planted rate: **0.046** (fingerprinter; the matrix baseline does not estimate a type-specific rate).

| noise type | rate MAE |
|---|---|
| uniform | 0.037 |
| class_dependent | 0.036 |
| instance_dependent | 0.050 |
| boundary | 0.061 |

## Diagnosis confusion (true -> predicted counts)

- uniform -> class_dependent: 4, instance_dependent: 1, uniform: 13
- class_dependent -> class_dependent: 14, instance_dependent: 2, uniform: 2
- instance_dependent -> boundary: 4, class_dependent: 2, instance_dependent: 12
- boundary -> boundary: 14, class_dependent: 1, instance_dependent: 3
