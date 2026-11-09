# Fixed-30k Original versus hybrid B on H200 CUDA

Both seed-1234 runs completed 30,000 updates with passed independent endpoint audits.

| Model | EPE, px | bad3, percent |
|---|---:|---:|
| Original non-I | 2.166783 | 10.219583 |
| Hybrid B | 2.289049 | 9.795966 |

Original-minus-hybrid EPE: -0.122266 px; relative reduction -5.643%. Hybrid wins 16/30 scenes; median paired difference 0.013316 px. Original-minus-hybrid bad3: 0.423617 percentage points.

All scenes, including large errors, are retained in scene-deltas.csv. Full planned Dev curves are in curves.csv. The five largest absolute paired scene gaps account for 83.45% of the total absolute gap and contribute -0.141957 px to the overall signed mean difference. The five hardest scenes ranked by Original endpoint EPE contribute -0.009423 px; their identities and all retained scenes are in comparison.json and scene-deltas.csv.

| Scene (five largest absolute gaps) | Original EPE | Hybrid B EPE | Original-minus-B |
|---|---:|---:|---:|
| scene_0134 | 10.236000 | 4.414094 | 5.821907 |
| scene_0142 | 3.798411 | 9.586338 | -5.787927 |
| scene_0135 | 2.598544 | 5.379901 | -2.781357 |
| scene_0144 | 4.803294 | 5.733756 | -0.930462 |
| scene_0125 | 2.515379 | 3.096241 | -0.580862 |

 One seed does not establish robust superiority; the comparison concerns the entire B addition and does not isolate spiking.

Historical CUDA values from spikestereonet-npu-baseline.md: Original EPE 2.691127 / bad3 10.712%; hybrid B EPE 2.034117 / bad3 9.626333%. They are separately labeled context in historical-cuda-comparison.csv. The reference CUDA values are historical aggregates; the two H200 endpoints are fresh native CUDA runs. No same-weight cross-backend prediction/gradient comparison was performed. All primary H200 CUDA comparisons use the fixed 30k endpoint, not a best interim checkpoint. No additional seeds or controls are launched by this analysis.
