# Soft Fuse Calibration Report

- Query count: 30
- Minimum recall requirement: 0.9500
- Recommended threshold: `0.869127`
- Relevant recall: 1.0000
- Irrelevant false-positive rate: 0.3333
- Accuracy: 0.8333
- Average Dense latency: 20.65 ms

## Confusion Matrix

| TP | FP | TN | FN |
| -: | -: | -: | -: |
| 15 | 5 | 10 | 0 |

## Score Distribution

| Group | Min | P25 | Median | P75 | Max | Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Relevant | 0.8691 | 0.8808 | 0.8961 | 0.9035 | 0.9137 | 0.8917 |
| Irrelevant | 0.8171 | 0.8555 | 0.8680 | 0.8726 | 0.8809 | 0.8615 |

## Recommended Environment

```dotenv
SOFT_FUSE_ENABLED=true
SOFT_FUSE_SIMILARITY_THRESHOLD=0.869127
```
