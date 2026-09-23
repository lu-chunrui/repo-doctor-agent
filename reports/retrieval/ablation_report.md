# Retrieval Ablation Report

| Method | Hit@1 | Hit@3 | Hit@5 | MRR | Latency ms | Rewrite ms | Index s | Irrelevant FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 0.2800 | 0.5200 | 0.6400 | 0.4080 | 1.3235 | - | 6.6612 | 1.0000 |
| Dense | 0.4400 | 0.7200 | 0.8000 | 0.5713 | 18.7288 | - | 21.1205 | 1.0000 |
| Hybrid + RRF | 0.4800 | 0.7200 | 0.8800 | 0.6180 | 21.0874 | - | 27.6297 | 1.0000 |
| Dense + Rewrite | 0.6000 | 0.7600 | 0.9200 | 0.7027 | 8139.9603 | 8125.7277 | 21.0494 | 0.0000 |
| Hybrid + Rewrite | 0.5600 | 0.8000 | 0.8800 | 0.6800 | 8176.5737 | 8160.4793 | 27.7835 | 0.0000 |
| Hybrid + Rewrite + Reranker | 0.6800 | 0.8800 | 0.9600 | 0.7867 | 11796.5120 | 6725.3352 | 27.7434 | 0.0000 |

## Current Best

`Hybrid + Rewrite + Reranker`

## Important Notes

- 当前查询数量较少，结论只是阶段性结果。
- Query Rewrite 方案包含真实 LLM 调用延迟。
- Reranker 结果只有在 reranker_available=true 时有效。
- 无关查询误检率越低越好。
