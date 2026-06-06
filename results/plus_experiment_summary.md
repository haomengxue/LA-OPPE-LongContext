# B 类期刊进一步补充实验结果

## 1. 8192 PPL / 滑窗 PPL

| method | seq_len | count | mean_ppl | std_ppl | min_ppl | max_ppl | eval_modes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RoPE | 8192.0000 | 50.0000 | 10.5157 | 1.3165 | 8.0318 | 12.9247 | full |
| YaRN_x2 | 8192.0000 | 50.0000 | 10.5235 | 1.3154 | 8.0109 | 12.8517 | full |
| LAOPPE_L2 | 8192.0000 | 50.0000 | 8.5646 | 1.0417 | 6.5015 | 10.5477 | full |

## 2. YaRN Needle 对比

| method | seq_len | success | total | accuracy |
| --- | --- | --- | --- | --- |
| RoPE | 2048.0000 | 17.0000 | 20.0000 | 0.8500 |
| RoPE | 4096.0000 | 15.0000 | 20.0000 | 0.7500 |
| RoPE | 8192.0000 | 18.0000 | 20.0000 | 0.9000 |
| YaRN_x2 | 2048.0000 | 16.0000 | 20.0000 | 0.8000 |
| YaRN_x2 | 4096.0000 | 4.0000 | 20.0000 | 0.2000 |
| YaRN_x2 | 8192.0000 | 6.0000 | 20.0000 | 0.3000 |
| NTK_dynamic_x2 | 2048.0000 | 20.0000 | 20.0000 | 1.0000 |
| NTK_dynamic_x2 | 4096.0000 | 12.0000 | 20.0000 | 0.6000 |
| NTK_dynamic_x2 | 8192.0000 | 8.0000 | 20.0000 | 0.4000 |
| LAOPPE_L2 | 2048.0000 | 20.0000 | 20.0000 | 1.0000 |
| LAOPPE_L2 | 4096.0000 | 20.0000 | 20.0000 | 1.0000 |
| LAOPPE_L2 | 8192.0000 | 20.0000 | 20.0000 | 1.0000 |

## 3. LongBench 小子集

| method | task | dataset | count | mean_input_tokens | em | f1 |
| --- | --- | --- | --- | --- | --- | --- |
| RoPE | passage_retrieval_en | local:LongBench/passage_retrieval_en.jsonl | 30.0000 | 8192.0000 | 0.0000 | 0.0478 |
| RoPE | qasper | local:LongBench/qasper.jsonl | 30.0000 | 5076.3000 | 0.2000 | 0.1101 |
| YaRN_x2 | passage_retrieval_en | local:LongBench/passage_retrieval_en.jsonl | 30.0000 | 8192.0000 | 0.0000 | 0.0301 |
| YaRN_x2 | qasper | local:LongBench/qasper.jsonl | 30.0000 | 5076.3000 | 0.2000 | 0.1038 |
| LAOPPE_L2 | passage_retrieval_en | local:LongBench/passage_retrieval_en.jsonl | 30.0000 | 8192.0000 | 0.0000 | 0.0135 |
| LAOPPE_L2 | qasper | local:LongBench/qasper.jsonl | 30.0000 | 5076.3000 | 0.1667 | 0.0590 |

## 4. 写作建议

若 YaRN_x2 在 PPL 或 Needle 上优于 LAOPPE_L2，应在论文中承认 YaRN 是更强 RoPE 扩展基线；若 LAOPPE_L2 在 LongBench 某些任务上更稳，可强调本文方法在冻结主模型、极少新增参数条件下的性价比。8192 若使用 sliding-window PPL，必须在表格中标注 eval_mode，不能与 full-context PPL 混写。
