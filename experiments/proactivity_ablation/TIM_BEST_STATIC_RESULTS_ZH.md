# TIM Train-only Utility-best-static 結果

## 結論

在 `(gap, surplus, cost) = (0.75, 0.2, 0.05)` 下，使用 training split 擬合、evaluation 前凍結的 static allocation 得到 utility `29.527`，高於原始 uniform wait 的 `16.330`，也高於 one-step exhaustive relocation 的平均 `18.921`。

因此，若論文的主要目標是 scalarized utility，這個設定目前**沒有證明需要動態調度或 RL**。先前 exhaustive 相對 uniform wait 的優勢，至少不足以排除「初始 allocation 不佳」這個解釋。

但 best-static 的 SLA violations 為 97/168 小時，比 uniform wait 的 64 與 exhaustive 的 34.7 都差。也就是說：best-static 同時改善 aggregate remaining gap 與 surplus，卻讓更多小時出現至少一個不滿足節點。這再次證明 scalarized aggregate reward 與逐時 SLA 並不等價。

## Frozen allocation

| Node ID | Units |
|---:|---:|
| 2841 | 7 |
| 1760 | 7 |
| 2042 | 10 |
| 2338 | 12 |
| 2340 | 14 |
| 2920 | 12 |
| 2041 | 11 |
| 2927 | 9 |
| 2345 | 11 |
| 2120 | 8 |
| 2774 | 9 |
| 3004 | 9 |

- Node units：119
- Pool units：61
- Total：180
- 原始 uniform allocation：每個 node 10，共 120；pool 60

這不是增加資源，而是依 training trace 重新分配相同總資源，並多保留一個 unit 在 pool。

## Evaluation comparison

| Method | Utility | Remaining gap | Surplus | Moves | SLA OK | Violations |
|---|---:|---:|---:|---:|---:|---:|
| Uniform wait | 16.330 | -1095.0 | 7341.0 | 0 | 104 | 64 |
| One-step exhaustive（10-seed mean） | 18.921 | -293.4 | 10072.8 | 42.6 | 133.3 | 34.7 |
| **Train-only utility-best-static** | **29.527** | -922.0 | **6999.0** | **0** | 71 | 97 |

相對 uniform wait，best-static：

- utility `+13.197`；
- aggregate remaining gap `+173`（較接近 0）；
- surplus `-342`；
- SLA violations `+33`；
- movements 維持 0。

相對 exhaustive，best-static：

- utility `+10.606`；
- aggregate remaining gap `-628.6`（較差）；
- surplus `-3073.8`；
- SLA violations `+62.3`；
- movements `-42.6`。

## 為何 aggregate gap 改善但 SLA 變差

`remaining_gap_total` 累加缺口幅度；`SLA violation hours` 計算該小時是否至少有一個 node 未滿足。Static allocation 可以把少數很大的 gap 攤成更多小 gap，使 aggregate gap 變小，卻增加 violation hours。

因此兩個指標回答不同問題：

- aggregate gap：總共缺多少 units；
- hourly SLA：多少小時完全滿足所有 nodes。

只用線性 reward 最佳化前者，不會自然保證後者。

## 完整性與方法核對

- Phase A 只允許讀 training data 與 config；沒有 evaluation/validation access attempt。
- Training hours：1008；12 nodes。
- 搜尋：multiple-choice integer knapsack dynamic programming。
- 搜尋空間：每 node 0–180 integer units，總和不超過 180，剩餘留 pool。
- Per-node objective 使用與 PRORL reward 相同的 floor delta、capacity 889、training normalization `[0,10]` 與 weights。
- 小型問題以 brute force 驗證 DP optimum，結果一致。
- Allocation JSON 在 evaluation 前設為 read-only 並 hash；前後 hash 相同。
- Seeds 1000/1900 結果逐欄完全相同，只作 determinism check，不當作兩個統計樣本。
- Evaluation/index/training hashes 不變；learning updates 與 movements 為 0。
- Source archive SHA256：`7ffb0597c0eaa93c786e98e2f735e87404201970839592ef552486546cfc77c9`

## 科學判斷

目前可以說：

1. **以 scalarized utility 為準**：train-only static capacity planning 已勝過現有動態 baselines，尚無必要導入 RL。
2. **以 hourly SLA 為準**：exhaustive relocation 仍明顯較好，但尚未與 train-only SLA-best-static 比較。
3. **Reward formulation**：0.75 設定也出現 utility 與 SLA 方向不一致；0.35 設定的錯位不是孤例。

因此下一個必要 control 是 `train-only SLA-best-static`：先最小化 training SLA violations，再以 aggregate gap、surplus 作 tie-break。只有它仍明顯輸給 dynamic exhaustive，才能說明 SLA 目標真的需要動態 relocation。
