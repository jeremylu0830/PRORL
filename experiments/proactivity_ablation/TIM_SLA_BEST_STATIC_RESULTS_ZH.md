# TIM Train-only SLA-best-static 結果

## 結論

在 `(gap, surplus, cost) = (0.75, 0.2, 0.05)`、相同 180 units 下，僅使用 training split 擬合並在 evaluation 前凍結的 SLA-best static allocation，只出現 **4/168 小時 SLA violations**。它明顯優於：

- uniform wait：64/168；
- train-only utility-best-static：97/168；
- one-step dynamic exhaustive：平均 34.7/168。

因此，在目前這一個固定 TIM evaluation week，**連 hourly SLA 目標也沒有證明需要 dynamic relocation 或 RL**。相反地，結果顯示主要問題是初始 uniform allocation 不佳；利用歷史 training trace 做一次離線 capacity planning，已能在不搬移資源的情況下滿足 97.6% 的 evaluation hours。

這不是說動態方法永遠沒有價值。SLA-best-static 仍有 4 個 violation hours，理想的動態方法可能消除它們；但在這個 evaluation week，相對此強 static baseline，動態方法可改善的上限只剩 4/168 小時。必須增加不同週期或 rolling evaluation，才能判斷這 4 小時是否具有可泛化的動態調度需求。

## Frozen allocation

| Node ID | Units |
|---:|---:|
| 2841 | 12 |
| 1760 | 9 |
| 2042 | 14 |
| 2338 | 17 |
| 2340 | 23 |
| 2920 | 18 |
| 2041 | 15 |
| 2927 | 14 |
| 2345 | 15 |
| 2120 | 12 |
| 2774 | 12 |
| 3004 | 14 |

- Node units：175
- Pool units：5
- Total：180
- Training：1008/1008 hours satisfied，aggregate deficit 0

這個結果有一個重要的結構性含義：training trace 中各 node 的 peak requirement 可以同時由 175 units 覆蓋，甚至仍有 5 units 留在 pool。因此該 training split 本身不存在「資源總量足夠、但必須隨時間在 nodes 間搬移才能滿足 SLA」的壓力。

## Evaluation comparison

| Method | Utility | Remaining gap | Surplus | Moves | SLA OK | Violations |
|---|---:|---:|---:|---:|---:|---:|
| Uniform wait | 16.330 | -1095.0 | 7341.0 | 0 | 104 | 64 |
| Train-only utility-best-static | **29.527** | -922.0 | **6999.0** | 0 | 71 | 97 |
| One-step exhaustive（10-seed mean） | 18.921 | -293.4 | 10072.8 | 42.6 | 133.3 | 34.7 |
| **Train-only SLA-best-static** | -37.773 | **-4.0** | 15545.0 | **0** | **164** | **4** |

相對 uniform wait，SLA-best-static：

- SLA violations `-60` 小時；
- aggregate remaining gap 改善 `1091` units；
- surplus增加 `8204` units；
- utility 降低 `54.103`；
- movements 維持 0。

相對 dynamic exhaustive，SLA-best-static：

- SLA violations減少 `30.7` 小時；
- aggregate remaining gap改善 `289.4` units；
- surplus增加 `5472.2` units；
- utility降低 `56.694`；
- movements減少 `42.6`。

## 這代表什麼

### 1. Dynamic exhaustive 的 SLA 優勢已被更強 static control 推翻

先前 exhaustive 相對 uniform wait 將 violations 從 64 降到 34.7，只能證明 uniform initial allocation 很差。加入 train-only SLA-best-static 後，固定配置可進一步降到 4，故不能再把 0.75 設定當作「動態搬移是必要的」證據。

### 2. Scalar utility 與 SLA 的衝突非常明確

Utility-best-static 的 utility 最高，但 SLA 最差；SLA-best-static 的 SLA 最佳，但 utility 最低。線性 scalar reward 並未表達「先滿足 SLA，再降低 surplus/cost」的營運偏好。

這支持把問題改寫成 constrained optimization：例如要求 `SLA violation rate <= epsilon`，再在可行解中最小化 surplus 與 movement cost，而不是繼續調整三個 reward weights。

### 3. 現在不能用這個結果宣稱一般化

兩個 evaluation seeds 對 deterministic wait policy 產生完全相同結果，它們只是 determinism check，不是兩個獨立樣本。所有比較仍只涵蓋同一個 168-hour evaluation week，因此沒有 confidence interval，也不能推論到其他週、其他 load 或其他 node sets。

## 完整性與方法核對

- Phase A 只讀 training data 與 config；沒有 evaluation/validation access attempt。
- Training data hash 前後一致：`4c331ce63d2bb6d5adb9ae83da2696ceb4edcdd59d1df95199c60dfc762b166c`。
- 使用 OR-Tools CP-SAT 做四階段 exact lexicographic optimization：最大化 satisfied hours、最小化 deficit、最小化 surplus、最小化 node units。
- 四階段狀態皆為 `OPTIMAL`，objective 與 best bound 完全相同；單執行緒、seed 0。
- 小型 synthetic instance 與完整 brute force 的唯一 optimum 一致。
- Required units 使用 `ceil(scaled_demand / 889)`，並逐列核對 PRORL floor-delta semantics，共 12,096 rows。
- Allocation 在 evaluation 前設為 read-only；前後 hash 一致。
- Evaluation 與 index hashes 前後一致；learning updates、movements、add/remove counts 全為 0。
- Seeds 1000/1900 所有結果欄位完全相同，只作 determinism check。
- Source archive SHA256：`360c66eabc2ceac404545b98c2ef6e6ba71d21b86850f10c4ae214f676a7ed50`。

## 下一步

不應立即投入更複雜 RL。下一個具判別力的實驗是：

1. 用 training-only optimization 建立 `SLA violation vs surplus/resource` 的 static Pareto frontier，而非只比較 utility 與極端 SLA 兩個端點；
2. 在多個不重疊 evaluation windows 或 rolling-origin splits 上凍結評估；
3. 只有當合理 SLA threshold 下的 static frontier 穩定輸給 dynamic oracle/MPC，才訓練 constrained PRORL 或 Forecast-PRORL；
4. 若 static 仍足夠，論文貢獻應轉向 benchmark/formulation audit，而不是宣稱 RL scheduling improvement。
