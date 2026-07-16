# TIM Train-only Static Pareto Frontier 結果

## 結論

Train-only static Pareto 實驗得到 8 個不同、且在 evaluation 的 `SLA violations–surplus` 平面上互不支配的固定配置。結果證明兩件事：

1. SLA-best-static 並非只有「4 violations、surplus 15545」這個極端選擇。容許少量 SLA violations，可以在零搬移下顯著降低 surplus。
2. 目前取樣的 static points 沒有同時在 violations、surplus、movements 三項支配 dynamic exhaustive；但這只表示發現一個動態與靜態的 trade-off，**不能直接推論 dynamic relocation 或 RL 已被證明必要**。

若營運 SLA 是 violation rate 不超過 10%，train-only static allocation 已可在 evaluation 達到 `15/168 = 8.93%`，surplus 13025、零搬移。Dynamic exhaustive 的平均 violations 為 `34.7/168 = 20.65%`，反而不符合這個 SLA。因此是否需要動態方法，取決於先明確指定的 SLA constraint，而不能只看 scalar utility 或「有沒有支配」。

## Static frontier

| Allowed train violation rate | Actual train violations | Node / pool units | Eval utility | Eval gap | Eval surplus | Eval violations | Eval rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0% | 0 | 175 / 5 | -37.773 | -4 | 15545 | 4 | 2.38% |
| 0.5% | 5 | 168 / 12 | -22.410 | -9 | 14367 | 7 | 4.17% |
| 1% | 9 | 165 / 15 | -15.943 | -13 | 13864 | 10 | 5.95% |
| 2.5% | 23 | 160 / 20 | -5.123 | -19 | 13025 | 15 | 8.93% |
| 5% | 50 | 155 / 25 | 4.683 | -41 | 12202 | 25 | 14.88% |
| 10% | 100 | 148 / 32 | 16.943 | -95 | 11073 | 38 | 22.62% |
| 20% | 200 | 137 / 43 | 28.727 | -298 | 9417 | 57 | 33.93% |
| 40% | 400 | 97 / 83 | 2.617 | -2125 | 4484 | 85 | 50.60% |

所有配置 movement count 均為 0。兩個 evaluation seeds 結果完全相同，只是 deterministic check，不是兩個獨立樣本。

## SLA 門檻下的選擇

### SLA violation rate <= 5%

`7/168 = 4.17%` 的固定配置符合門檻：

- surplus：14367；
- 相對極端 SLA-best-static 減少 1178（7.58%）；
- node units 從 175 降至 168；
- movements：0。

### SLA violation rate <= 10%

目前最低 surplus 的合格固定配置為：

- violations：15/168（8.93%）；
- surplus：13025；
- remaining gap：-19；
- node/pool units：160/20；
- movements：0。

相對極端 SLA-best-static，它以增加 11 個 violation hours，換取 surplus 減少 2520（16.21%）及少配置 15 個 node units。

這不是客觀唯一的「最佳點」。5% 或 10% 門檻必須由 SLA／營運需求事先決定；不能看完 evaluation 後再挑最有利的 threshold。

## 與 Dynamic exhaustive 的比較

| Method | Utility | Gap | Surplus | Violations | Violation rate | Moves |
|---|---:|---:|---:|---:|---:|---:|
| Static，train threshold 5% | 4.683 | -41 | 12202 | **25** | **14.88%** | **0** |
| Static，train threshold 10% | 16.943 | **-95** | 11073 | 38 | 22.62% | **0** |
| Dynamic exhaustive mean | **18.921** | -293.4 | **10072.8** | 34.7 | 20.65% | 42.6 |

Dynamic exhaustive 位於兩個 sampled static points 之間：

- 相對 train-threshold-5% static，它少 2129.2 surplus，但多 9.7 violation hours及 42.6 movements；
- 相對 train-threshold-10% static，它少 1000.2 surplus、少 3.3 violations，但 aggregate gap 較差 198.4，且多 42.6 movements；
- scalar utility 相對 train-threshold-10% static 只高 1.978。

因此目前沒有 sampled static point 同時以更低 violations、更低 surplus、且更少 movements 支配 exhaustive；反過來，exhaustive 也沒有支配鄰近 static points。這是多目標 trade-off，不是單一勝負。

## 為何還不能宣稱 dynamic relocation 必要

### 1. Frontier 只有 8 個 thresholds

Training allowed violations 從 50 直接跳到 100。Dynamic exhaustive 恰好落在相鄰 evaluation points 的中間，尚未排除 allowed violations 51–99 中存在更強 static allocation。要做正式 dominance 判斷，應密集求解這一段，或枚舉所有 0–403 thresholds 並去重。

### 2. 仍只有一個 evaluation week

所有方法評估同一個 168-hour traffic week。即使 dense frontier 確認 dynamic 在這一週有優勢，也不能推論到其他週期或 load。

### 3. Dynamic baseline 不是 dynamic oracle

目前 exhaustive 是原程式中的 one-step action baseline，受 reward、單步決策與動作空間限制。它不是「所有動態策略可達到的最佳邊界」，也不能代表 Forecast-PRORL 或 MPC 的上界。

### 4. 必須先指定 constraint

若 SLA 要求 <=10% violations，現有 static 可達標、exhaustive 不達標；若只要求 <=25%，兩者皆可行，才需要在 surplus、movement cost 與 gap 間比較。沒有事先指定 SLA epsilon，就無法定義哪個 trade-off 是營運上更好。

## 完整性核對

- Archive SHA256：`63d600d7857935a535d5f410a0dac5fc9180e60e01ecd445ffeb7234f4762af3`。
- Training：1008 hours、12 nodes、12096 rows。
- Training hash 前後一致：`4c331ce63d2bb6d5adb9ae83da2696ceb4edcdd59d1df95199c60dfc762b166c`。
- Phase A forbidden access attempts：0。
- 8 thresholds × 3 sequential stages 全為 `OPTIMAL`，objective 與 best bound 相同。
- Synthetic CP-SAT 結果與完整 brute force optimum 一致。
- 所有 allocation 都滿足 node + pool = 180，並在 evaluation 前凍結。
- Evaluation/index hashes 前後一致。
- 8 個 evaluation 均完成 168 hours；WaitAgent、learning updates 0、movements/add/remove/cost 0。
- Seeds 1000/1900 逐欄完全相同，只用於 determinism check，未計算 CI 或 p-value。

## 下一個判別實驗

後續 `50..100` dense frontier 已完成，結果見 `TIM_STATIC_PARETO_DENSE_RESULTS_ZH.md`。K=85 static 得到 34 violations、surplus 11406；dynamic exhaustive mean 為 34.7 violations、surplus 10072.8。因此動態方法可能在相近 SLA 下減少 overprovisioning，但需支付 movement，且尚未排除 K=101..201 與 training-optimum ties 中更強的 static solution。

其後必須做 rolling-origin multi-week evaluation。只有在多個未見 traffic windows 上，dynamic oracle／MPC 穩定優於 train-only dense static frontier，才值得把 constrained Forecast-PRORL 當作主要演算法改進。
