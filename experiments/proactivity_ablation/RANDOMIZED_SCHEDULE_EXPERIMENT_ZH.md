# Randomized-Schedule PRORL 實驗規格

## 目的

Same-checkpoint inference ablation 已證明 Forecast-No-Time policy 會使用 forecast 的存在、node identity 與 horizon order。下一個問題不是換 LSTM，而是：

> 在訓練 schedule 不固定時，Forecast-No-Time 能否學到可泛化的 forecast-to-action mapping，並在未見過的 scenario combinations 上超越沒有 forecast 的 policy？

## 公平對照

正式訓練只有兩組：

| Condition | Time encoded | Oracle forecast h=6 | Randomized workload |
|---|---:|---:|---:|
| Randomized-No-Time | 否 | 否 | 是 |
| Randomized-Forecast-No-Time | 否 | 是 | 是 |

兩組沿用相同 PRORL agent、reward、action space、training steps 與 seeds。唯一的 state 差異是 `node-demand-forecast`。

## Scenario bank

固定 manifest：`randomized_schedule_scenarios.json`。

- Training bank：12 個 scenarios。
- Held-out bank：4 個完整 scenario combinations。
- 每個 scenario 都指定兩個 demand couples 的 peak day/hour、target endpoint、stress magnitude 與 duration。
- Training 與 held-out 的 ID 及完整 payload 均不重複。
- 每個 168 小時 training episode reset 時，使用 generator random state 從 training bank 抽一個 scenario。
- 相同 environment seed 的兩組條件抽到完全相同的 scenario sequence。
- Schedule Oracle 不使用額外亂數，因此不會改變真實 demand noise sequence。

Randomization 是 SyntheticModel 的 optional config，預設為 disabled；舊 config 的行為不變。

## 正式規模

```text
2 conditions × 10 training seeds = 20 training runs
4 held-out scenarios × 5 evaluation seeds × 20 frozen checkpoints
= 80 evaluation jobs / 400 episodes
```

## 完整性規則

- 遠端必須由 Git 的同一個 commit 執行，不能在遠端另外修改 code。
- Training queue 必須為空；launcher 遇到非空 queue 會拒絕執行。
- Launcher 必須確認 scheduler 建立 exactly 20 runs。
- Held-out evaluation 設為 `RunMode.Eval`，禁止 `learn()`。
- 每個 evaluation job 比較前後所有 agent tensors，任何變動都視為失敗。
- Held-out evaluation 會關閉 schedule randomization，逐一套用固定 manifest scenario。

## 已完成的相容性與 smoke test

- Legacy forecast config：randomization disabled，原 schedule 不變。
- Manifest：12 train／4 held-out，完整 payload disjoint。
- Paired config：移除 forecast feature 後，兩份 run config 相同。
- Same-seed sampling：No-Time 與 Forecast-No-Time scenario histories 相同。
- 192-step paired training smoke：兩組都完成真正的 `prorl-no-split` training。
- 兩組各有 25/73 agent tensors 更新。
- Smoke scenario sequence 相同：`train-09 → train-07`。
- 只有 Forecast condition 產生 forecast state。

## 判定標準

主要 paired contrast 為：

```text
Randomized-Forecast-No-Time − Randomized-No-Time
```

每個 training seed 先平均 5 evaluation seeds 與 4 held-out scenarios，再對 10 個 paired training seeds 做 exact two-sided sign-flip test。

主要 metrics：

- utility
- remaining gap
- event allocation lift
- six-hour pre-peak target rate
- gap hours
- surplus
- movement cost

解讀：

1. Held-out allocation lift／pre-peak targeting 改善：forecast-to-action behavior 能泛化。
2. Utility／remaining gap 也改善：forecast 的行為價值成功轉成 end-to-end objective gain。
3. 行為改善但 objective 不改善：training generalization 已改善，下一個瓶頸是 action space 或 reward。
4. 行為也不改善：目前 raw forecast encoder／training algorithm 仍無法學到穩定 mapping，才需要考慮 encoder 或 LSTM。

## 遠端執行

Redis 與 MongoDB 啟動後，由 `launch_randomized_training.py` 統一產生 config、schedule 20 runs、啟動 training/validation workers 並寫入：

```text
/tmp/prorl_randomized_training_status.json
```

建議先使用 2 個 concurrent training processes；確認 CPU、RAM 與 GPU memory 穩定後才提高到 3–4。

## 正式結果（2026-07-14）

完整性稽核：

- 20/20 training runs 完整，兩組各含 seeds 10–19。
- 每組都訓練 172,800 iterations，使用相同 12-scenario training bank。
- 20/20 best-validation checkpoints 存在。
- Held-out evaluation 完成 80 jobs／400 episodes。
- 所有 jobs 都禁止 learning，前後 73 個 agent tensors 完全一致。

四個 held-out scenarios 聚合後，Randomized-Forecast-No-Time 相較 Randomized-No-Time：

| Metric | Forecast − No-Time | 95% CI | Exact p |
|---|---:|---:|---:|
| Utility | -2.362 | [-3.999, -0.724] | 0.0137 |
| Remaining gap | +6.565 | [-39.031, 52.161] | 0.7441 |
| Surplus | +52.855 | [-31.590, 137.300] | 0.1855 |
| Movement cost | -0.435 | [-1.337, 0.467] | 0.3438 |
| Event allocation | +0.881 | [0.405, 1.358] | 0.0020 |
| Event allocation lift | +0.770 | [0.397, 1.143] | 0.0020 |
| Pre-peak target rate | +0.085 | [0.036, 0.135] | 0.0078 |
| Any correct pre-peak add | +0.412 | [0.318, 0.507] | 0.0020 |

所有 10 個 paired training seeds 的 held-out allocation-lift difference 都為正。這提供比 fixed-schedule 與 same-checkpoint inference ablation 更強的證據：模型已學到可跨未見 scenario combinations 泛化的 forecast-to-action mapping。

但 end-to-end objective 沒有改善。Utility 反而顯著降低 2.362；remaining gap、gap hours 與 movement cost 沒有顯著差異。主要張力是 Forecast policy 在 event 時多配置約 0.881 units，同時 surplus 呈增加方向。未做 multiplicity correction 的 exploratory scenario 分析中，heldout-01 的 surplus 增加 92.0（p=0.0215），utility 降低 4.263（p=0.0117）；主要結論仍以跨四個 scenarios 的 paired aggregate 為準。

## 更新後的研究判斷

Randomized training 已經完成原本要回答的問題：forecast usage 不只是背固定 schedule，也能泛化到 held-out workload combinations。因此下一步不應優先換 LSTM；Oracle 已排除 forecast accuracy，randomized evaluation 也排除固定 calendar memorization。

現在最合理的瓶頸是「知道未來需求」無法被現有 action／reward 設計有效利用：

1. Agent 每步可移動的 quantity 有限，且容易持續搬動。
2. 線性 reward 將 remaining gap、surplus 與 movement cost 混合，提前累積資源可能被 surplus penalty 抵銷。
3. Forecast policy 能把資源送到正確節點，但缺少精準 quantity 與停止搬動的機制。

下一個實驗已完成：frozen-policy movement-gated action ablation。Combined add/remove gate 將 utility 提高 32.631、remaining gap 提高 44.750、surplus 降低 182.130、movement cost 降低 327.870（四者 exact p=0.0020），並將平均 active sub-actions 從 336.3 降至 6.9。這確認 continuous action thrashing 與 add/remove 失衡是目前主要 action-conversion 瓶頸。完整設計與限制見 `MOVEMENT_GATE_ABLATION_ZH.md`。

後續 always-wait、current-only 與 forecast horizon 1／3／6 controls 已完成。Always-wait utility 166.032，高於 h6 的 165.586，證明先前 utility improvement 主要是 movement suppression。Forecast horizon 增加仍會改善 remaining gap 與 event allocation lift，但 surplus/cost 上升，使 utility 在 h1 後下降；完整結果見 `MOVEMENT_GATE_CONTROLS_ZH.md`。

這也揭露目前 benchmark 的退化：初始 allocation 已足以讓 static policy 接近最佳 scalarized utility。下一步應先建立 always-wait 會失敗的 movement-necessary scenarios（imbalanced initial allocation、resource scarcity、global/local overdemand），再實作 training-time action masking，而不是先換 LSTM。
