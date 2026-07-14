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
