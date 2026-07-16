# 最小版 Constrained Forecast-PRORL 實作紀錄

## 目的與範圍

這個版本用最小改動驗證一件事：在已有的六小時 oracle forecast state 上，將 SLA
需求由原始線性 reward 的一個目標，改成可學習乘子的限制條件，是否能減少 reward
權重錯置與無效 relocation。

原始 PRORL、`gap-surplus-cost` reward，以及既有 config 都保留不變。只有 config
明確指定 `constrained-gap-surplus-cost` 時才啟用新路徑。因此這個版本可與原方法做
paired comparison，也能直接退回原始行為。

這是 RCPO-style 的最小原型，不宣稱已提供有限樣本下的硬 SLA 保證，也尚未改變
原始 action space、DQN 架構或 replay schema。

## 狀態與目標

新 config 使用：

- `node-capacity`
- `node-demand`
- `node-delta`
- `node-demand-forecast`：node-major 的未來六小時 oracle demand
- `current-lives`
- `node-add`

它移除 `time-encoded`，避免 agent 仍以固定行事曆取代 forecast-to-action mapping。

primal objective 不含 remaining gap：

```text
base_reward = 0.5 * normalized(-surplus)
            + 0.5 * normalized(-movement_cost)
```

每個有效的 hourly reward step 定義 binary constraint cost：

```text
c_t = 1  if any aggregate remaining gap is negative
      0  otherwise
```

目標違反率為 `epsilon = 0.10`，實際送入 DQN 的 shaped reward 是：

```text
r_t = base_reward - lambda * (c_t - epsilon)
```

lambda 每 168 個有效 reward steps 更新一次：

```text
lambda <- clip(lambda + dual_lr * (mean(c) - epsilon), 0, lambda_max)
```

預設值為 `dual_lr=0.05`、`lambda_0=1.0`、`lambda_max=20.0`。訓練與驗證／評估的
objective normalization 都設為 `[0, 1]`，讓 multiplier 的尺度一致。

## Config 開關

產生設定：

```bash
.venv/bin/python experiments/proactivity_ablation/generate_configs.py
```

新檔為：

```text
experiments/proactivity_ablation/generated/constrained_forecast_no_time.yaml
```

關鍵設定：

```yaml
state:
  features:
    - node-capacity
    - node-demand
    - node-delta
    - node-demand-forecast
    - current-lives
    - node-add
  additional_properties:
    forecast_horizon: 6
    forecast_type: oracle
reward:
  type: constrained-gap-surplus-cost
  parameters:
    objective_weights: [0.5, 0.5]
    sla_target_rate: 0.1
    dual_learning_rate: 0.05
    dual_initial_lambda: 1.0
    dual_max_lambda: 20.0
    dual_update_interval: 168
```

## Checkpoint 與 evaluation 語意

lambda、目前 dual window 的 cost/count、更新次數與最近觀察到的 violation rate 都會
存入 agent checkpoint。舊 checkpoint 沒有這個欄位時會載入空 state，維持向後相容。

validation 與 evaluation 會載入 checkpoint 中的 lambda，但不更新它。如此比較的是
同一個已訓練好的 constrained policy，而不是讓 evaluation data 反過來調整限制條件。

新增追蹤欄位包括：

- `sla_cost`
- `base_reward`
- `constraint_penalty`
- `lagrangian_multiplier`
- `observed_sla_rate`
- `dual_updates`

## 已完成驗證

相關 33 個 compatibility／forecast／randomized schedule／movement gate／TIM audit／
constrained tests 全數通過。額外完成 192-step bounded training smoke：

- 實際 agent：`prorl-no-split`
- 73 個 tensors 中有 25 個改變
- forecast horizon：6
- dual update：1 次
- 最終 lambda：約 `1.03131`

這只能證明資料流、更新、checkpoint 與短訓練可執行，不能代替多 seed 正式實驗。

## 已知限制

目前 replay buffer 儲存的是收集 transition 當下已經 shaped 的 reward。lambda 之後改變
時，舊 transition 不會用最新 lambda 重新計算。這維持了最小修改與舊 replay schema
相容，但不是最完整的 constrained off-policy 實作。若正式結果顯示 lambda 變動大或
學習不穩，下一版應在 replay 中分別保存 `base_reward` 與 `sla_cost`，sample 時才使用
目前 lambda 組合 reward。

binary SLA cost 目前代表「該有效 reward step 是否有 aggregate gap」，並非逐 node SLA、
連續 violation magnitude 或 tail-risk 指標。非 hourly／含 null reward 的 config 需要另外
確認 constraint 的時間尺度。

oracle forecast 是機制驗證用上界，不是可部署預測器。只有 oracle 版本證明 constrained
架構有效後，才應替換 seasonal-naive、LSTM 或其他 learned forecast，並單獨報告預測誤差。

## 下一個正式實驗

rolling multi-week benchmark 可在遠端獨立準備，不阻擋本架構開發。架構比較至少應使用
相同 train/evaluation origins、seeds 與 action space，比較：

1. 原始 PRORL。
2. Forecast-No-Time PRORL。
3. Constrained Forecast-No-Time PRORL。
4. train-only static Pareto policy。
5. dynamic oracle／MPC upper bound。

主要報告 evaluation violation rate、surplus、movement cost、moves，以及各週 paired
difference 與 confidence interval。建議預先固定 `epsilon in {0.05, 0.10, 0.20}`；不可用
evaluation week 挑 epsilon 或 checkpoint。只有 constrained policy 跨週達到目標違反率，
且在相同 SLA 下改善 surplus／cost，才可支持架構改善的主張。
