# Static／Current／Forecast-Horizon Controls

## 為什麼需要這組實驗

前一階段的 forecast-h6 combined gate 將 utility 提高 32.631，並把 active sub-actions 從 336.3 降到 6.9。但這個結果有一個重要替代解釋：held-out workload 的初始資源配置可能本來就足夠，combined gate 的主要作用只是讓 policy 接近完全不移動。

因此本實驗用相同 frozen checkpoints 比較 static policy、只看當下需求的 gate，以及不同 forecast horizons，將「停止亂搬」與「使用 forecast」的效果拆開。

## 實驗條件

六組條件：

1. `policy baseline`：原 Forecast-No-Time policy，不修改 action。
2. `always-wait`：policy 仍執行 inference，但 add/remove 一律改成 wait。
3. `current-only`：combined safety gate 只看 current demand。
4. `forecast-h1`：使用 current 與 oracle t+1。
5. `forecast-h3`：使用 current 與 oracle t+1...t+3。
6. `forecast-h6`：使用 current 與 oracle t+1...t+6，即前一實驗的 combined gate。

沿用 10 個 randomized Forecast-No-Time best checkpoints、4 個 held-out scenarios、5 個 evaluation seeds：

```text
6 conditions × 10 checkpoints × 4 scenarios × 5 evaluation seeds
= 240 jobs / 1,200 episodes
```

本次只新增 always-wait、current-only、h1、h3 的 160 jobs／800 episodes；baseline 與 h6 重用已完成且通過完整性稽核的資料。所有 evaluation 禁止 learning，並確認前後 73 個 agent tensors 不變。

## 完整性

- 240/240 jobs、1,200/1,200 episodes 可用。
- 新增 160 jobs 無失敗，所有 tensors 維持 frozen。
- Baseline 與原 randomized held-out evaluation 的 10,000 個 non-timing fields 逐值一致。
- Forecast、randomized schedule、movement gate 共 19 個測試通過。

## 絕對結果

| Control | Utility | Remaining gap | Surplus | Movement cost | Event allocation lift | Active sub-actions |
|---|---:|---:|---:|---:|---:|---:|
| policy baseline | 132.954 | -52.335 | 201.485 | 334.725 | 0.880 | 336.3 |
| always-wait | **166.032** | -15.200 | **2.400** | **0.000** | 0.000 | **0.0** |
| current-only | 165.769 | -13.540 | 6.970 | 3.755 | 0.442 | 3.8 |
| forecast-h1 | 165.941 | -10.735 | 8.400 | 5.335 | 1.021 | 5.3 |
| forecast-h3 | 165.861 | -8.510 | 13.475 | 6.195 | 1.511 | 6.2 |
| forecast-h6 | 165.586 | **-7.585** | 19.355 | 6.855 | **1.696** | 6.9 |

Remaining gap 是負值 penalty，因此越接近 0 越好。

## 關鍵 paired contrasts

每個 training seed 先平均 4 scenarios 與 5 evaluation seeds，再以 10 個 paired seeds 做 exact two-sided sign-flip test。

| Contrast | Metric | Difference | 95% CI | Exact p |
|---|---|---:|---:|---:|
| always-wait − policy | Utility | +33.078 | [30.806, 35.349] | 0.0020 |
| always-wait − policy | Remaining gap | +37.135 | [10.142, 64.128] | 0.0234 |
| always-wait − policy | Surplus | -199.085 | [-269.470, -128.700] | 0.0020 |
| always-wait − policy | Movement cost | -334.725 | [-335.443, -334.007] | 0.0020 |
| current-only − always-wait | Utility | -0.263 | [-0.711, 0.185] | 0.0215 |
| forecast-h1 − current-only | Utility | +0.172 | [0.128, 0.216] | 0.0020 |
| forecast-h1 − current-only | Remaining gap | +2.805 | [2.121, 3.489] | 0.0020 |
| forecast-h1 − current-only | Event allocation lift | +0.579 | [0.412, 0.746] | 0.0020 |
| forecast-h3 − forecast-h1 | Utility | -0.080 | [-0.131, -0.030] | 0.0137 |
| forecast-h6 − forecast-h3 | Utility | -0.275 | [-0.351, -0.198] | 0.0020 |
| forecast-h6 − current-only | Remaining gap | +5.955 | [4.995, 6.915] | 0.0020 |
| forecast-h6 − current-only | Event allocation lift | +1.254 | [1.005, 1.502] | 0.0020 |
| forecast-h6 − current-only | Utility | -0.183 | [-0.288, -0.079] | 0.0039 |
| forecast-h6 − always-wait | Utility | -0.446 | [-0.893, 0.000] | 0.0020 |
| forecast-h1 − always-wait | Utility | -0.091 | [-0.560, 0.378] | 1.0000 |

Exact randomization p-value 與 t-based 95% CI 使用不同推論假設，因此少數 contrast 可能出現 CI 涵蓋 0、exact p 小於 0.05 的情況；主要判斷同時報告 effect size、CI 與 exact p，不只看單一門檻。

## 科學結論

### 1. 先前 combined gate 的 utility gain 主要來自停止搬動

Always-wait 相較原 policy 提高 utility 33.078，而 forecast-h6 相較原 policy提高 32.631。這表示原 policy 幾乎每一步搬動的行為是主要失敗來源；在目前 workload 與初始 allocation 下，static policy 已接近 scalarized objective 的最佳解。

### 2. Forecast 仍具有可測量的 SLA 價值

Forecast horizon 越長，remaining gap 越接近 0，event allocation lift 也從 current-only 的 0.442，依序提高至 h1 的 1.021、h3 的 1.511、h6 的 1.696。這與之前的 masking、permutation、time-reversal 結果一致：模型與 safety layer 都能使用 forecast 的時間與節點資訊。

### 3. 目前 scalarized utility 不獎勵較長的 proactive horizon

較長 horizon 在改善 SLA 的同時增加 surplus 與 movement cost。Utility 在 h1 達到最高的非 static 值 165.941，之後 h3、h6 逐步下降；h6 甚至比 current-only 低 0.183。這不是 forecast 無效，而是 SLA gain 被目前 reward 的 surplus/cost trade-off 抵銷。

### 4. 目前 benchmark 對資源搬動的必要性不足

Always-wait 同時得到最高 utility、最低 surplus 與零 movement cost。若直接在這個 benchmark 上加入 training-time hard masking，模型很可能只會學成 no-op policy，無法證明更一般的資源調度能力。

## 對下一步架構的影響

目前不應直接把 h6 combined gate 固化成最終演算法。下一個必要步驟是建立「必須搬動才有機會成功」的 evaluation/training scenarios，例如：

1. **Imbalanced initial allocation**：總 capacity 不變，但 episode 開始時故意放在錯誤節點。
2. **Resource scarcity**：降低 pool/total capacity，使所有節點不可能同時覆蓋 peak upper bound。
3. **Global/local overdemand mixture**：同時測試可重新配置與真正不可行的 demand。
4. **Always-wait admission criterion**：新 benchmark 必須讓 always-wait 明顯低於合理 relocation policy，否則不得用來宣稱 action masking 改善調度。

通過這個 movement-necessary benchmark 後，再比較：

- learned WAIT/MOVE head；
- forecast-aware soft/hard action masking；
- learnable horizon 或 risk margin；
- quantity-aware factorized action。

若研究重點是 SLA，則應把 remaining gap 設為 constraint，再最小化 surplus/movement cost；目前結果已顯示單一線性 scalarization 會把可觀的 forecast SLA gain 判成 utility loss。

## 重現指令

```bash
ENV=test python experiments/proactivity_ablation/run_movement_gate_ablation.py \
  <randomized Forecast-No-Time result root> \
  --output-dir experiments/proactivity_ablation/movement_gate_evaluations \
  --modes always-wait combined-current combined-h1 combined-h3

python experiments/proactivity_ablation/analyze_movement_gate_controls.py \
  experiments/proactivity_ablation/movement_gate_evaluations \
  --baseline-reference \
    experiments/proactivity_ablation/randomized_heldout_evaluations/Randomized-Forecast-No-Time \
  --output-dir experiments/proactivity_ablation/movement_gate_control_analysis
```
