# Frozen-policy Movement-Gating Ablation

## 研究問題

Randomized-schedule 實驗已證明 Forecast-No-Time policy 能把 forecast 轉成可泛化的提前配置行為，但 utility 反而下降。本實驗不重新訓練，直接用相同 frozen checkpoints 回答：問題是否來自持續且不安全的資源搬動？

## 實驗設計

使用 randomized Forecast-No-Time 的 10 個 best-validation checkpoints。每個 checkpoint 評估 4 個 held-out scenarios、5 個 evaluation seeds，共：

```text
10 checkpoints × 4 gates × 4 scenarios × 5 seeds
= 160 jobs / 800 episodes
```

四種 inference-only gates：

1. `baseline`：完全不修改 policy action。
2. `satisfied`：若 add target 的現有 capacity 已涵蓋該節點「current demand 與未來六小時 oracle forecast」的最大值，將 add 子動作改成 wait。
3. `safe-remove`：若 remove 後的 capacity 會低於相同需求上界，將 remove 子動作改成 wait。
4. `combined`：同時套用兩者。

PRORL 的 pool action 是 add 與 remove 兩個獨立子動作；gate 只取消違反條件的一邊。修改發生在 `env.step` 前，因此原 rollout action history 記錄實際執行動作，metadata 另存 proposed／canceled／executed counts。

這個改動只存在於 evaluation runner，不修改 checkpoint、network、reward、training config 或原始 agent。所有 jobs 均禁止 learning，並比較 evaluation 前後 73 個 agent tensors。

## 完整性檢查

- 160/160 jobs、800/800 episodes 完成。
- 所有 checkpoint tensors 前後完全相同。
- Baseline 與先前 randomized held-out Forecast 評估的 10,000 個 non-timing fields 逐值一致。
- 五個 movement-gate unit tests 通過。
- 完整 forecast/randomized/movement 測試共 16 tests 通過。

## 絕對結果

下表先平均 evaluation seeds、scenarios 與 training seeds：

| Gate | Utility | Remaining gap | Surplus | Movement cost | Active sub-actions | Add cancel rate | Remove cancel rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 132.954 | -52.335 | 201.485 | 334.725 | 336.3 | 0.000 | 0.000 |
| satisfied | 142.457 | -138.830 | 5.600 | 170.945 | 171.5 | 0.496 | 0.000 |
| safe-remove | 106.096 | -4.680 | 774.080 | 299.145 | 300.1 | 0.000 | 0.103 |
| combined | 165.586 | -7.585 | 19.355 | 6.855 | 6.9 | 0.980 | 0.894 |

## Paired contrasts（gate − baseline）

每個 training seed 先平均 4 scenarios 與 5 evaluation seeds，再對 10 個 paired seeds 做 exact two-sided sign-flip test。

| Gate | Metric | Difference | 95% CI | Exact p |
|---|---|---:|---:|---:|
| satisfied | Utility | +9.503 | [-7.123, 26.129] | 0.2188 |
| satisfied | Remaining gap | -86.495 | [-161.586, -11.404] | 0.0020 |
| satisfied | Surplus | -195.885 | [-265.017, -126.753] | 0.0020 |
| satisfied | Movement cost | -163.780 | [-261.724, -65.836] | 0.0020 |
| safe-remove | Utility | -26.858 | [-29.971, -23.745] | 0.0020 |
| safe-remove | Remaining gap | +47.655 | [21.017, 74.293] | 0.0020 |
| safe-remove | Surplus | +572.595 | [478.619, 666.571] | 0.0020 |
| safe-remove | Movement cost | -35.580 | [-79.890, 8.730] | 0.0020 |
| combined | Utility | **+32.631** | **[30.296, 34.967]** | **0.0020** |
| combined | Remaining gap | **+44.750** | **[18.124, 71.376]** | **0.0020** |
| combined | Surplus | **-182.130** | **[-255.178, -109.082]** | **0.0020** |
| combined | Movement cost | **-327.870** | **[-329.464, -326.276]** | **0.0020** |
| combined | Event allocation lift | **+0.816** | **[0.623, 1.008]** | **0.0020** |

## 解讀

兩個單獨 gate 揭露明確交互作用：

- 只阻止「已滿足節點的 add」會讓 remove 繼續抽走資源，surplus 幾乎消失，但 remaining gap 惡化。
- 只阻止 unsafe remove 會讓 add 繼續送入資源，remaining gap 改善，但 surplus 暴增。
- Combined 同時切斷兩個失衡方向，保留少量真正必要的 movement，因此 gap、surplus、cost 與 utility 一起改善。

這是很強的診斷證據：Forecast policy 並非不知道資源該去哪裡，而是原 action execution 缺少「已足夠就停止」與「不要從即將需要的節點移除」的結構性限制。原 policy 平均每 episode 執行 336.3 個 active sub-actions，combined 僅需 6.9 個；主要浪費是 action thrashing。

## 尚不能宣稱的事

Combined gate 使用 oracle forecast，而且初始 allocation 可能已接近這組 synthetic workloads 的合理配置。因為它取消約 98.0% adds 與 89.4% removes，目前不能直接宣稱 improvement 全部來自 forecast；它也可能部分近似 static/no-move policy。

因此下一個必要 control 不需訓練：在相同 checkpoints 與 held-out protocol 比較：

1. `always-wait` static policy；
2. `current-only combined gate`；
3. forecast horizon 1／3／6 combined gates；
4. 原 `forecast-h6 combined gate`。

若 h6 明顯勝過 current-only 與 always-wait，才能將額外收益歸因於 forecast-aware gating。之後再把 combined rule 移入 action masking／training，檢驗 DQN 是否能 end-to-end 學會受約束的行為。

## 重現指令

```bash
ENV=test python experiments/proactivity_ablation/run_movement_gate_ablation.py \
  <randomized Forecast-No-Time result root> \
  --output-dir experiments/proactivity_ablation/movement_gate_evaluations

python experiments/proactivity_ablation/analyze_movement_gate_ablation.py \
  experiments/proactivity_ablation/movement_gate_evaluations \
  --baseline-reference \
    experiments/proactivity_ablation/randomized_heldout_evaluations/Randomized-Forecast-No-Time \
  --output-dir experiments/proactivity_ablation/movement_gate_analysis
```
