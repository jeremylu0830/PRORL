# PRORL Proactivity 實驗分析與改進方向

## 1. 研究問題

PRORL 將自身描述為 proactive resource orchestrator，但原始模型沒有顯式需求預測模組。Agent 的 state 包含星期與時間編碼（`time-encoded`），因此觀察到的尖峰前資源配置可能有兩種解釋：

1. Agent 學會了可泛化的需求預測行為。
2. Agent 只記住訓練環境中固定的星期／小時尖峰排程。

本實驗的核心問題是：

> PRORL 的 proactive behavior 究竟來自需求訊號、固定時間表記憶，還是真正可泛化的需求 anticipation？

---

## 2. 實驗模型

使用相同訓練設定與 training seeds 10–19，比較三組模型：

| 模型 | `time-encoded` | `node-demand`／`node-delta` | 目的 |
|---|---:|---:|---|
| Original | 有 | 有 | 原始完整 PRORL |
| No-Time | 無 | 有 | 測試移除時間資訊後的表現 |
| Calendar-Only | 有 | 無 | 測試只靠時間表能否對準尖峰 |

每組包含 10 個 training seeds，共 30 個訓練結果。

### 固定排程測試結果

| 指標 | Original | No-Time | Calendar-Only |
|---|---:|---:|---:|
| Utility | 138.640 ± 3.147 | 138.643 ± 1.965 | 137.739 ± 3.174 |
| Remaining gap | −32.7 ± 43.5 | −24.0 ± 34.8 | −45.9 ± 45.0 |
| 尖峰前正確加入目標節點 | 75% | 30% | 80% |
| 尖峰時目標節點資源量 | 4.0 | 3.3 | 4.0 |
| 嚴重失敗 runs | 3/10 | 2/10 | 4/10 |

Original 與 No-Time 的 utility 配對差為 −0.003，精確 sign-flip test `p=0.9961`，表示加入時間資訊並沒有提升平均 utility。

但是 Original 的尖峰目標動作比 No-Time 高 45 個百分點（`p=0.0078`），而 Calendar-Only 也能達到 80% 尖峰命中率。這表示 calendar input 確實控制了固定尖峰的動作時機。

---

## 3. 時序對齊修正

原先的分析將設定中的尖峰 hour `h` 直接對應至 action history index `h`，並在尖峰前六小時尋找目標節點動作，因此得到約 4–5 小時的 lead time。

檢查 `EnvWrapper.step()` 後發現實際順序是：

1. Agent 根據目前 state 選擇 action。
2. Environment 套用 action。
3. Environment 前進到下一個 demand state。
4. 使用下一個 demand 計算 reward。

因此 simulation hour `h` 實際對應 action/reward history index `h-1`。

此外，三組策略的 add/remove sub-action 啟用率皆約為 99.6–99.7%。Agent 幾乎每小時都在搬資源，所以在六小時視窗中找到一次目標節點動作，很可能只是持續搬動造成的巧合。

結論：

> 原先的 4–5 小時 lead time 不能被解釋為 PRORL 的需求預測 horizon。

---

## 4. Shifted-Peak Evaluation

### 4.1 目的

固定排程測試無法區分時間表記憶與真正預測，因此使用已完成訓練的 frozen checkpoints，只修改 evaluation 環境的需求尖峰時間。

整個 shifted-peak 實驗不進行重新訓練，也不允許任何 `learn()` 呼叫。

### 4.2 測試排程

| Schedule | 第一個尖峰 | 第二個尖峰 | 用途 |
|---|---|---|---|
| Control | 週四 09:00 | 週六 12:00 | 原始固定排程 |
| Shift +6h | 週四 15:00 | 週六 18:00 | 測試小時偏移 |
| Shift +1d | 週五 09:00 | 週日 12:00 | 測試日期偏移 |
| Unseen | 週二 17:00 | 週日 04:00 | 測試未見排程 |

實驗規模：

```text
30 checkpoints × 4 schedules × 5 evaluation seeds
= 600 evaluation episodes
```

所有 episodes 均完成 168 steps，且確認：

- `run_mode=evaluation`
- `learning_enabled=false`
- checkpoint 成功載入
- control smoke test 的 utility、gap、cost 與全部 action history 均與原結果完全相同

---

## 5. Shifted-Peak 結果

### 5.1 行為結果

| 模型 | Control 命中率 | Shifted 新尖峰命中率 | 舊尖峰誤動作率 | 與 Control 動作相同率 |
|---|---:|---:|---:|---:|
| Original | 75% | 25–40% | 75% | 95.9–98.6% |
| No-Time | 30% | 20–30% | 20–30% | 81.7–95.2% |
| Calendar-Only | 80% | 20–30% | 80% | 100% |

環境共有四個節點，因此隨機選中正確節點的基準為 25%。三組模型在 shifted 新尖峰的命中率均未顯著高於 25%。

### 5.2 Utility Regret

定義：

```text
Utility regret = Control utility − Shifted utility
```

Original 的結果：

| Shift | Utility regret | Exact sign-flip p-value |
|---|---:|---:|
| +6 小時 | 0.403 | 0.0078 |
| +1 天 | 0.415 | 0.0117 |
| Unseen | 0.433 | 0.0098 |

Original 在三種 shifted schedules 的 utility 都顯著下降。

Calendar-Only 的 regret 約為 0.38–0.42，且在三種 shift 中也達統計顯著。No-Time 對 +6h 與 +1d 沒有顯著 regret，但它在 control 中原本就沒有可靠的固定尖峰命中能力。

### 5.3 Calendar-Only

Calendar-Only 在需求尖峰移動後：

- 保持 100% 相同的 add/remove action sequence。
- 仍以 80% 的比例在已失效的舊尖峰時間搬向目標節點。
- 新尖峰命中率只剩 20–30%，與隨機選擇相當。

由於 Calendar-Only 看不到 demand 或 delta，改變需求時間不可能影響它的 observation。這組結果直接證明，它只是在執行記住的 calendar schedule。

### 5.4 Original

Original 能看到需求與 demand delta，但 shifted 後：

- 仍保留 95.9–98.6% 的 control action sequence。
- 在已失效的舊尖峰時段仍有 75% 目標動作率。
- 新尖峰命中率下降至 25–40%。
- 三種 shifted schedules 的 utility 均顯著下降。

因此 Original 雖然會對需求變化做少量反應，主要動作時機仍由 calendar pattern 主導。

---

## 6. 主要研究結論

> Under temporally shifted demand peaks, PRORL preserved 95.9–98.6% of its original action sequence and continued targeting obsolete peak slots at a rate of 75%. Its hit rate at unseen peaks dropped to 25–40%, which was not statistically distinguishable from the 25% chance level. A Calendar-Only agent exhibited 100% action-sequence invariance and an 80% obsolete-slot action rate. These results indicate that the reported proactive behavior is predominantly explained by fixed-schedule memorization rather than generalized demand anticipation.

中文結論：

> PRORL 在原始固定排程中展現的尖峰前資源配置，主要可由 calendar schedule memorization 解釋。模型在尖峰時間改變後仍繼續於舊時段執行原動作，且對新尖峰的命中率沒有顯著高於隨機基準。因此，現有實驗不足以證明 PRORL 具有可泛化的 proactive demand anticipation 能力。

---

## 7. 為何總 Utility 容易掩蓋問題

每條 evaluation episode 有 168 小時，但只有兩個單小時需求尖峰。即使模型錯過兩次尖峰，其餘 166 個平靜小時仍會主導總 utility。

因此只報告 total utility 會低估 proactive failure。建議新增以下指標：

- New-peak hit rate
- Obsolete old-slot action rate
- Target allocation at the peak
- Action-sequence agreement
- Utility regret
- Gap hours
- 每週資源搬移次數
- 無效來回搬移比例

---

## 8. 訓練穩定性與動作問題

固定排程測試中出現的嚴重失敗 runs：

- Original：seeds 11、15、19
- No-Time：seeds 10、18
- Calendar-Only：seeds 11、12、14、16

失敗策略不是只錯過兩個尖峰，而是整週反覆產生 remaining gap，代表 agent 學到不良的週期性配置循環。

此外，三組模型約 99.6–99.7% 的 add/remove sub-actions 都處於啟用狀態，顯示 agent 幾乎每小時搬動資源。這會造成：

- Orchestration chattering
- 額外 movement cost
- 資源在節點間反覆來回
- 偶然命中尖峰，被錯誤解釋成 proactive behavior

---

## 9. 建議改進：Forecast-PRORL

下一階段應將 proactivity 從隱式 calendar memorization 改成顯式需求預測。

### 9.1 State 改進

原始 state：

```text
資源配置 + 當前需求 + demand delta + calendar time
```

Forecast-PRORL：

```text
資源配置
+ 當前需求
+ 過去需求歷史
+ 未來 k 小時需求預測
+ 預測不確定性
```

### 9.2 建議比較模型

| 模型 | 未來資訊 | 用途 |
|---|---|---|
| Original | 無顯式 forecast | 原始基準 |
| No-Time | 無 | 需求反應基準 |
| Seasonal-Naive PRORL | 上週同時段需求 | 簡單 forecast baseline |
| LSTM／GRU PRORL | 學習式需求預測 | 主要改善模型 |
| Oracle PRORL | 真實未來需求 | 性能上限 |

Forecast-PRORL 應在 shifted／unseen peaks 中達成：

- 新尖峰命中率顯著高於 25%
- 舊尖峰誤動作率降低
- Utility regret 降低
- Gap hours 減少
- 對 forecast error 保持一定穩健性

### 9.3 減少無效搬移

可同時加入：

- Hold／No-op action
- Switching penalty
- Minimum holding time
- Hysteresis threshold
- Quantity-aware action
- 更高或自適應的 movement cost

目標是讓改善版不只會預測新尖峰，也能減少每小時反覆搬動資源。

---

## 10. 實驗產出

- `dynamics_analysis_3way/dynamics_report.md`：三組固定排程分析
- `dynamics_analysis_3way/dynamics_summary.csv`：三組逐 seed 統計
- `shifted_analysis/shifted_report.md`：完整 shifted-peak 報告
- `shifted_analysis/shifted_summary.csv`：每個 checkpoint／schedule 統計
- `shifted_analysis/per_evaluation_seed.csv`：600 episodes 明細
- `shifted_analysis/figures/utility_regret.png`
- `shifted_analysis/figures/new_vs_old_peak_actions.png`
- `shifted_analysis/figures/action_sequence_agreement.png`
- `run_shifted_evaluations.py`：evaluation-only runner
- `analyze_shifted_evaluations.py`：shifted 結果分析器

所有新增程式與報告都位於 `experiments/proactivity_ablation/`，沒有修改 `prorl/` 核心程式。
