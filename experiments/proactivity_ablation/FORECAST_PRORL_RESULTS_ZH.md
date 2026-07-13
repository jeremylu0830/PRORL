# Forecast-No-Time PRORL：正式實驗結果

> HackMD 相容版本。日期：2026-07-14。

## 1. 結論摘要

這次實驗得到的是一個重要但需要精確表述的**部分成功**：

1. **Forecast-No-Time 確實學會使用 forecast 改變資源配置時機。**
2. 在 `+6h`、`+1d` 與 unseen shifted schedules，Forecast-No-Time 都能在新 peak 發生時，將比平時更多的資源放到正確節點。
3. 相較 Original，Forecast-No-Time 在三種 shifted schedules 的 event-allocation lift 分別多 `0.568`、`0.533`、`0.384` units，exact paired p-value 分別為 `0.0039`、`0.0039`、`0.0059`。
4. 但是，這些較正確的 anticipatory allocations **沒有顯著提高 utility 或 remaining gap**。
5. 因此目前不能宣稱「加入 forecast 改善 PRORL 整體效能」；可以宣稱「explicit forecast 產生了可跨時間位移轉移的 anticipatory behavior，但現有 reward／action／learning design 無法穩定把它轉換成端到端績效」。

這個結果比單純報告 reward 更有研究價值：它把「有沒有使用 forecast」與「使用後有沒有得到更高分」分開回答。

## 2. 結果完整性稽核

收到的壓縮檔：

```text
forecast-no-time-results.tar.gz
size: 約 46 MiB
SHA-256: b6831d34535d8b136a1226ef389391c76ca77db94f0cd24f854885007f3ccecf
```

正式結果根目錄：

```text
data/models/results/
proactivity-ablation-forecast_no_time-scheduled_at=13-07-2026_23-13-31/
```

稽核結果：

- training seeds 10–19：10/10 完整
- 每個 seed 都有 `full_data.json`
- 每個 seed 都有 `agent_state.pth`
- 每個 seed 都有 `best_validation_agent_state.pth`
- 30 個正式結果檔案均非空
- 10 個 JSON 均能完整解析
- 未發現 `NaN`、`Infinity`、`Traceback` 或 `ERROR`
- 每個 evaluation action/reward history 都是 168 steps
- config training iterations：172800
- tracker total training steps：175799，包含 bootstrap 階段
- 所有正式 run 的 agent type：`prorl-no-split`
- state 包含 `node-demand-forecast`
- forecast horizon：6
- forecast type：`oracle`
- 不含 `time-encoded`

另外重新執行修正後的 smoke test，確認 smoke 與 scheduler 使用相同的 full-space agent：

```text
scheduled config key: double-dqn-full-space
actual agent type: prorl-no-split
status: passed
changed tensors: 25 / 73
```

## 3. 評估方法

### 固定 schedule

比較四組各 10 個 training seeds：

1. Original
2. No-Time
3. Calendar-Only
4. Forecast-No-Time

正式 training 結果只保存 evaluation seed 1000，因此固定 schedule 的推論單位是 10 個 paired training seeds，不能把 168 個 time steps 當成獨立樣本。

### Shifted-peak evaluation

使用 40 個 frozen best-validation checkpoints：

```text
4 conditions × 10 training seeds
```

每個 checkpoint 評估：

```text
4 schedules × 5 evaluation seeds
```

合計：

```text
160 evaluation jobs
800 evaluation episodes
```

Schedules：

- control
- shift_plus_6h
- shift_plus_1d
- unseen

所有 shifted evaluations：

- 只做 inference
- 載入同一個 frozen checkpoint
- 明確禁止 `learn()`
- 不重新訓練
- 每組使用相同 evaluation seeds：1000、1100、1200、1300、1400

## 4. 固定 schedule 結果

| Condition | Utility | Remaining gap | Gap hours | Satisfied hours | Deadline target hit | Event target allocation |
|---|---:|---:|---:|---:|---:|---:|
| Original | 138.640 ± 3.147 | -32.700 ± 43.469 | 28.300 ± 41.897 | 138.900 ± 41.821 | 0.750 ± 0.354 | 4.000 ± 0.707 |
| No-Time | 138.643 ± 1.965 | -24.000 ± 34.788 | 18.600 ± 34.996 | 149.400 ± 34.996 | 0.300 ± 0.258 | 3.300 ± 0.258 |
| Calendar-Only | 137.739 ± 3.174 | -45.900 ± 44.953 | 41.400 ± 43.671 | 125.800 ± 43.310 | 0.800 ± 0.350 | 4.000 ± 0.745 |
| Forecast-No-Time | 138.308 ± 1.353 | -37.900 ± 21.789 | 33.000 ± 21.736 | 134.600 ± 21.645 | 0.450 ± 0.369 | 3.700 ± 0.350 |

Forecast-No-Time 相較 Original：

```text
utility difference:      -0.332, p=0.7715
remaining-gap difference: -5.200, p=0.7617
```

Forecast-No-Time 相較 No-Time：

```text
utility difference:       -0.335, p=0.6211
remaining-gap difference: -13.900, p=0.3281
```

固定 schedule 上沒有證據顯示 Forecast-No-Time 的整體 objective 優於 Original 或 No-Time。

## 5. Shifted-peak 結果

### 5.1 Forecast-No-Time 本身

| Schedule | Utility | Utility regret | New-peak hit | Old-slot action | Action agreement | Event allocation lift |
|---|---:|---:|---:|---:|---:|---:|
| control | 138.355 ± 1.368 | 0.000 ± 0.000 | 0.450 ± 0.369 | 0.450 ± 0.369 | 1.000 ± 0.000 | 0.544 ± 0.366 |
| shift_plus_6h | 138.222 ± 1.443 | 0.133 ± 0.078 | 0.450 ± 0.369 | 0.250 ± 0.264 | 0.851 ± 0.142 | 0.549 ± 0.366 |
| shift_plus_1d | 137.842 ± 1.714 | 0.512 ± 0.354 | 0.450 ± 0.369 | 0.200 ± 0.258 | 0.790 ± 0.104 | 0.564 ± 0.370 |
| unseen | 137.370 ± 4.511 | 0.985 ± 5.044 | 0.400 ± 0.394 | 0.250 ± 0.354 | 0.610 ± 0.252 | 0.466 ± 0.359 |

`action agreement` 是 shifted schedule 與 control 的 add/remove action pair 完全相同比例。數字愈低，代表 policy 愈會因 workload timing 改變而調整動作。

Calendar-Only 的 agreement 永遠是 `1.000`；Original 在三種 shifts 為 `0.959`、`0.986`、`0.976`；Forecast-No-Time 則下降到 `0.851`、`0.790`、`0.610`。這顯示 Forecast-No-Time 對 schedule 改變明顯更敏感。

### 5.2 最關鍵的 anticipatory allocation 證據

Event allocation lift 定義為：

```text
新 peak 發生時，stressed node allocation
−
同一條 episode 中該 target node 的 all-hour average allocation
```

Forecast-No-Time 的 lift 在所有 schedules 都顯著大於 0：

```text
control:        0.544, p=0.0059
shift_plus_6h:  0.549, p=0.0059
shift_plus_1d:  0.564, p=0.0059
unseen:         0.466, p=0.0059
```

相較 Original：

```text
shift_plus_6h:  Forecast lift 多 0.568 units, p=0.0039
shift_plus_1d:  Forecast lift 多 0.533 units, p=0.0039
unseen:         Forecast lift 多 0.384 units, p=0.0059
```

在 PRORL 的 timing 中，configured stress hour `h` 對應 action/reward history index `h-1`。也就是 agent 必須在 demand 真正跳升以前選擇動作。No-Time 沒有 calendar 或 forecast，無法從當前 calm demand 知道下一步會 stress；Forecast-No-Time 的顯著 allocation lift 因此是 explicit anticipatory behavior 的直接證據。

### 5.3 但 objective 沒有改善

Forecast-No-Time 相較 Original：

| Schedule | Utility difference | p | Remaining-gap difference | p |
|---|---:|---:|---:|---:|
| shift_plus_6h | -0.040 | 0.9863 | -4.980 | 0.7852 |
| shift_plus_1d | -0.408 | 0.7246 | -11.220 | 0.5527 |
| unseen | -0.862 | 0.5117 | +8.740 | 0.5312 |

沒有一個 shifted schedule 顯示 Forecast-No-Time 在 utility 或 remaining gap 上顯著優於 Original。

Forecast-No-Time 的 seven-hour target-add rate也沒有顯著優於 Original。這表示它確實改變配置軌跡，並在 event 時保有更多正確資源，但動作不夠精準或代價結構不利，未形成 objective gain。

## 6. 對原研究問題的回答

### Q1：Original 是否主要記住固定時間表？

**是，證據仍然支持。**

- Original 在 control 的 deadline hit 為 0.750。
- shift 後仍在 obsolete old slots 動作，rate 維持 0.750。
- shifted action agreement 高達 0.959–0.986。
- shifted event allocation lift 接近 0。
- Calendar-Only 保留 100% control actions，更直接顯示 calendar memorization。

### Q2：Forecast-No-Time 是否真的使用 forecast？

**是。**

- shifted action agreement 顯著下降。
- obsolete old-slot actions 降至 0.200–0.250。
- 新 event allocation lift 在所有 schedules 顯著大於 0。
- lift 在三種 shifted schedules 都顯著高於 Original。

### Q3：Forecast 是否改善整體效能？

**目前沒有。**

- 固定與 shifted schedules 的 utility／remaining-gap improvement 都不顯著。
- unseen schedule 變異反而更大。
- seed 15 unseen episode 出現高 surplus，導致 utility 約 124.97，是需要追查的 instability case。

## 7. 合理解釋

目前最合理的解釋不是「forecast 沒用」，而是：

1. Forecast signal 足以改變 policy 的 allocation timing。
2. Agent 幾乎每一步都在移動資源；四組 active add/remove subaction rate 都接近 1。持續移動使「正確提前配置」未必帶來額外效益。
3. 每小時只能移動有限數量的資源，可能限制 forecast 的實際價值。
4. 24 維 raw horizon forecast 對 DQN 而言較稀疏；policy 可能只學到粗略 bias，沒有學到精準 quantity/timing。
5. Reward 同時混合 gap、surplus 與 movement cost。提前配置改善 demand readiness，但可能被 surplus 或不必要 movement 抵銷。
6. Training 只使用固定 schedule；即使拿到 forecast，網路仍可能學到特定 pattern，而不是一般 forecast-to-action mapping。

## 8. 下一步優先順序

### 第一優先：Forecast usage inference ablation

不重新訓練，對相同 Forecast-No-Time checkpoint 比較：

- 正常 Oracle forecast
- calm/masked forecast
- node-permuted forecast
- time-shuffled forecast

比較 Q/action、event allocation lift 與 utility。這能確認網路究竟使用 forecast 的哪一部分。

### 第二優先：Randomized-schedule training

訓練期間隨機化 peak day/hour，evaluation 使用 held-out schedules。若 Forecast-No-Time 真正學會一般 forecast-to-action mapping，它應在 held-out schedule 保持 allocation lift，並比無 forecast policy 更穩定。

這比立即換 LSTM 更重要：Schedule Oracle 已是最乾淨的 upper bound；若 policy 無法把 Oracle 穩定轉成 reward，加入有誤差的 LSTM 只會讓問題更難解釋。

### 第三優先：讓 action 與 reward 能利用提前資訊

- quantity-aware action：node → quantity
- 限制或懲罰無效的 continuous movement
- 將 SLA remaining gap 改成 constrained objective
- 分開報告 satisfaction gain 與 movement/surplus cost

## 9. 論文可使用的結論文字

> The original calendar-conditioned policy retained 95.9–98.6% of its control action sequence under temporal demand shifts and continued targeting obsolete peak slots, supporting fixed-schedule memorization. In contrast, Forecast-No-Time changed its actions substantially and produced a significant stressed-node allocation lift on all shifted schedules. Relative to Original, this lift increased by 0.568, 0.533, and 0.384 resource units for the +6-hour, +1-day, and unseen schedules, respectively. However, these behavioral improvements did not yield statistically significant gains in utility or remaining demand gap. Explicit forecast information therefore enabled transferable anticipatory behavior, but the current reward, action space, and learning design did not reliably convert that behavior into end-to-end performance improvements.

## 10. 可重現指令

固定 schedule 分析：

```bash
python experiments/proactivity_ablation/analyze_forecast_comparison.py \
  <Original roots> <No-Time roots> <Calendar-Only root> <Forecast-No-Time root> \
  --output-dir experiments/proactivity_ablation/forecast_analysis
```

Shifted evaluation：

```bash
ENV=test python experiments/proactivity_ablation/run_shifted_evaluations.py \
  <Forecast-No-Time root> \
  --output-dir experiments/proactivity_ablation/shifted_evaluations \
  --conditions Forecast-No-Time \
  --training-seeds 10 11 12 13 14 15 16 17 18 19 \
  --evaluation-seeds 1000 1100 1200 1300 1400 \
  --schedules control shift_plus_6h shift_plus_1d unseen \
  --checkpoint best
```

四組 shifted 分析：

```bash
python experiments/proactivity_ablation/analyze_shifted_evaluations.py \
  experiments/proactivity_ablation/shifted_evaluations \
  --output-dir experiments/proactivity_ablation/forecast_shifted_analysis
```
