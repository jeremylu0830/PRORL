# Forecast-No-Time PRORL：實作與實驗紀錄

> 本文件使用 HackMD 相容 Markdown，可直接貼入 HackMD。

## 1. 目標

本次修改用來檢驗 PRORL 的「proactivity」是否能由明確的未來需求資訊產生，而不是只依賴 `time-encoded` 記住固定的需求時間表。

新增的實驗條件為 **Forecast-No-Time PRORL**：

- 移除 `time-encoded`
- 新增 `node-demand-forecast`
- 預測範圍為未來 6 小時
- 第一版使用 Schedule Oracle
- 保留原始 PRORL；未啟用 forecast 的設定不改變行為

## 2. State 設計

原本主要 state：

```text
pool-capacity
node-capacity
node-demand
node-delta
time-encoded
current-lives
node-add
```

Forecast-No-Time state：

```text
pool-capacity
node-capacity
node-demand
node-delta
node-demand-forecast
current-lives
node-add
```

`node-demand-forecast` 採 node-major 排列：

```text
[
  node_0(t+1), ..., node_0(t+6),
  node_1(t+1), ..., node_1(t+6),
  node_2(t+1), ..., node_2(t+6),
  node_3(t+1), ..., node_3(t+6)
]
```

目前為 4 個 nodes、1 個 resource、horizon 6，因此新增：

```text
4 × 1 × 6 = 24 dimensions
```

Forecast 使用與 `node-demand` 相同的 resource-unit conversion、`ceil` 與 normalization 邏輯。

## 3. Schedule Oracle 定義

第一版不是訓練 LSTM，而是從 synthetic workload 已知的 stress schedule 取得未來六小時的**期望需求**。

Oracle 的特性：

- 能預先看見 calm/stress demand transition
- 能模擬 `swap_stress` 狀態
- 能處理 synthetic model 的 distribution multiplier transition
- 不加入未來 Gaussian observation noise
- 不消耗原 demand generator 的 random stream
- 不修改原 demand generator 的 couple status

排除未來 noise 的原因，是避免 Oracle 偷看尚未發生的隨機數；它取得的是 schedule 與 expected demand，而不是完整未來 realization。

這仍然是 Oracle upper-bound，不能宣稱為可部署的 forecasting model。後續應以 seasonal-naive、LSTM 或其他需求預測器取代，並加入 forecast error sensitivity。

## 4. 修改檔案

### 核心程式

- `prorl/environment/state_builder.py`
  - 新增 `StateFeatureName.NodeDemandForecast`
  - 新增 forecast feature builder
  - 新增 feature-size 計算
- `prorl/environment/wrapper.py`
  - 只有 state 要求 forecast 時才呼叫 generator forecast
  - 將 forecast 傳入 state builder
- `prorl/emulator/models/abstract.py`
  - 新增 model forecast interface
- `prorl/emulator/models/synthetic_model.py`
  - 實作 non-mutating Schedule Oracle

### 實驗與測試

- `experiments/proactivity_ablation/generate_configs.py`
  - 新增 `forecast_no_time` variant 產生流程
- `experiments/proactivity_ablation/generated/forecast_no_time.yaml`
  - 正式 10-seed training config
- `experiments/proactivity_ablation/test_forecast_state.py`
  - compatibility 與 forecast integration tests
- `experiments/proactivity_ablation/run_forecast_smoke.py`
  - 96-step bounded smoke training
- `experiments/proactivity_ablation/monitor_forecast_smoke.py`
  - tmux background monitor
- `experiments/proactivity_ablation/README.md`
  - 更新執行方式

## 5. 相容性策略

Forecast 並未加入全域預設 state。只有 config 明確包含：

```yaml
- node-demand-forecast
```

時才會執行 forecast。因此：

- Original config 不受影響
- No-Time config 不受影響
- 舊 checkpoint 的 input dimension 不變
- Forecast-No-Time 使用新的 input dimension，必須重新訓練，不能載入舊網路權重

重新執行 config generator 後，既有 `original.yaml` 與 `no_time.yaml` 的 SHA-256 前後一致：

```text
original.yaml: d28bd9fe0aac2ea45ebcc263df484cf5b3cc57b33e872a66705ae598161c8939
no_time.yaml:  e33010c25621228225638bb7385a7baee7b1dad16bf3c4ebf42cb6bccb561a0e
```

另外使用修改前訓練完成的 No-Time seed 10 checkpoint，成功載入並完成 168-step control evaluation，證明舊 checkpoint 路徑仍可使用。

## 6. 測試結果

整合測試：

```bash
ENV=test python -m unittest \
  experiments.proactivity_ablation.test_forecast_state -v
```

結果：

```text
Ran 3 tests
OK
```

測試內容：

1. 舊 No-Time config 能初始化，state size 與 action sub-state 宣告一致
2. Forecast-No-Time 不含 `time-encoded`，新增維度為 24，node-major 數值正確
3. Oracle 能在 peak 前一小時看見需求跳升，而且 forecast 前後 RNG 與 generator status 不變

## 7. Smoke training 結果

Smoke test 使用：

```text
training steps: 96
bootstrap steps: 16
batch size: 16
forecast horizon: 6
```

結果：

```text
status: passed
elapsed: 1.67 seconds
forecast feature size: 24
changed agent tensors: 28 / 146
```

28 個 agent tensors 在 smoke training 後改變，表示流程不只是成功建立環境，而是真的進入 replay/learning update。

## 8. Git 版本

實作分支：

```text
forecast-prorl
```

實作 commit：

```text
a3bd6ed Add six-hour oracle forecast state ablation
```

遠端同步：

```bash
git fetch origin
git switch forecast-prorl
git pull --ff-only origin forecast-prorl
```

## 9. 正式訓練狀態

目前完成：

- [x] Forecast state 實作
- [x] Oracle horizon 6
- [x] Forecast-No-Time config
- [x] 相容性測試
- [x] 短 smoke training
- [x] 推送 GitHub `forecast-prorl`
- [ ] 遠端正式 10-seed training
- [ ] control 與 shifted-peak evaluation
- [ ] Original／No-Time／Forecast-No-Time 統計比較

正式訓練應只 schedule：

```text
experiments/proactivity_ablation/generated/forecast_no_time.yaml
```

不要直接 schedule 整個 `generated/`，否則 Original 與 No-Time 也會再次排入訓練。

## 10. 後續分析問題

正式結果完成後，應回答：

1. Forecast-No-Time 在原始 peak 上是否優於 No-Time？
2. Forecast-No-Time 在 shifted peaks 上是否優於 Original？
3. Agent 是否在 peak 前 1 至 6 小時提前搬移資源？
4. 改善是否來自更高的 demand satisfaction，而非明顯增加 movement cost？
5. Oracle 的改善幅度多大，能否作為真實 forecast model 的 performance upper-bound？

若 Forecast-No-Time 能在 shifted peaks 命中需求，而 Original 失敗，證據會支持：原始 PRORL 的 anticipatory behavior 主要依賴 calendar memorization；加入明確 forecast 才能提供對時間偏移的真正泛化能力。
