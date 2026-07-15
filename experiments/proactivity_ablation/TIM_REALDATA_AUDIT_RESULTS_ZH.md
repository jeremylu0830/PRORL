# TIM 真實資料 Necessity Audit 結果

## 結論

這個 benchmark **不是完全不需要動態調度**，但動態調度的價值高度依賴 reward weights，而且 scalarized utility 有時會獎勵明顯更差的 SLA。

- 在 `(gap, surplus, cost) = (0.6, 0.3, 0.1)`，`wait` 與 one-step exhaustive search 完全相同：最佳逐步行為是不搬。
- 在 `(0.9, 0.1, 0)`，`wait` 同時有最高 utility 與最低 SLA violations。
- 在 `(0.75, 0.2, 0.05)`，exhaustive search 是唯一同時明顯改善 utility 與 SLA 的設定，證明這個固定 evaluation week 確實存在可利用的 relocation opportunity。
- 在 `(0.35, 0.6, 0.05)`，exhaustive search 雖大幅提高 utility，卻讓 SLA violation 從 64 小時增加到平均 135.6 小時。這不是營運改善，而是高 surplus weight 導致 objective 與 SLA 錯位。

因此不能宣稱「真實 TIM trace 根本不需要 RL」，但也不能用這個 benchmark 無條件支持複雜 RL。較精確的結論是：**兩個設定偏好 static allocation；一個設定支持 relocation；另一個設定暴露 reward misalignment。**

## 實驗完整性

- Protocol：`tim-realdata-necessity-audit-v1`
- Git commit：`ebef86bd89c9de7cb7af2205eb788714dfd1baec`
- 4 reward weights × 5 agents × 10 evaluation seeds = 200 jobs
- 200/200 passed，每個 job 完成 168 個 hourly steps
- learning updates 全為 0
- `wait` movement count/cost 全為 0
- dataset/index hashes 執行前後一致
- Source archive SHA256：`e9a2d0208ef74568571d7385e008a66cbf53e399f2c86d781e362040a49692c7`

遠端 CSV 將前三個 weight 欄位命名為 `weight_utility, weight_gap, weight_cost`，但原 reward 實際傳入順序是 `[remaining gap, surplus, movement cost]`。本分析輸出已更正為 `gap_weight, surplus_weight, cost_weight`。

## 絕對結果

數值是十個 evaluation seeds 的平均。Remaining gap 越接近 0 越好；surplus、cost、moves、SLA violations 越低越好。

| Weights (gap/surplus/cost) | Agent | Utility | Remaining gap | Surplus | Cost / moves | SLA OK | Violations |
|---|---|---:|---:|---:|---:|---:|---:|
| 0.60/0.30/0.10 | **wait** | **-21.180** | **-1095.0** | 7341.0 | **0.0** | **104.0** | **64.0** |
|  | exhaustive | **-21.180** | **-1095.0** | 7341.0 | **0.0** | **104.0** | **64.0** |
|  | greedy | -25.440 | -2713.0 | **3762.0** | 222.0 | 87.0 | 81.0 |
|  | sampling | -29.625 | -1053.1 | 7396.1 | 180.3 | 92.2 | 75.8 |
|  | random | -93.560 | -2149.8 | 8080.1 | 308.2 | 45.2 | 122.8 |
| 0.75/0.20/0.05 | wait | 16.330 | -1095.0 | **7341.0** | **0.0** | 104.0 | 64.0 |
|  | **exhaustive** | **18.921** | **-293.4** | 10072.8 | 42.6 | **133.3** | **34.7** |
|  | greedy | -22.413 | -2713.0 | 3762.0 | 222.0 | 87.0 | 81.0 |
|  | sampling | 18.374 | -626.7 | 8556.7 | 206.3 | 104.8 | 63.2 |
|  | random | -53.967 | -2149.8 | 8080.1 | 308.2 | 45.2 | 122.8 |
| 0.90/0.10/0.00 | **wait** | **53.840** | **-1095.0** | 7341.0 | **0.0** | **104.0** | **64.0** |
|  | exhaustive | -21.011 | -2717.1 | 3968.7 | 237.0 | 80.6 | 87.4 |
|  | greedy | -19.387 | -2713.0 | **3762.0** | 222.0 | 87.0 | 81.0 |
|  | sampling | 5.740 | -2147.6 | 5082.4 | 279.6 | 78.1 | 89.9 |
|  | random | -14.374 | -2149.8 | 8080.1 | 308.2 | 45.2 | 122.8 |
| 0.35/0.60/0.05 | wait | -148.310 | **-1095.0** | 7341.0 | **0.0** | **104.0** | **64.0** |
|  | **exhaustive** | **-41.590** | -5800.8 | **1871.7** | 88.4 | 32.4 | 135.6 |
|  | greedy | -48.493 | -2713.0 | 3762.0 | 222.0 | 87.0 | 81.0 |
|  | sampling | -68.061 | -3156.4 | 3982.3 | 239.2 | 59.2 | 108.8 |
|  | random | -210.183 | -2149.8 | 8080.1 | 308.2 | 45.2 | 122.8 |

## Paired evidence versus wait

### Weights 0.60/0.30/0.10

Exhaustive 與 wait 在所有 objective、actions 與 SLA 完全相同。Sampling 的 aggregate gap 平均改善 41.9 units，但 utility 下降 8.445（95% CI `[-11.327, -5.563]`），SLA violations 增加 11.8 小時（95% CI `[5.875, 17.725]`）。這表示小幅 aggregate gap 改善不足以支付 surplus/cost，且沒有改善逐時 SLA。

### Weights 0.75/0.20/0.05

Exhaustive 相對 wait：

- utility `+2.591`，95% CI `[2.227, 2.955]`；
- remaining gap `+801.6`（更接近 0）；
- satisfied hours `+29.3`，95% CI `[28.286, 30.314]`；
- 代價是 surplus `+2731.8` 與平均 42.6 次 movements。

這是四組 weights 中唯一清楚支持 relocation necessity 的結果。Sampling utility 平均 `+2.044`，但 95% CI `[-1.201, 5.289]`、exact sign-flip `p=0.1914`，只靠 action sampling 並不穩定。

### Weights 0.90/0.10/0

所有 relocation baselines 的 utility 與 SLA 都比 wait 差。表現最好的 relocation baseline（sampling）相對 wait utility 仍為 `-48.100`，SLA violations 增加 25.9 小時。這個設定沒有提供動態搬移的必要性證據。

### Weights 0.35/0.60/0.05

Exhaustive 相對 wait utility `+106.720`，主要來自 surplus 減少 `5469.3`；但 remaining gap 惡化 `4705.8`、satisfied hours 減少 `71.6`，SLA violations 達平均 135.6/168 小時。Utility 的大幅提升與服務品質方向相反，直接支持將 SLA 改成 constraint，而不是只放在線性 scalarization 中。

## 統計限制

這十個 seeds 共用**同一個固定 TIM evaluation week**。它們只改變 random actions、sampling 與 tie-breaking，並不是十個獨立 traffic samples。因此：

- confidence intervals 與 exact sign-flip p-values只描述 policy RNG sensitivity；
- 不能據此推論到其他週、其他區域或未見需求分布；
- `wait` 與 deterministic greedy 的十次結果完全重複，對這類 contrast 報窄 CI 或 `p=0.002` 會造成偽重複。本分析將其標為 descriptive only，不做推論解讀。

## 對論文改進方向的影響

1. **先改 formulation**：0.35/0.60/0.05 證明 scalarized utility 可以犧牲 SLA 換 surplus。下一個演算法方向應是 Constrained MDP／Lagrangian SLA constraint。
2. **保留 WAIT/MOVE hierarchy**：兩組 weights 選擇 static，說明 agent 首先需要可靠判斷是否搬移，而不是每小時強迫產生 relocation decision。
3. **只在 movement-necessary slice 評估 forecast**：0.75/0.20/0.05 是目前最有價值的真實資料設定，可用來比較 PRORL、Forecast-PRORL 與 MPC。
4. **建立 train-only best-static**：目前 wait 只保留原始 uniform initial allocation；用 training split 擬合的 static allocation 會是更強且無 leakage 的 baseline。
5. **增加 workload/capacity variation**：下一步應跑 load 0.6/0.8/1.0/1.2，並使用多個 evaluation windows。否則不能判斷 relocation value 是普遍現象或單週特例。
6. **原 PRORL 尚未在此 audit 中比較**：這些結果只建立 decision-problem baseline，不能用來宣稱 PRORL 或 Forecast-PRORL 的優劣。

## 可重現產物

- `tim_realdata_audit_analysis/absolute_summary.csv`
- `tim_realdata_audit_analysis/paired_contrasts_vs_wait.csv`
- `analyze_tim_realdata_audit.py`

原始 200-job archive 保留在 workspace root，未加入 Git，避免把遠端 logs 與絕對路徑納入 source history。
