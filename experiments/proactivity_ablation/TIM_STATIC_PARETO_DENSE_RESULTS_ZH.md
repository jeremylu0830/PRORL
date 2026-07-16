# TIM Dense Static Pareto（K=50..100）結果

## 結論

Dense train-only experiment 完整求解 allowed training violations `K=50..100`，共 51 個 thresholds、25 個 unique allocations，其中 6 個在 evaluation 的 `violations–surplus–movements` 指標下不被其他 static points 支配。

最接近 dynamic exhaustive SLA 的固定配置是 `K=85`：

- static：34/168 violations、surplus 11406、remaining gap -90、utility 12.780、moves 0；
- dynamic exhaustive mean：34.7/168 violations、surplus 10072.8、remaining gap -293.4、utility 18.921、moves 42.6。

相對 K=85 static，dynamic exhaustive 以平均多 0.7 個 violation hours、較差 203.4 units aggregate gap及 42.6 次搬移，換取 surplus 減少 1333.2（11.69%）與 utility 增加 6.141。

因此 dense result 提供了第一個較清楚的訊號：**在約 20% violation rate 附近，動態共享資源可能降低固定 overprovisioning**。但它仍沒有證明 RL 必要；目前做到這件事的是 one-step exhaustive baseline，而且結論只來自單一 evaluation week。

## Evaluation nondominated static points

| K | Actual train violations | Node / pool units | Utility | Gap | Surplus | Eval violations | Moves |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 50 | 155 / 25 | 4.683 | -41 | 12202 | 25 | 0 |
| 68 | 68 | 153 / 27 | 8.847 | -46 | 11869 | 29 | 0 |
| 75 | 75 | 152 / 28 | 10.707 | -52 | 11706 | 31 | 0 |
| 85 | 85 | 150 / 30 | 12.780 | -90 | 11406 | 34 | 0 |
| 91 | 91 | 150 / 30 | 14.427 | -64 | 11380 | 35 | 0 |
| 100 | 100 | 148 / 32 | 16.943 | -95 | 11073 | 38 | 0 |

在 sampled dense range 中：

- evaluation violations <=34 的最低 surplus 是 K=85 的 11406；
- 沒有 static point 同時達到 violations <=34.7 與 surplus <=10072.8；
- 沒有 static point在 violations、surplus、movements 三項同時支配 dynamic mean；
- dynamic mean 也沒有支配 K=85，因為它 violations 與 movements 較多。

## Dynamic exhaustive 的 seed variability

Dynamic exhaustive 的 10 個 seeds 使用同一 traffic week，只改變 policy RNG／tie-breaking。Violations 範圍為 33–37、surplus 為 10009–10112、moves 為 41–44；其中 4/10 runs 的 violations <=34。這表示 dynamic 結果在同一 workload 上仍受 action tie-breaking 影響。

這些 seeds 不是 10 個獨立 traffic samples，因此不能把平均值或窄 CI 當作跨週泛化證據。與 deterministic static 的比較只能描述這一週的策略變異。

## 為何還不能把它稱為完整 static frontier

### 1. K=101..201 尚未密集補完

K=100 static 的 surplus 仍為 11073；先前 K=201 point 才降至 9417。Dynamic 的 surplus 10072.8 落在兩者之間。由於 evaluation violations 對 K 並非單調，尚不能排除 K=101..201 中存在 `surplus <=10072.8` 且 `violations <=34` 的 static point。

### 2. Training optimum 存在 allocation ties

例如 K=65、68、69 的 training surplus、deficit 與 node units 完全相同，卻得到不同 allocation；其 evaluation violations 分別為 31、29、34。K=68 allocation 對 K=69 仍可行且 training objectives 相同，但 CP-SAT 在 K=69 任意回傳了另一個 generalization 較差的 optimum。

因此單一 solver solution 不是該 training objective 下所有 static allocations 的能力上界。後續必須加入固定且無 evaluation leakage 的 tie-break，例如：

1. 先凍結 training 的 surplus、deficit、units optimum；
2. 使用 validation split 最大化 satisfied hours；
3. 再以 validation surplus、deficit做 deterministic tie-break；
4. 最後只在 evaluation split 評估一次。

這是正常的 train/validation model selection，不是 evaluation leakage，也會形成比任意 CP-SAT tie 更公平的 static baseline。

### 3. 尚未定義營運 SLA epsilon

若要求 violation rate <=10%，K=85 static 與 dynamic exhaustive 都不合格，較保守 static 已可達標。若 SLA 約為21%，兩者才都可行，此時 dynamic 的 surplus 優勢才有營運意義。

## 對論文方向的影響

目前可支持：

1. 原始 uniform wait 不是充分的 static baseline；
2. scalar utility 可由 utility-best-static 擊敗，不能用它證明 RL；
3. 嚴格 SLA 可由 static 達成，dynamic exhaustive 未必更好；
4. 在較寬鬆 SLA 下，dynamic relocation 可能以搬移成本換取較低 surplus；
5. 這提供研究 dynamic scheduling 的理由，但仍不提供使用 RL、forecast 或特定 PRORL 架構的理由。

所以演算法順序仍應是：強 static baseline → dynamic oracle/MPC → constrained Forecast-PRORL。只有 dynamic oracle/MPC 在多週 evaluation 上穩定形成 static 無法達到的 frontier，RL 改進才有充分問題基礎。

## 完整性核對

- Archive SHA256：`b987c3b2a067421c7a92500bf32800d3123e47cf7c54c247349fbebc072dbe03`。
- 51/51 thresholds 完成，25 unique allocations，6 nondominated static points。
- Training：1008 hours、12 nodes、12096 rows。
- Training hash 前後一致；forbidden access attempts 0。
- 51 × 3 solver stages 全為 `OPTIMAL`，objective 等於 best bound。
- Synthetic CP-SAT 與完整 brute-force optimum 一致。
- K=50、K=100 的 training 與 evaluation anchors 完全重現。
- 所有 allocation 都滿足 node + pool = 180。
- 25 個 frozen evaluations 全部完成 168 hours；WaitAgent、learning updates 0、moves/add/remove/cost 0。
- Evaluation/index/allocation hashes 前後一致。
- Seeds 1000/1900 完全相同，只作 determinism check。

## 下一步

後續 K=50..201 與 validation-only tie selection 已完成，結果見 `TIM_STATIC_PARETO_VALIDATION_RESULTS_ZH.md`。Validation 改變 11/152 allocations，但 K=85 未變；dynamic exhaustive 仍位於 validation-selected static 的 SLA–surplus 邊界之外。

不要立即訓練 Forecast-PRORL。單一週的 static control 已足夠，下一步改為：

1. rolling-origin multi-week evaluation；
2. 每週比較 validation-selected static、dynamic exhaustive 與 dynamic oracle/MPC。

只有動態方法在多個未見 weeks 上穩定勝過 static reference，才進入 constrained Forecast-PRORL。
