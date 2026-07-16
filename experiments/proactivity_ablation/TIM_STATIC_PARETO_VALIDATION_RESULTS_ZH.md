# TIM Validation-selected Static Pareto（K=50..201）結果

## 結論

完整求解 training allowed violations `K=50..201`，並在凍結 training optimal objectives 後，僅使用 validation split 解決 allocation ties。152 個 K 最終形成 57 個 unique allocations、17 個 evaluation nondominated static points。

結果確認：

- Validation tie-break 改變了 11/152 個 K 的 allocation，但沒有改變 K=85。
- 在 evaluation violations <=34 的 static points 中，最低 surplus 仍是 K=85 的 11406。
- 在 evaluation surplus <=10072.8 的 static points 中，最低 violations 是 K=180/181 的 53，surplus 9733。
- Dynamic exhaustive mean 為 34.7 violations、surplus 10072.8、42.6 movements。

因此，在這個固定 evaluation week，dynamic exhaustive 的確擴展了 validation-selected static 的 `SLA violations–surplus` 邊界：它能在約 35 violations 時，把 surplus 降到 static 尚未達到的水準。但它付出 movement、較大的 aggregate gap，且仍只是一個 one-step heuristic。

最嚴謹的表述是：

> Dynamic relocation 在此 evaluation week 可能具有降低 overprovisioning 的價值；尚未證明 relocation 在不同 traffic weeks 都必要，更沒有證明 RL 必要。

## 關鍵比較

### 相近 SLA

| Method | Utility | Gap | Surplus | Violations | Moves |
|---|---:|---:|---:|---:|---:|
| Static K=85 | 12.780 | **-90** | 11406 | **34** | **0** |
| Dynamic exhaustive mean | **18.921** | -293.4 | **10072.8** | 34.7 | 42.6 |

相對 K=85 static，dynamic：

- surplus 減少 1333.2（11.69%）；
- utility 增加 6.141；
- violations 平均增加 0.7；
- aggregate gap 惡化 203.4 units；
- movements 增加 42.6。

兩者沒有互相支配。若營運目標允許約 21% violation rate，dynamic 的 surplus 優勢可能有價值；是否值得 42.6 次搬移，仍需明確 movement cost／限制。

### 相近 surplus 上限

Static 要將 surplus 壓到 dynamic 的 10072.8 以下，目前最低 violations point 是：

| Method | Utility | Gap | Surplus | Violations | Moves |
|---|---:|---:|---:|---:|---:|
| Static K=180/181 | 25.640 | **-276** | **9733** | 53 | **0** |
| Dynamic exhaustive mean | 18.921 | -293.4 | 10072.8 | **34.7** | 42.6 |

Dynamic 以 movements 換得約 18.3 個較少的 violation hours；static 則有較低 surplus、更高 utility、較小 aggregate gap。這再次顯示不能只用單一 scalar score或單一 SLA 指標宣稱勝負。

## Validation tie-break 的影響

Validation selection 改變 11 個 K，但 K=85 allocation 與 evaluation 完全不變。它修正了部分任意 solver ties，例如 K=68/69 最終選到相同且 validation 較好的 allocation，使 evaluation 結果不再由 CP-SAT 任意 witness 決定。

流程為：

1. Phase A 只用 training，依序最小化 training surplus、deficit、node units。
2. Phase V 凍結上述 exact optima，只在同分 allocations 中最大化 validation satisfied hours，再最小化 validation surplus、deficit。
3. 最後依固定 node 順序做 12-stage lexicographic tie-break，取得唯一 allocation。
4. Phase E 才讀 evaluation，使用 frozen WaitAgent 評估。

這是合法的 train/validation model selection，沒有 evaluation leakage，比任意 solver witness 更適合作為強 static baseline。

## Static frontier 的其他重要結果

K=196–199 static 得到：

- utility：29.653；
- remaining gap：-248；
- surplus：9536；
- violations：55；
- movements：0。

它的 evaluation utility 甚至略高於先前 train-utility-best-static 的 29.527，也高於 dynamic exhaustive 的 18.921；同時 gap、surplus、movements 都優於 dynamic，只有 SLA violations 較差。這不是 evaluation optimization，而是不同 train/validation objective 在未見 evaluation week 上得到的結果。

因此：

- 若只追求 scalar utility，static 仍足夠，沒有 RL 必要性。
- 若要求嚴格 SLA，較保守 static 仍較好。
- 只有在中度 SLA 約束、且重視降低 surplus 時，dynamic relocation 顯示出明確潛力。

## 為何仍不能宣稱 relocation 必要

### 單一 evaluation week

所有 static 與 dynamic comparisons 都只使用同一個 168-hour week。Validation selection 降低了 allocation tie 的不穩定性，但沒有提供跨週泛化證據。

### Dynamic exhaustive 不是 oracle

它受單步 action、reward weights 與 tie-breaking 影響。10 個 seeds 在同一 traffic week 上仍產生 33–37 violations、41–44 movements。它既不是完整 dynamic upper bound，也不是 RL。

### SLA epsilon 尚未由營運需求指定

在 <=10% violations 的要求下，dynamic exhaustive 不合格；在約21%門檻下，dynamic 與 K=85 static 才同時可行。只有事先指定 constraint，才能比較可行方案的 surplus 與 movement cost。

## 完整性核對

- Archive SHA256：`6bddd1b07f3a8c205bf5d3a0ad5fed2eb1fe21d853a213a0369f5d07b6a94235`。
- 152/152 thresholds 完成。
- Phase A：所有 stages `OPTIMAL`，training-only forbidden accesses 0。
- Phase V：所有 validation 與 12-stage deterministic tie-break stages `OPTIMAL`，evaluation/full-data/old-results accesses 0。
- Train hash：`4c331ce63d2bb6d5adb9ae83da2696ceb4edcdd59d1df95199c60dfc762b166c`。
- Validation hash：`77953335683684f3d25412637979110fc2dfd727c841b7c8e55f247af0698331`。
- Evaluation hash：`1c7921a66e7cad62e57acfed754f4f21cfce18f18ecca412f713c466b8567e12`。
- Index hash：`668531e90403274e789022abeed26e3f0ee70148828a391a39a8c2648c17352f`。
- Phase A 與 Phase V synthetic brute-force checks通過。
- K=50、100、201 training anchors 完全重現。
- 57 個 unique frozen allocations 全部完成 168 hours。
- 所有 allocation node + pool = 180；learning updates、moves/add/remove/cost 全為 0。
- Seeds 1000/1900 完全相同，只作 determinism check，不是獨立統計樣本。

## 下一步

Static control 已足夠強，不需要繼續細化單一週的 K。下一個真正有判別力的步驟是 rolling-origin multi-week evaluation：

1. 以連續過去 weeks 做 training；
2. 下一週作 validation selection；
3. 再下一週作 frozen evaluation；
4. 每個 evaluation week 比較 validation-selected static、dynamic exhaustive，以及 dynamic oracle/MPC；
5. 使用不重疊 evaluation weeks 作為獨立 workload samples。

只有 dynamic oracle/MPC 在多週都穩定形成 static 無法達到的 SLA–surplus–movement frontier，才進入 constrained Forecast-PRORL。
