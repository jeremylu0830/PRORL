# TIM 真實資料 Benchmark Necessity Audit

## 目的

這個 audit 回答一個比「PRORL 能否得到高 reward」更基本的問題：在論文原始的 TIM 12-node evaluation protocol 下，資源搬移是否真的有必要？

若固定初始配置的 `wait` 已與 relocation policies 相當，便不能只因資料來自真實網路，就主張此 benchmark 證明複雜 RL scheduler 有實務必要。反之，若合理 relocation policy 在相同資料、容量與 reward 下穩定勝過 `wait`，才證明 decision problem 本身不是退化的。

## 保留與新增的內容

保留論文設定：

- TIM 12 nodes 與原始 train/validation/evaluation 時間切分；
- 168-hour evaluation episode；
- 四組 reward weights；
- evaluation seeds 1000–1900；
- 初始 allocation、action space、reward normalization 與 demand scaling；
- 作者提供的 random、greedy-optimal、sampling-optimal、exhaustive-search baselines。

只新增：

- `wait` agent：每一步都選 action space 的 wait index，整週保持原設定的初始 allocation；
- 獨立 audit config generator 與 preflight；
- 舊 baseline 名稱的相容映射。原始 YAML 與 PRORL 演算法均未改寫。

`wait` 是必要但不是充分的 static baseline。若它表現強，下一步仍要加入只用 training split 擬合的 `best-static`；不可用 evaluation week 選 static allocation，否則會 data leakage。

## 第一階段：原容量設定

生成四份 load=0.8 audit configs：

```bash
.venv/bin/python experiments/proactivity_ablation/generate_tim_realdata_audit.py --loads 0.8
```

每份 config 產生 50 runs：5 agents × 10 evaluation seeds。先執行 preflight：

```bash
.venv/bin/python experiments/proactivity_ablation/tim_realdata_preflight.py \
  --json-output experiments/proactivity_ablation/generated_tim_realdata_audit/preflight.json
```

只有 `ready_for_baselines: true` 才能排程。以其中一組權重為例，沿用 repository 的 multi-run scheduler：

```bash
.venv/bin/python prorl.py run scheduler multi-runs \
  -mcp experiments/proactivity_ablation/generated_tim_realdata_audit/load_0.8-1-12_nodes-weights_0.6_0.3_0.1-baselines.yaml \
  -q tim-realdata-audit
```

其餘三份 config 也必須執行，不能只報告對論點有利的權重。

## 第二階段：容量／負載敏感度

第一階段完成後再生成 load sweep：

```bash
.venv/bin/python experiments/proactivity_ablation/generate_tim_realdata_audit.py \
  --loads 0.6 0.8 1.0 1.2
```

這個 sweep 用來區分兩種結論：

1. `wait` 只在寬裕容量下足夠，接近容量邊界時 relocation 變得必要；
2. `wait` 在所有 load 都接近最佳，表示 action/reward/initial allocation 很可能使 benchmark 退化。

load sweep 不是調參。四個 load 都應完整報告，並以 load=0.8 作為論文原設定的主要結果。

## 必須報告的指標

不要只比較 scalarized utility。每組 agent、weight、load 至少報告：

- utility；
- remaining gap；
- surplus；
- movement cost 與 movement count；
- satisfied-node rate 或 SLA violation rate；
- evaluation seeds 的 paired difference 與 95% confidence interval。

同一 evaluation seed 的結果要成對比較。主要 contrasts：

- relocation baseline − wait；
- PRORL − wait；
- PRORL − 最佳非學習 relocation baseline。

PRORL 比較必須使用原 TIM checkpoint，或依原 training protocol 重訓後鎖定 checkpoint。不能用 synthetic randomized-schedule checkpoint 代替。

## 判定規則

建議在看結果前固定規則：

- 若 relocation policy 的 utility 提升很小，且 SLA violation 沒有實質下降，判為「目前設定未證明需要動態調度」。
- 若 utility 接近但 relocation 明顯降低 SLA violation，判為「scalarized reward 掩蓋了調度價值」，應改成 constrained/SLA evaluation。
- 若 relocation 在多數 weights 與 capacity-bound loads 都穩定勝過 wait，判為「動態調度問題成立」，才繼續比較 forecast、RL 與 MPC 架構。
- 不可用「真實 trace」本身取代上述必要性證據。

實質差異門檻應由營運 SLA 或單位成本決定；沒有營運門檻時，至少同時報 effect size、confidence interval 與原始 objective，不可只報 p-value。

## 本機 preflight（2026-07-15）

目前本機尚不能執行數值 audit：

- 缺 `data/models/tim_dataset/12-nodes/full_data.csv`；
- 缺 `data/models/tim_dataset/indexes/12-nodes-data-index.json`；
- 缺 `data/models/tim_dataset/aggregated_bs_data-LTE.csv`；
- 只有約 7.8 GiB 可用空間，低於 preflight 建議的 12 GiB safety margin；
- repository 內沒有原 TIM PRORL checkpoints。

公開原始 trace 約需傳輸 20.8 GB。資料可用逐日下載、處理、刪除中間檔的 streaming pipeline 生成，避免同時保存全部 raw files；但開始前仍應清出磁碟空間並確認網路流量。資料管線應以作者的 `tim-dataset-pipeline` 為基礎，保留來源與版本記錄。

因此目前合理狀態是：audit 程式與 configs 已就緒，資料／checkpoint 尚未就緒；不能把「尚未跑」寫成 `wait` 在真實 TIM 上獲勝。
