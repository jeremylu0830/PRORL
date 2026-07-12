# PRORL proactivity ablation

This experiment tests whether PRORL's time encoding contributes to anticipatory
resource movements. It does not modify the original configuration or any file under
`prorl/`.

## 1. Generate isolated configs

```bash
source .venv/bin/activate
python experiments/proactivity_ablation/generate_configs.py
```

This creates `generated/original.yaml` and `generated/no_time.yaml`. Both use training
seeds 10--19 and otherwise inherit the paper's longer single-peak configuration.

## 2. Schedule and run

Start Redis and MongoDB as described in the project README, then:

```bash
python prorl.py run scheduler multi-runs \
  --from-folder experiments/proactivity_ablation/generated \
  -q proactivity-ablation
python prorl.py run worker run-worker -p 0 --stop-empty -q proactivity-ablation
```

Multiple workers can consume the same queue.

## 3. Analyze

Pass the folder containing the resulting runs:

```bash
python experiments/proactivity_ablation/analyze.py \
  data/models/results/proactivity-ablation-original-* \
  data/models/results/proactivity-ablation-no_time-* \
  --output experiments/proactivity_ablation/proactivity_summary.csv
```

The analyzer reports the original objective totals plus movements made in the six-hour
window preceding each configured peak. A positive lead only establishes calendar-based
anticipation; a shifted-peak evaluation is needed to test temporal generalization.

## Interpretation

- Original better than No-Time on fixed peaks: calendar information is useful.
- Original moves early but fails shifted peaks: it memorizes the schedule.
- No-Time matches Original: the claimed proactivity is not explained by time encoding.

Shifted-peak evaluation must load the same trained checkpoints under a changed evaluation
workload. It should not be approximated by retraining on shifted peaks.
