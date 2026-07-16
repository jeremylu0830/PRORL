# PRORL proactivity ablation

This experiment tests whether PRORL's time encoding contributes to anticipatory
resource movements. It does not modify the original configuration or any file under
`prorl/`.

## 1. Generate isolated configs

```bash
source .venv/bin/activate
python experiments/proactivity_ablation/generate_configs.py
```

This creates `generated/original.yaml`, `generated/no_time.yaml`, and
`generated/forecast_no_time.yaml`. All use training seeds 10--19 and otherwise inherit
the paper's longer single-peak configuration. Forecast-No-Time removes `time-encoded`
and adds a six-hour, node-major Schedule Oracle forecast to the state. The oracle exposes
the configured future demand mean and intentionally excludes random observation noise.

Run the compatibility and forecast integration tests with:

```bash
ENV=test python -m unittest experiments.proactivity_ablation.test_forecast_state -v
```

The bounded training smoke test can be detached and monitored independently:

```bash
tmux new-session -d -s forecast-smoke \
  '.venv/bin/python experiments/proactivity_ablation/run_forecast_smoke.py'
tmux new-session -d -s forecast-smoke-monitor \
  '.venv/bin/python experiments/proactivity_ablation/monitor_forecast_smoke.py'
```

## 2. Schedule and run

Start Redis and MongoDB as described in the project README, then:

```bash
python prorl.py run scheduler multi-runs \
  --from-folder experiments/proactivity_ablation/generated \
  -q proactivity-ablation
python prorl.py run worker run-worker -p 0 --stop-empty -q proactivity-ablation
```

Multiple workers can consume the same queue.

The folder command schedules all three variants. To schedule only Forecast-No-Time,
place or copy that config into an otherwise empty scheduling folder.

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

## 4. Randomized-schedule training

The second-stage experiment trains a paired No-Time and Forecast-No-Time condition on
the same episode-level randomized workload. The tracked scenario manifest contains 12
training scenarios and four disjoint held-out combinations. The randomized model option
is disabled by default, so legacy configurations retain their original behavior.

Generate the two isolated configs and run the bounded paired smoke test:

```bash
python experiments/proactivity_ablation/generate_randomized_configs.py
ENV=test python -m unittest experiments.proactivity_ablation.test_randomized_schedule -v
ENV=test python experiments/proactivity_ablation/run_randomized_smoke.py --steps 192
```

On the remote training host, Redis and MongoDB must already be running. The launcher
refuses non-empty experiment queues, schedules exactly 20 runs (two conditions by ten
seeds), starts a validation worker, and audits the failed queue when training finishes:

```bash
python experiments/proactivity_ablation/launch_randomized_training.py \
  --processes 2 \
  --validation-processes 1 \
  --queue proactivity-randomized-v1
```

After training, run evaluation-only held-out jobs and analysis:

```bash
ENV=test python experiments/proactivity_ablation/run_randomized_heldout_evaluations.py \
  <randomized training result roots> \
  --output-dir experiments/proactivity_ablation/randomized_heldout_evaluations \
  --checkpoint best

python experiments/proactivity_ablation/analyze_randomized_heldout.py \
  experiments/proactivity_ablation/randomized_heldout_evaluations \
  --output-dir experiments/proactivity_ablation/randomized_heldout_analysis
```

Diagnose action/reward conversion without retraining by applying inference-time
movement gates to the same frozen randomized Forecast-No-Time checkpoints:

```bash
ENV=test python experiments/proactivity_ablation/run_movement_gate_ablation.py \
  <randomized Forecast-No-Time training result root> \
  --output-dir experiments/proactivity_ablation/movement_gate_evaluations

python experiments/proactivity_ablation/analyze_movement_gate_ablation.py \
  experiments/proactivity_ablation/movement_gate_evaluations \
  --baseline-reference \
    experiments/proactivity_ablation/randomized_heldout_evaluations/Randomized-Forecast-No-Time \
  --output-dir experiments/proactivity_ablation/movement_gate_analysis
```

The four modes are `baseline`, `satisfied` (cancel an add when its target
already covers the maximum current/six-hour forecast demand), `safe-remove`
(cancel a remove that would make its source fall below that requirement), and
`combined`. The normal rollout history stores executed, not proposed, actions;
separate metadata records proposed, canceled, and executed sub-action counts.

Run the static/current/forecast-horizon controls (the existing `baseline` and
`combined` directories provide policy and h6 results):

```bash
ENV=test python experiments/proactivity_ablation/run_movement_gate_ablation.py \
  <randomized Forecast-No-Time training result root> \
  --output-dir experiments/proactivity_ablation/movement_gate_evaluations \
  --modes always-wait combined-current combined-h1 combined-h3

python experiments/proactivity_ablation/analyze_movement_gate_controls.py \
  experiments/proactivity_ablation/movement_gate_evaluations \
  --baseline-reference \
    experiments/proactivity_ablation/randomized_heldout_evaluations/Randomized-Forecast-No-Time \
  --output-dir experiments/proactivity_ablation/movement_gate_control_analysis
```

## TIM real-data benchmark necessity audit

The synthetic `always-wait` result does not establish the same conclusion on
the paper's TIM trace. The isolated real-data audit, its preflight, and the
pre-registered interpretation rules are documented in
`TIM_REALDATA_NECESSITY_AUDIT_ZH.md`.

```bash
.venv/bin/python experiments/proactivity_ablation/generate_tim_realdata_audit.py --loads 0.8
.venv/bin/python experiments/proactivity_ablation/tim_realdata_preflight.py
```

The completed 200-job audit analysis and its statistical limitations are in
`TIM_REALDATA_AUDIT_RESULTS_ZH.md`. Regenerate the paired summaries with:

```bash
.venv/bin/python experiments/proactivity_ablation/analyze_tim_realdata_audit.py \
  <extracted-audit>/all_runs.csv \
  --output-dir experiments/proactivity_ablation/tim_realdata_audit_analysis
```

The subsequent train-only utility-best-static control is documented in
`TIM_BEST_STATIC_RESULTS_ZH.md`.
