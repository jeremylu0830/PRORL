import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml

from experiments.proactivity_ablation.generate_tim_realdata_audit import build_audit_config
from experiments.proactivity_ablation.tim_realdata_preflight import inspect
from prorl.environment.agent import AgentType
from prorl.environment.agent.baseline.wait import WaitAgent


class TimRealDataAuditTests(unittest.TestCase):
    def test_legacy_paper_baseline_names_are_supported(self):
        self.assertEqual(AgentType("greedy-optimal"), AgentType.Heuristic)
        self.assertEqual(AgentType("exhaustive-search"), AgentType.Greedy)

    def test_sampling_optimal_uses_standard_baseline_rollout(self):
        self.assertTrue(AgentType.is_baseline(AgentType.SamplingOptimal))
        self.assertFalse(AgentType.is_mc_method(AgentType.SamplingOptimal))

    def test_audit_config_preserves_source_and_adds_wait(self):
        source = {
            "multi_run_name": "paper",
            "base_run_config": {"emulator": {"model": {"tim_dataset_model_options": {
                "loads_with_respect_to_capacity": {"evaluation": [0.8]}
            }}}},
            "hyperparameters": {"random": [{"key": "param.key", "values": [1]}]},
        }
        audit = build_audit_config(source, 1.0, ["wait", "random"])
        self.assertNotIn("wait", source["hyperparameters"])
        self.assertIn("wait", audit["hyperparameters"])
        self.assertEqual(
            audit["base_run_config"]["emulator"]["model"]["tim_dataset_model_options"]
            ["loads_with_respect_to_capacity"]["evaluation"],
            [1.0],
        )

    def test_wait_agent_uses_wait_indices(self):
        agent = object.__new__(WaitAgent)
        agent.action_space_wrapper = SimpleNamespace(
            add_node_space=SimpleNamespace(wait_action_index=12),
            remove_node_space=SimpleNamespace(wait_action_index=13),
            quantity_space=SimpleNamespace(wait_action_index=2),
            add_action_space=SimpleNamespace(wait_action_index=7),
            remove_action_space=SimpleNamespace(wait_action_index=8),
        )
        self.assertEqual(agent._choose_node_add(None), 12)
        self.assertEqual(agent._choose_node_remove(None), 13)
        self.assertEqual(agent._choose_quantity(None), 2)
        self.assertEqual(agent._choose_add(None), 7)
        self.assertEqual(agent._choose_remove(None), 8)

    def test_preflight_detects_missing_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            config = repo / "audit.yaml"
            config.write_text(yaml.safe_dump({"base_run_config": {"emulator": {"model": {
                "tim_dataset_model_options": {
                    "full_data_path": "models/tim/full.csv",
                    "index_data_path": "models/tim/index.json",
                    "bs_ids": "models/ids.json",
                    "ran_configurations_path": "models/ran.json",
                    "bs_data_path": "models/bs.csv",
                }
            }}}}), encoding="utf-8")
            report = inspect(repo, config)
            self.assertFalse(report["ready_for_baselines"])
            self.assertIn("full_data", report["blocking_checks"])


if __name__ == "__main__":
    unittest.main()
