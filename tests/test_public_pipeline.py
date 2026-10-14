"""CPU checks for sample conversion and metric reporting."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_examples import convert


class ExamplesTest(unittest.TestCase):
    def setUp(self):
        self.row = {
            "qid": "q1",
            "input": {"user_text": "Nighttime crashes on an unlit road.",
                      "candidates": [{"evidence_id": "1", "countermeasure": "Install lighting"}]},
            "target": {"recommendations": [{"rank": 1, "evidence_ids": ["1"]}]},
        }

    def test_labels_do_not_change_inference_input(self):
        original = convert([self.row])
        changed = copy.deepcopy(self.row)
        changed["target"]["recommendations"] = []
        empty = convert([changed])
        self.assertEqual(original["inputs"], empty["inputs"])
        self.assertEqual(original["queries"], empty["queries"])
        self.assertNotEqual(original["sft"][0]["completion"], empty["sft"][0]["completion"])

    def test_preserves_payload_and_empty_reference(self):
        self.row["target"]["recommendations"] = []
        result = convert([self.row])
        payload = json.loads(result["inputs"][0]["prompt"].split("INPUT_JSON:\n", 1)[1])
        self.assertEqual(payload["candidates"], self.row["input"]["candidates"])
        self.assertEqual(payload["user_text"], self.row["input"]["user_text"])
        self.assertEqual(json.loads(result["sft"][0]["completion"]), {"recommendations": []})

    def test_rejects_duplicate_queries(self):
        with self.assertRaises(ValueError):
            convert([self.row, self.row])

    def test_rejects_reference_outside_pool(self):
        self.row["target"]["recommendations"][0]["evidence_ids"] = ["missing"]
        with self.assertRaises(ValueError):
            convert([self.row])

    def test_rejects_duplicate_candidates(self):
        self.row["input"]["candidates"] *= 2
        with self.assertRaises(ValueError):
            convert([self.row])


class MetricsTest(unittest.TestCase):
    def run_evaluator(self, labels, predictions):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            records = {"labels": labels, "predictions": predictions,
                       "evidence": [{"evidence_id": "1", "countermeasure": "Install roadway lighting"}]}
            for name, rows in records.items():
                (folder / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            command = [sys.executable, str(ROOT / "scripts/evaluate_v318_uniform_model_performance.py"),
                       "--labels", str(folder / "labels.jsonl"), "--predictions", str(folder / "predictions.jsonl"),
                       "--evidence-store", str(folder / "evidence.jsonl"), "--out", str(folder / "metrics.json")]
            result = subprocess.run(command, capture_output=True, text=True, timeout=20)
            metrics = json.loads((folder / "metrics.json").read_text()) if result.returncode == 0 else None
            return result, metrics

    @staticmethod
    def prediction(qid, ids):
        return {"qid": qid, "predicted_evidence_ids": ids, "json_error": "",
                "all_ids_in_frozen_candidates": True, "duplicate_ids": False}

    def test_nonempty_and_empty_denominators(self):
        labels = [{"qid": "q1", "topk": [{"rank": 1, "evidence_ids": ["1"]}]}, {"qid": "q2", "topk": []}]
        result, metrics = self.run_evaluator(labels, [self.prediction("q1", ["1"]), self.prediction("q2", [])])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(metrics["exact_evidence_top1"], 1.0)
        self.assertEqual(metrics["mechanism_hit_at_3"], 1.0)
        self.assertEqual(metrics["empty_set_decision_accuracy"], 1.0)
        self.assertEqual(metrics["nonempty_reference_queries"], 1)

    def test_all_empty_subset(self):
        result, metrics = self.run_evaluator([{"qid": "q1", "topk": []}], [self.prediction("q1", [])])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIsNone(metrics["exact_evidence_top1"])
        self.assertEqual(metrics["empty_set_decision_accuracy"], 1.0)

    def test_duplicate_predictions_fail(self):
        result, _ = self.run_evaluator([{"qid": "q1", "topk": []}], [self.prediction("q1", []), self.prediction("q1", [])])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("duplicate qid", result.stderr)


if __name__ == "__main__":
    unittest.main()
