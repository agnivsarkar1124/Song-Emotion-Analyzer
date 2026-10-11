import json

import benchmark


def test_benchmark_runs_end_to_end_on_synthetic_songs(tmp_path):
    out = tmp_path / "results.json"
    report = benchmark.main(["--synthetic", "--seeds", "1", "--budgets", "4", "8", "--neural", "off", "--out", str(out)])
    assert json.loads(out.read_text())["headline"] == report["headline"]
    systems = report["results"][8]
    assert {"mood only (no personalization)", "linear taste model", "linear, no interaction terms"} <= set(systems)
    for metrics in systems.values():
        assert 0 <= metrics["ndcg"] <= 1 and 0 <= metrics["precision"] <= 1
        assert set(metrics["per_listener_ndcg"]) == set(benchmark.LISTENERS)
    assert "random taste (control)" not in report["headline"]["systems"]


def test_ndcg_and_precision():
    rel = {"a": 1.0, "b": 0.5, "c": 0.0}
    assert benchmark.ndcg_at_k(["a", "b", "c"], rel, 2) == 1.0
    assert benchmark.ndcg_at_k(["c", "b", "a"], rel, 2) < 0.5
    assert benchmark.precision_at_k(["a", "c"], {"a"}, 2) == 0.5
