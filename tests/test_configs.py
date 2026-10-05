"""
Guards on the experiment configs.

The baseline comparison (SRS FR-5, success criterion §174) is only meaningful
if the control differs from the treatment in exactly one place. Asserting that
here rather than trusting a hand-edited copy — a stray hyperparameter drift
between the two configs would silently invalidate the headline comparison.
"""

from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).parent.parent / "configs"
HETERO = CONFIG_DIR / "heterophily_gnn_hismall.yaml"
BASELINE = CONFIG_DIR / "baseline_sage_hismall.yaml"
SUB2M = CONFIG_DIR / "heterophily_gnn_hismall_sub2m.yaml"


def _flatten(d, prefix=""):
    """Flatten nested dicts to {dotted.key: value}."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, prefix=f"{key}."))
        else:
            out[key] = v
    return out


def test_baseline_differs_only_in_model_type():
    hetero = _flatten(yaml.safe_load(HETERO.read_text(encoding="utf-8")))
    baseline = _flatten(yaml.safe_load(BASELINE.read_text(encoding="utf-8")))

    # run_id_prefix must differ, or the two sweeps overwrite each other.
    assert hetero["run_id_prefix"] != baseline["run_id_prefix"]

    ignored = {"run_id_prefix"}
    differing = {
        k
        for k in set(hetero) | set(baseline)
        if k not in ignored and hetero.get(k) != baseline.get(k)
    }
    assert differing == {"model.type"}, (
        f"baseline config must differ from the heterophily config only in "
        f"model.type, but these differ: {sorted(differing)}"
    )
    assert hetero["model.type"] == "heterophily_gnn"
    assert baseline["model.type"] == "baseline_sage"


def test_sub2m_differs_from_hetero_only_in_subsampling():
    """The sub2m config claims to be a copy of the heterophily config plus a
    subsample block. It drifted once (epochs and per_typology_cap were bumped
    in the original but not the copy), so assert the claim."""
    hetero = _flatten(yaml.safe_load(HETERO.read_text(encoding="utf-8")))
    sub2m = _flatten(yaml.safe_load(SUB2M.read_text(encoding="utf-8")))

    assert hetero["run_id_prefix"] != sub2m["run_id_prefix"]
    differing = {
        k
        for k in set(hetero) | set(sub2m)
        if k != "run_id_prefix" and not k.startswith("subsample.")
        and hetero.get(k) != sub2m.get(k)
    }
    assert not differing, (
        f"sub2m config must match the heterophily config apart from "
        f"run_id_prefix and subsample, but these differ: {sorted(differing)}"
    )
    assert any(k.startswith("subsample.") for k in sub2m)


def test_both_configs_run_on_the_full_graph():
    """Neither may carry a subsample block, or they read different graphs."""
    for path in (HETERO, BASELINE):
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert not cfg.get("subsample"), f"{path.name} must run on the full graph"


def test_every_config_has_the_required_sections():
    """CONVENTIONS.md §4: every config must include dataset, model, seed,
    train and loss."""
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        for section in ("dataset", "model", "seed", "train", "loss"):
            assert section in cfg, f"{path.name} is missing '{section}'"
