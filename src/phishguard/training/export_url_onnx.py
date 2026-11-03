"""Export the trained URL LightGBM booster to ONNX and verify it against native.

Steps:
    1. Load booster, calibrator and feature names from the config's artifacts.
    2. Convert with onnxmltools (TreeEnsembleClassifier, zipmap removed).
    3. Patch split thresholds back to float64. onnxmltools writes float32
       thresholds and a float32 input; LightGBM compares float64 against
       float64. On this model that flipped a branch for ~0.8% of URLs
       (max |diff| 9e-3). ai.onnx.ml opset 3 allows ``nodes_values_as_tensor``
       in float64, so the patched graph takes a float64 input and compares
       against the exact thresholds LightGBM learned.
    4. Save ``<model_stem>.onnx`` and ``url_tld_categories_<tag>.json`` next
       to the booster. The TLD list is what the ONNX serving path needs to map
       TLD strings to LightGBM's category codes.
    5. Verify: max |onnx - native| on a test sample and on the Tranco probe,
       then the usual metrics computed through the ONNX path + calibrator.
       Optionally write a Markdown report.

Usage:
    python -m phishguard.training.export_url_onnx --config configs/url_model_v0_2.yaml \
        --report reports/url_onnx_2026-11-03.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import onnx
import pandas as pd
import yaml
from onnx import TensorProto, helper, numpy_helper
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)

from phishguard.features import URLFeatureExtractor
from phishguard.serving.url_onnx import (
    INPUT_NAME,
    UrlOnnxModel,
    booster_tld_categories,
    encode_features,
    tld_code_map,
)

TARGET_OPSET = 15
ML_OPSET_WITH_TENSOR_ATTRS = 3


def _dump_split_thresholds(booster: lgb.Booster) -> dict[tuple[int, int, np.float32], list[float]]:
    """(tree, feature, float32(threshold)) -> list of float64 thresholds in the dump."""
    out: dict[tuple[int, int, np.float32], list[float]] = {}

    def walk(tree_id: int, node: dict[str, Any]) -> None:
        if "split_feature" in node:
            if node["decision_type"] == "<=":
                key = (tree_id, int(node["split_feature"]), np.float32(node["threshold"]))
                out.setdefault(key, []).append(float(node["threshold"]))
            walk(tree_id, node["left_child"])
            walk(tree_id, node["right_child"])

    for tree in booster.dump_model()["tree_info"]:
        walk(int(tree["tree_index"]), tree["tree_structure"])
    return out


def patch_double_thresholds(model: onnx.ModelProto, booster: lgb.Booster) -> dict[str, int]:
    """Rewrite the TreeEnsembleClassifier to compare float64 input against float64 thresholds.

    Returns counts: how many ``<=`` nodes were restored to float64 and how many
    were left as float32 because two thresholds in the same tree on the same
    feature rounded to the same float32 (ambiguous; never seen on this model).
    """
    tree_node = next(n for n in model.graph.node if n.op_type == "TreeEnsembleClassifier")
    attrs = {a.name: a for a in tree_node.attribute}
    exact = _dump_split_thresholds(booster)

    tree_ids = list(attrs["nodes_treeids"].ints)
    feat_ids = list(attrs["nodes_featureids"].ints)
    modes = [m.decode() for m in attrs["nodes_modes"].strings]
    values32 = list(attrs["nodes_values"].floats)

    values64: list[float] = []
    restored = ambiguous = 0
    for tid, fid, mode, v32 in zip(tree_ids, feat_ids, modes, values32, strict=True):
        if mode != "BRANCH_LEQ":
            values64.append(float(v32))
            continue
        candidates = set(exact.get((tid, fid, np.float32(v32)), []))
        if len(candidates) == 1:
            values64.append(candidates.pop())
            restored += 1
        else:
            values64.append(float(v32))
            ambiguous += 1

    def replace_with_tensor(name: str, data: np.ndarray) -> None:
        old = attrs.pop(name)
        tree_node.attribute.remove(old)
        tree_node.attribute.append(
            helper.make_attribute(f"{name}_as_tensor", numpy_helper.from_array(data))
        )

    replace_with_tensor("nodes_values", np.asarray(values64, dtype=np.float64))
    replace_with_tensor(
        "nodes_hitrates", np.asarray(list(attrs["nodes_hitrates"].floats), dtype=np.float64)
    )
    replace_with_tensor(
        "class_weights", np.asarray(list(attrs["class_weights"].floats), dtype=np.float64)
    )
    if "base_values" in attrs:
        replace_with_tensor(
            "base_values", np.asarray(list(attrs["base_values"].floats), dtype=np.float64)
        )

    graph_input = next(i for i in model.graph.input if i.name == tree_node.input[0])
    graph_input.type.tensor_type.elem_type = TensorProto.DOUBLE
    for opset in model.opset_import:
        if opset.domain == "ai.onnx.ml":
            opset.version = ML_OPSET_WITH_TENSOR_ATTRS
    onnx.checker.check_model(model)
    return {"restored": restored, "ambiguous": ambiguous}


def export_booster(
    booster: lgb.Booster,
    feature_names: list[str],
    onnx_path: Path,
    tld_path: Path,
    double_thresholds: bool = True,
) -> dict[str, int]:
    import onnxmltools
    from onnxmltools.convert.common.data_types import FloatTensorType

    initial_types = [(INPUT_NAME, FloatTensorType([None, len(feature_names)]))]
    onx = onnxmltools.convert_lightgbm(
        booster, initial_types=initial_types, zipmap=False, target_opset=TARGET_OPSET
    )
    model = onnx.load_from_string(onx.SerializeToString())
    stats = patch_double_thresholds(model, booster) if double_thresholds else {}
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(onnx_path))
    tld_path.write_text(json.dumps(booster_tld_categories(booster)))
    return stats


def _featurize(urls: list[str]) -> list[dict[str, Any]]:
    return URLFeatureExtractor().transform(urls)


def native_predict(
    booster: lgb.Booster, rows: list[dict[str, Any]], feature_names: list[str]
) -> np.ndarray:
    df = pd.DataFrame(rows)[feature_names]
    if "tld" in df.columns:
        df["tld"] = df["tld"].astype("category")
    return np.asarray(booster.predict(df), dtype=np.float64)


def compare(
    booster: lgb.Booster, model: UrlOnnxModel, urls: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    rows = _featurize(urls)
    return native_predict(booster, rows, model.feature_names), model.predict_raw(rows)


def _diff_summary(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    d = np.abs(a - b)
    return {
        "n": int(len(d)),
        "max_abs_diff": float(d.max()),
        "mean_abs_diff": float(d.mean()),
        "n_over_1e-5": int((d > 1e-5).sum()),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--sample", type=int, default=5000, help="test rows for the native check")
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--no-double-thresholds", action="store_true")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    art = cfg["artifacts"]
    model_path = Path(art["model_path"])
    booster = lgb.Booster(model_file=str(model_path))
    # Trusted local artifact written by train_url.py (same pattern as evaluate.py).
    calibrator = joblib.load(art["calibrator_path"])
    feature_names: list[str] = json.loads(Path(art["feature_names_path"]).read_text())
    onnx_path = model_path.with_suffix(".onnx")
    tag = model_path.stem.replace("url_model", "")
    tld_path = model_path.parent / f"url_tld_categories{tag}.json"

    t0 = time.perf_counter()
    stats = export_booster(
        booster,
        feature_names,
        onnx_path,
        tld_path,
        double_thresholds=not args.no_double_thresholds,
    )
    export_s = time.perf_counter() - t0
    print(
        f"exported {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB) in {export_s:.1f}s {stats}"
    )

    model = UrlOnnxModel.from_paths(onnx_path, Path(art["feature_names_path"]), tld_path)

    test_df = pd.read_parquet(cfg["data"]["test_path"])
    label = cfg["data"]["label_col"]
    sample = test_df.sample(n=min(args.sample, len(test_df)), random_state=0)
    nat_s, onx_s = compare(booster, model, sample["url"].astype(str).tolist())
    sample_diff = _diff_summary(nat_s, onx_s)
    print("test sample  native vs onnx:", sample_diff)

    rows_test = _featurize(test_df["url"].astype(str).tolist())
    y = test_df[label].astype(int).to_numpy()
    raw_onnx = model.predict_raw(rows_test)
    raw_native = native_predict(booster, rows_test, feature_names)
    full_diff = _diff_summary(raw_native, raw_onnx)
    cal = np.asarray(calibrator.transform(raw_onnx))
    metrics = {
        "auc": float(roc_auc_score(y, cal)),
        "ap": float(average_precision_score(y, cal)),
        "f1": float(f1_score(y, (cal >= 0.5).astype(int))),
        "brier": float(brier_score_loss(y, cal)),
        "n": int(len(y)),
    }
    print("full test split native vs onnx:", full_diff)
    print("test metrics through ONNX + calibrator:", metrics)

    probe: dict[str, float] = {}
    probe_diff: dict[str, float] = {}
    probe_path = Path("data/processed/url_probe_tranco.parquet")
    if probe_path.exists():
        probe_df = pd.read_parquet(probe_path)
        nat_p, onx_p = compare(booster, model, probe_df["url"].astype(str).tolist())
        probe_diff = _diff_summary(nat_p, onx_p)
        cal_p = np.asarray(calibrator.transform(onx_p))
        probe = {
            "n": int(len(cal_p)),
            "median_p_phish": float(np.median(cal_p)),
            "fpr_at_0.5": float((cal_p >= 0.5).mean()),
            "fpr_at_0.7": float((cal_p >= 0.7).mean()),
            "fpr_at_0.9": float((cal_p >= 0.9).mean()),
        }
        print("tranco probe native vs onnx:", probe_diff)
        print("tranco probe through ONNX + calibrator:", probe)

    one = encode_features(_featurize(["https://example.com"]), feature_names, tld_code_map([]))
    one = one.astype(model.input_dtype)
    t = time.perf_counter()
    for _ in range(200):
        model.session.run(None, {model.input_name: one})
    single_ms = (time.perf_counter() - t) / 200 * 1000

    if args.report:
        import onnxmltools
        import onnxruntime

        cmd = " ".join(["python -m phishguard.training.export_url_onnx", *(argv or sys.argv[1:])])
        lines = [
            f"# URL model ONNX export ({date.today().isoformat()})",
            "",
            f"Command: `{cmd}`",
            "",
            f"Source booster: `{model_path}` ({booster.num_trees()} trees, "
            f"{len(feature_names)} features). Output: `{onnx_path}` "
            f"({onnx_path.stat().st_size / 1e6:.1f} MB), TLD codes in `{tld_path}`.",
            f"lightgbm {lgb.__version__}, onnxmltools {onnxmltools.__version__}, "
            f"onnx {onnx.__version__}, onnxruntime {onnxruntime.__version__}, "
            f"float64 thresholds: {not args.no_double_thresholds} {stats}.",
            "",
            "## ONNX vs native booster (raw, uncalibrated p_phish)",
            "",
            "| set | n | max abs diff | mean abs diff | rows with diff > 1e-5 |",
            "|---|---|---|---|---|",
        ]
        for name, d in (
            ("test sample (random_state=0)", sample_diff),
            ("full test split", full_diff),
            ("Tranco top-5000 probe", probe_diff),
        ):
            if d:
                lines.append(
                    f"| {name} | {d['n']} | {d['max_abs_diff']:.3e} | "
                    f"{d['mean_abs_diff']:.3e} | {d['n_over_1e-5']} |"
                )
        lines += [
            "",
            "## Test split, served path (ONNX session + isotonic calibrator in Python)",
            "",
            f"n={metrics['n']}",
            "",
            "| metric | value |",
            "|---|---|",
        ]
        for k in ("auc", "ap", "f1", "brier"):
            lines.append(f"| {k} | {metrics[k]:.4f} |")
        if probe:
            lines += [
                "",
                "## Tranco top-5000 probe, served path (benign-only, FPR-style)",
                "",
                "| metric | value |",
                "|---|---|",
            ]
            for k, v in probe.items():
                lines.append(f"| {k} | {v} |" if k == "n" else f"| {k} | {v:.4f} |")
        lines += [
            "",
            f"Single-URL ONNX session latency: {single_ms:.3f} ms "
            "(mean of 200 runs, CPUExecutionProvider, excludes feature extraction).",
        ]
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text("\n".join(lines) + "\n")
        print(f"wrote report -> {args.report}")


if __name__ == "__main__":
    main()
