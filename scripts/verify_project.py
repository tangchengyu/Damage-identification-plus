"""Smoke checks for recovered files and historical inference, without training."""
from __future__ import annotations

import ast
import contextlib
import json
import math
import os
import runpy
import subprocess
import sys
import time
from pathlib import Path

from scripts.audit_dataset import ROOT, CLASSES, audit, sha256


def verify() -> dict:
    started = time.monotonic()
    source_paths = [ROOT / name / "DATA.json" for name in ["data", "extra_data"]]
    before = {str(p): sha256(p) for p in source_paths}
    images, summary = audit()
    assert len(images) == 98
    assert not summary["corrupt_images"]
    assert not summary["cross_dataset_exact_duplicates"]
    assert len(summary["datasets"]["data"]["path_corrections"]) == 4
    assert len(summary["datasets"]["data"]["unresolved_metadata"]) == 2
    assert len(summary["datasets"]["data"]["duplicate_references"]) == 1
    assert len(summary["datasets"]["data"]["unreferenced_images"]) == 3
    assert all(x["specimen_id"] is None and x["capture_group"] is None for x in images)
    assert all(not x["lifetime_ready"] for x in images)
    assert all(not x["classification_ready"] for x in images if x["dataset"] == "extra_data")
    assert before == {str(p): sha256(p) for p in source_paths}
    saved = json.loads((ROOT / "docs/data_audit/manifest.json").read_text(encoding="utf-8"))
    assert saved == images, "Saved inventory is stale; run project.py audit"
    provenance_path = ROOT / "docs/project_provenance.local.json"
    provenance = (json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.exists() else {"copied": []})
    for record in provenance["copied"]:
        assert sha256(ROOT / record["path"]) == record["sha256"], record["path"]
    syntax_files = [ROOT / "project.py"]
    for folder in [ROOT / "scripts", ROOT / "sunshangshibie_tiaozhan/src", ROOT / "sunshangshibie_tiaozhan/scripts", ROOT / "shoumingyuce/src"]:
        syntax_files.extend(folder.glob("*.py"))
    syntax_files.append(ROOT / "sunshangshibie_tiaozhan/utils.py")
    for path in syntax_files:
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        torch.set_num_threads(4)
        sys.path.insert(0, str(ROOT / "sunshangshibie_tiaozhan/src"))
        from predict import DamagePredictor
        predictions = []
        for variant in ["original", "updated"]:
            predictor = DamagePredictor(str(ROOT / "sunshangshibie_tiaozhan/output/models"), torch.device("cpu"), variant)
            assert predictor.model_hongguan is not None and predictor.model_tishijing is not None
            for view in ["hongguan", "tishijing"]:
                for label in CLASSES:
                    sample = next(x for x in images if x["dataset"] == "data" and x["label"] == label
                                  and x["photo_type"] == view and x["classification_ready"])
                    kwargs = {"image_hongguan_path" if view == "hongguan" else "image_tishijing_path": str(ROOT / sample["path"])}
                    output = predictor.predict_damage_type(**kwargs)
                    probabilities = [float(x) for x in output["probabilities"].values()]
                    assert all(math.isfinite(x) and 0 <= x <= 1 for x in probabilities)
                    assert abs(sum(probabilities) - 1) < 1e-5
                    assert output["used_models"] == [view]
                    predictions.append({"weights": variant, "view": view, "image": sample["path"],
                                        "prediction": output["predicted_class"], "probability": float(output["confidence"])})
            try:
                predictor.predict_damage_type()
                raise AssertionError("Missing images were accepted")
            except ValueError:
                pass
        del predictor
        lifetime_image = ROOT / "data/boluo/hongguan/train/4052-1.jpg"
        command = [sys.executable, str(ROOT / "project.py"), "lifetime", "--image", str(lifetime_image), "--sample-id", "boluo_figure1"]
        process = subprocess.run(command, text=True, encoding="utf-8", capture_output=True, check=True)
        lifetime = json.loads(process.stdout)
        assert math.isfinite(lifetime["predicted_lifetime"]) and lifetime["predicted_lifetime"] >= 0
        assert not lifetime["label_provenance_verified"]
        # Build the real desktop controls with a hidden root, without entering mainloop.
        import tkinter as tk
        constructor = tk.Tk
        def hidden_root(*args, **kwargs):
            window = constructor(*args, **kwargs)
            window.withdraw()
            return window
        tk.Tk = hidden_root
        namespace = None
        previous_dir = Path.cwd()
        try:
            os.chdir(ROOT / "sunshangshibie_tiaozhan")
            namespace = runpy.run_path(str(ROOT / "sunshangshibie_tiaozhan/utils.py"), run_name="project_check")
            assert namespace["PREDICTION_AVAILABLE"]
            assert len(namespace["btn"]) == 10
            assert namespace["initialize_predictor"]()
            # Test the actual GUI lifetime bridge, then ensure classification still works.
            raw = json.loads(source_paths[0].read_text(encoding="utf-8"))
            gui_value = namespace["predict_gear_lifetime"](str(lifetime_image), raw["boluo"]["boluo_figure1"]["shuxing"])
            assert gui_value is not None and abs(gui_value - lifetime["predicted_lifetime"]) < 1
            gui_predictor = namespace["initialize_predictor"].__globals__["predictor"]
            assert gui_predictor.model_variant == 'updated'
            assert all(Path(path).name.endswith('_updated.pth') for path in gui_predictor.checkpoints.values())
            assert gui_predictor.predict_damage_type(image_hongguan_path=str(lifetime_image))["used_models"] == ["hongguan"]
            namespace["root"].update_idletasks()
        finally:
            if namespace:
                namespace["root"].destroy()
            tk.Tk = constructor
            os.chdir(previous_dir)
    result = {"passed": True, "scope": "project_smoke_checks_not_independent_model_evaluation",
              "images_verified": len(images), "classification_ready_images": sum(x["classification_ready"] for x in images),
              "source_metadata_unchanged": True, "copied_files_verified": len(provenance["copied"]),
              "python_files_syntax_checked": len(syntax_files), "classification_checks": predictions,
              "lifetime_smoke_check": lifetime, "gui_controls": 10,
              "gui_lifetime_bridge_and_subsequent_classification": "passed",
              "gui_default_model_variant": "updated",
              "runtime": {"python": sys.version.split()[0], "torch": torch.__version__, "cuda_available": torch.cuda.is_available()},
              "elapsed_seconds": round(time.monotonic() - started, 2)}
    destination = ROOT / "output/verification"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "verification.local.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(verify(), ensure_ascii=False))
