"""Portable entry point for the historical gear inspection project."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def json_value(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(type(value).__name__)


def main():
    parser = argparse.ArgumentParser(description="Historical gear inspection project")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("audit", help="Regenerate the image inventory without editing source metadata")
    gui = commands.add_parser("gui", help="Open the desktop application")
    gui.add_argument("--weights", choices=["original", "updated"], default="updated")
    commands.add_parser("verify", help="Check files, weights, CPU inference and hidden GUI initialization")
    predict = commands.add_parser("predict", help="Run the original or fine-tuned classifiers")
    predict.add_argument("--hongguan", type=Path)
    predict.add_argument("--tishijing", type=Path)
    predict.add_argument("--weights", choices=["original", "updated"], default="updated")
    predict.add_argument("--macro-weight", type=float, default=0.3,
                         help="Macro fraction in two-view fusion (0 to 1)")
    lifetime = commands.add_parser("lifetime", help="Run the historical lifetime model; labels remain unverified")
    lifetime.add_argument("--image", type=Path, required=True)
    attributes = lifetime.add_mutually_exclusive_group(required=True)
    attributes.add_argument("--sample-id")
    attributes.add_argument("--attributes-stdin", action="store_true")
    args = parser.parse_args()
    if args.command == "audit":
        from scripts.audit_dataset import write_audit
        summary = write_audit()
        print(json.dumps({"images": summary["total_images"], "datasets": {
            name: data["status_counts"] for name, data in summary["datasets"].items()}}, ensure_ascii=False))
    elif args.command == "verify":
        from scripts.verify_project import verify
        print(json.dumps(verify(), ensure_ascii=False, default=json_value))
    elif args.command == "gui":
        module = ROOT / "sunshangshibie_tiaozhan"
        os.chdir(module)
        sys.path.insert(0, str(module / "src"))
        sys.argv = [str(module / "utils.py"), "--weights", args.weights]
        runpy.run_path(str(module / "utils.py"), run_name="__main__")
    elif args.command == "predict":
        if not 0 <= args.macro_weight <= 1:
            parser.error("--macro-weight must be finite and between 0 and 1")
        if not args.hongguan and not args.tishijing:
            parser.error("At least one image is required")
        with contextlib.redirect_stdout(sys.stderr):
            sys.path.insert(0, str(ROOT / "sunshangshibie_tiaozhan" / "src"))
            import torch
            torch.set_num_threads(4)
            from predict import DamagePredictor
            model = DamagePredictor(str(ROOT / "sunshangshibie_tiaozhan/output/models"),
                                    device=torch.device("cpu"), model_variant=args.weights)
            result = model.predict_damage_type(
                str(args.hongguan.resolve()) if args.hongguan else None,
                str(args.tishijing.resolve()) if args.tishijing else None,
                weight_hongguan=args.macro_weight, weight_tishijing=1-args.macro_weight)
        print(json.dumps(result, ensure_ascii=False, default=json_value))
    else:
        if args.attributes_stdin:
            gear_attributes = json.load(sys.stdin)
        else:
            metadata = json.loads((ROOT / "data/DATA.json").read_text(encoding="utf-8"))
            matches = [record for samples in metadata.values() for key, record in samples.items() if key == args.sample_id]
            if len(matches) != 1:
                parser.error("Sample id must identify exactly one metadata record")
            gear_attributes = matches[0]["shuxing"]
        with contextlib.redirect_stdout(sys.stderr):
            sys.path.insert(0, str(ROOT / "shoumingyuce/src"))
            import torch
            torch.set_num_threads(4)
            from predict import LifetimePredictor
            model = LifetimePredictor(str(ROOT / "shoumingyuce/output/models/best_lifetime_model.pth"),
                                      device=torch.device("cpu"))
            value = model.predict_single(str(args.image.resolve()), gear_attributes)
        print(json.dumps({"predicted_lifetime": float(value), "unit": "cycles",
                          "historical_model": True, "label_provenance_verified": False}, ensure_ascii=False))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    main()
