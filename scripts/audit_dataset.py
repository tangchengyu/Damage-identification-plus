"""Inventory historical images without changing source metadata or guessing identities."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["boluo", "cashang", "dianshi", "yashang"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(root: Path = ROOT) -> tuple[list[dict], dict]:
    module = root
    images, datasets, all_issues = [], {}, []
    for name in ["data", "extra_data"]:
        base = module / name
        original = json.loads((base / "DATA.json").read_text(encoding="utf-8"))
        references = defaultdict(list)
        corrections, unresolved, vector_errors = [], [], []
        count = 0
        for label, records in original.items():
            for sample_id, record in records.items():
                count += 1
                prefix = name + "_"
                parts = record["path"].removeprefix(prefix).split("_", 3)
                if len(parts) != 4 or not record["path"].startswith(prefix):
                    unresolved.append({"id": sample_id, "path": record["path"], "reason": "invalid_path_format"})
                    continue
                category, view, split, filename = parts
                expected = [int(category == c) for c in CLASSES]
                if category != label or record.get("category-vector") != expected:
                    vector_errors.append(sample_id)
                source_relative = Path(category) / view / split / (filename + ".jpg")
                resolved = base / source_relative
                reason = None
                if not resolved.exists() and split == "tset":
                    candidate = base / category / view / "test" / (filename + ".jpg")
                    if candidate.exists():
                        resolved, reason = candidate, "tset_to_test"
                if not resolved.exists() and split == "train":
                    other = "test"
                    candidate = base / category / view / other / (filename + ".jpg")
                    if candidate.exists():
                        resolved, reason = candidate, "split_directory_mismatch"
                if not resolved.exists():
                    unresolved.append({"id": sample_id, "path": record["path"], "reason": "file_not_found_no_safe_mapping"})
                    continue
                reference = {"sample_id": sample_id, "original_path": record["path"],
                             "lifetime": record.get("shouming"), "correction": reason}
                references[resolved].append(reference)
                if reason:
                    corrections.append({"id": sample_id, "original_path": record["path"],
                                        "resolved_path": resolved.relative_to(module).as_posix(), "reason": reason})
        corrupt, duplicates = [], []
        hashes = defaultdict(list)
        dataset_images = []
        for path in sorted(base.rglob("*.jpg")):
            category, view, split, filename = path.relative_to(base).parts
            refs = references[path]
            status = "matched"
            if not refs:
                status = "unreferenced"
            elif len(refs) > 1:
                status = "metadata_conflict"
                duplicates.append({"path": path.relative_to(module).as_posix(), "records": refs})
            elif refs[0]["correction"]:
                status = "path_corrected"
            if name == "extra_data":
                status = "source_unverified"
            width = height = 0
            mode = ""
            try:
                with Image.open(path) as image:
                    image.load()
                    width, height = image.size
                    mode = image.mode
            except Exception as error:
                corrupt.append({"path": path.relative_to(module).as_posix(), "error": str(error)})
                status = "corrupt"
            digest = sha256(path)
            hashes[digest].append(path.relative_to(root).as_posix())
            entry = {"image_id": digest[:16], "path": path.relative_to(root).as_posix(),
                     "dataset": name, "label": category, "photo_type": view,
                     "original_directory_split": split, "filename_prefix": filename.split("-")[0] if name == "data" else None,
                     "specimen_id": None, "capture_group": None, "status": status,
                     "classification_ready": status in ["matched", "path_corrected"] and not vector_errors,
                     "label_verification": "metadata_and_directory_only_not_visual_reviewed",
                     "lifetime_ready": False, "metadata_records": refs,
                     "width": width, "height": height, "mode": mode, "bytes": path.stat().st_size,
                     "sha256": digest}
            images.append(entry)
            dataset_images.append(entry)
        exact_duplicates = [paths for paths in hashes.values() if len(paths) > 1]
        datasets[name] = {"metadata_rows": count, "image_files": len(dataset_images),
                          "class_counts": dict(Counter(x["label"] for x in dataset_images)),
                          "view_counts": dict(Counter(x["photo_type"] for x in dataset_images)),
                          "directory_split_counts": dict(Counter(x["original_directory_split"] for x in dataset_images)),
                          "status_counts": dict(Counter(x["status"] for x in dataset_images)),
                          "path_corrections": corrections, "unresolved_metadata": unresolved,
                          "duplicate_references": duplicates, "invalid_class_vectors": vector_errors,
                          "unreferenced_images": [x["path"] for x in dataset_images if not x["metadata_records"]],
                          "corrupt_images": corrupt, "exact_duplicate_images": exact_duplicates,
                          "source_metadata_sha256": sha256(base / "DATA.json")}
        all_issues.extend(corrupt)
    combined_hashes = defaultdict(list)
    for entry in images:
        combined_hashes[entry["sha256"]].append(entry["path"])
    summary = {"audit_version": 1, "scope": "historical_files_only_no_training",
               "source_metadata_modified": False, "datasets": datasets,
               "total_images": len(images), "corrupt_images": all_issues,
               "cross_dataset_exact_duplicates": [v for v in combined_hashes.values() if len(v) > 1],
               "limitations": ["Labels were checked against metadata and folders, not independently re-annotated.",
                               "No verified physical specimen or capture group identifiers are available.",
                               "Exact byte equality is tested; visual near duplicates are not certified absent.",
                               "Lifetime provenance is unresolved; no lifetime record is certified usable.",
                               "Extra image provenance is unresolved and these images are excluded from the ready subset."]}
    return images, summary


def write_audit(root: Path = ROOT) -> dict:
    images, summary = audit(root)
    destination = root / "docs" / "data_audit"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "manifest.json").write_text(json.dumps(images, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (destination / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = ["image_id", "path", "dataset", "label", "photo_type", "original_directory_split",
              "filename_prefix", "specimen_id", "capture_group", "status", "classification_ready",
              "lifetime_ready", "width", "height", "mode", "bytes", "sha256", "metadata_records"]
    with (destination / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for item in images:
            writer.writerow({**item, "metadata_records": json.dumps(item["metadata_records"], ensure_ascii=False)})
    return summary


if __name__ == "__main__":
    print(json.dumps(write_audit(), ensure_ascii=False, indent=2))
