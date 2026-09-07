#!/usr/bin/env python
"""Cluster STEP files by B-rep similarity using a trained Brep2Shape checkpoint.

Recursively walks one or more STEP directories, extracts 128-d solid
embeddings with the pretrained encoders, estimates the number of clusters
automatically (KMeans + silhouette on a subsample), then copies each file
into numbered folders (0001, 0002, ...) ordered by descending cluster size.

Usage:
  python cluster_step.py --checkpoint results/.../last.ckpt \
      --input_dir DIR_A [--input_dir DIR_B ...] --output_dir clustered
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
from collections import Counter, defaultdict

import numpy as np
import torch

import step_preprocess
from datasets.pretraining_dataset import PretrainingDataset
from models.backbone import encode_brep
from models.pretraining import PretrainingPL


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Pretraining ckpt from pretrain.py")
    parser.add_argument(
        "--input_dir",
        action="append",
        required=True,
        help="STEP file or directory (repeatable, recursive)",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_clusters", type=int, default=50)
    parser.add_argument("--min_clusters", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--preprocess_workers",
        type=int,
        default=8,
        help="CPU processes for STEP preprocessing",
    )
    parser.add_argument("--max_faces", type=int, default=256)
    parser.add_argument("--silhouette_sample", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def build_cache(files, cache_dir: pathlib.Path, args) -> dict[str, str]:
    """Preprocess STEP files into cache_dir (resumable); return manifest."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / "source_index.json"
    manifest: dict[str, str] = {}
    if index_path.is_file():
        manifest = json.loads(index_path.read_text(encoding="utf-8"))

    def sample_done(name: str) -> bool:
        return all(
            (cache_dir / rel).is_file()
            for rel in (
                f"features/triangles/{name}.pt",
                f"features/topology/{name}.pt",
                f"graphs/{name}.bin",
                f"line_graphs/{name}.bin",
            )
        )

    done_sources = {src for name, src in manifest.items() if sample_done(name)}
    manifest = {name: src for name, src in manifest.items() if sample_done(name)}
    todo = [f for f in files if str(pathlib.Path(f).resolve()) not in done_sources]
    print(f"Cache: {len(done_sources)} sources already cached, {len(todo)} to process")

    skipped = []
    if todo:
        new_manifest, skipped = step_preprocess.preprocess_files(
            todo,
            str(cache_dir),
            workers=args.preprocess_workers,
            max_faces=args.max_faces,
        )
        manifest.update(new_manifest)

    step_preprocess.write_datasplit(str(cache_dir), manifest, splits=("test",))
    step_preprocess.write_meta(str(cache_dir), manifest, skipped)
    return manifest


@torch.no_grad()
def extract_embeddings(dataset_dir: str, checkpoint: str, args) -> np.ndarray:
    device = torch.device(args.device)
    dataset = PretrainingDataset(dataset_dir, split="test", lazy_load=True)
    loader = dataset.get_dataloader(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    model = PretrainingPL.load_from_checkpoint(checkpoint, map_location="cpu")
    model.eval().to(device)

    embeddings = []
    for batch in loader:
        batch["graph"] = batch["graph"].to(device)
        batch["line_graph"] = batch["line_graph"].to(device)
        enc = encode_brep(
            batch,
            curve_layer=model.model.curve_layer,
            surface_layer=model.model.surface_layer,
            graph_layer=model.model.graph_layer,
            use_checkpoint=False,
        )
        embeddings.append(enc.solid.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def estimate_k(embeddings: np.ndarray, k_min: int, k_max: int, sample: int, seed: int):
    """Pick k by silhouette score (cosine) on a subsample; fall back to k_min."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = len(embeddings)
    k_max = min(k_max, max(1, n - 1))
    k_min = min(k_min, k_max)
    if k_max <= 1 or n < 3:
        return 1, None, {}

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(sample, n), replace=False)
    sub = embeddings[idx]

    scores = {}
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(sub)
        if len(set(km.labels_)) < 2:
            scores[k] = -1.0
            continue
        scores[k] = float(silhouette_score(sub, km.labels_, metric="cosine"))
    best_k = max(scores, key=scores.get)
    return best_k, scores.get(best_k), scores


def copy_with_suffix(src: pathlib.Path, dst_dir: pathlib.Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    i = 0
    while dst.exists():
        i += 1
        dst = dst_dir / f"{src.stem}_{i}{src.suffix}"
    shutil.copy2(src, dst)


def main():
    args = build_parser().parse_args()
    out_dir = pathlib.Path(args.output_dir).resolve()
    cache_dir = out_dir / "_cache"

    files = step_preprocess.collect_step_files(args.input_dir, exclude_dir=out_dir)
    print(f"Found {len(files)} STEP files")
    if not files:
        return

    manifest = build_cache(files, cache_dir, args)
    if not manifest:
        print("No processable samples; nothing to cluster")
        return

    embeddings = extract_embeddings(str(cache_dir), args.checkpoint, args)
    names = [
        item["face"].split("/")[-1][:-3]
        for item in json.loads((cache_dir / "datasplit.json").read_text(encoding="utf-8"))["test"]
    ]
    assert len(names) == len(embeddings), (len(names), len(embeddings))
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-12, None)
    print(f"Embeddings: {embeddings.shape}")

    best_k, best_score, all_scores = estimate_k(
        embeddings,
        args.min_clusters,
        args.max_clusters,
        args.silhouette_sample,
        args.seed,
    )
    print(f"Auto-selected k={best_k} (silhouette={best_score})")

    if best_k <= 1:
        sample_labels = np.zeros(len(embeddings), dtype=int)
    else:
        from sklearn.cluster import KMeans

        sample_labels = KMeans(n_clusters=best_k, n_init=10, random_state=args.seed).fit_predict(embeddings)

    # file-level labels: majority vote over its solids
    file_votes: dict[str, Counter] = defaultdict(Counter)
    for name, label in zip(names, sample_labels):
        file_votes[manifest[name]][int(label)] += 1
    file_labels = {src: votes.most_common(1)[0][0] for src, votes in file_votes.items()}

    cluster_files: dict[int, list[str]] = defaultdict(list)
    for src, label in file_labels.items():
        cluster_files[label].append(src)
    ordered = sorted(cluster_files.items(), key=lambda kv: (-len(kv[1]), kv[0]))

    np.save(out_dir / "embeddings.npy", embeddings)
    report = {
        "checkpoint": args.checkpoint,
        "num_samples": len(names),
        "num_files": len(file_labels),
        "selected_k": best_k,
        "silhouette": best_score,
        "silhouette_by_k": all_scores,
        "clusters": {},
        "input_dirs": args.input_dir,
        "output_dir": str(out_dir),
    }
    for rank, (label, srcs) in enumerate(ordered, start=1):
        folder = out_dir / f"{rank:04d}"
        for src in srcs:
            copy_with_suffix(pathlib.Path(src), folder)
        report["clusters"][f"{rank:04d}"] = {
            "cluster_id": label,
            "count": len(srcs),
            "files": sorted(srcs),
        }
        print(f"{rank:04d}: {len(srcs)} files")

    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print(f"Done -> {out_dir} (report.json)")


if __name__ == "__main__":
    main()
