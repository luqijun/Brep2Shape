#!/usr/bin/env python
"""Convert STEP (.stp/.step) files into the Brep2Shape pretraining data format.

几何与拓扑处理复用官方 ``preprocess`` 包（BRT 管线）：
  - ``preprocess.batch.load_step_model``               STEP 加载
  - ``preprocess.patches.extract_face_patches``        面片（三角 Bezier (P,28,4)
    + in_mask + tri_normals + 采样点）
  - ``preprocess.topology.edge_bezier_control_points`` 边曲线 (弧段, 11, 4)
本脚本仅补充 Brep2Shape 训练所需的部分：
  - uv_face_points [F,3,3,3] / uv_edge_points [E,3,3] 监督目标（参数域均匀采样）
  - 面邻接图 graph.bin 与边邻接图 line_graph.bin（DGL）
  - datasplit.json（train/val/test）

Usage:
  python step_preprocess.py --input_dir DIR [--input_dir DIR2 ...] \
      --output_dir OUT [--workers N] [--max_faces 256] [--num_samples 64] [--timeout 300]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pathlib
import re
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError

import dgl
import numpy as np
import torch
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopAbs import TopAbs_EDGE
from OCC.Core.TopExp import topexp_MapShapes
from OCC.Core.TopTools import TopTools_IndexedMapOfShape
from occwl.entity_mapper import EntityMapper

from preprocess.batch import load_step_model
from preprocess.patches import extract_face_patches
from preprocess.sampling import face_point_evaluator
from preprocess.topology import edge_bezier_control_points

# line-graph 全连接边对的面内边数上限（超过则按环绕序连接）
LINE_GRAPH_FULL_PAIRS = 24
# uv 监督目标的采样密度（Brep2Shape 默认 3x3 UV 网格）
UV_SAMPLES = 3


# ---------------------------------------------------------------------------
# 单实体处理
# ---------------------------------------------------------------------------


def _face_uv_grid(face, n: int = UV_SAMPLES) -> np.ndarray:
    """面参数域包围盒上 n x n 均匀网格的 3D 点 [n, n, 3]。"""
    box = face.uv_bounds()
    lo, hi = np.asarray(box.min_point()), np.asarray(box.max_point())
    umin, umax, vmin, vmax = lo[0], hi[0], lo[1], hi[1]
    evaluate = face_point_evaluator(face)
    grid = np.zeros((n, n, 3), dtype=np.float32)
    for i, u in enumerate(np.linspace(umin, umax, n)):
        for j, v in enumerate(np.linspace(vmin, vmax, n)):
            grid[i, j] = evaluate(np.array([u, v]))
    return grid


def _edge_uv_points(edge) -> np.ndarray:
    """边曲线参数区间上均匀 3 点 [3, 3]。"""
    curve, first, last = BRep_Tool.Curve(edge.topods_shape())
    points = np.zeros((UV_SAMPLES, 3), dtype=np.float32)
    for i, t in enumerate(np.linspace(first, last, UV_SAMPLES)):
        p = curve.Value(float(t))
        points[i] = (p.X(), p.Y(), p.Z())
    return points


def process_solid(compound, solid, max_faces: int, num_samples: int, rng) -> dict | None:
    """把一个 occwl Solid 转换为 Brep2Shape 样本（face/topo/graph/line_graph）。"""
    faces = list(solid.faces())
    if not faces or len(faces) > max_faces:
        return None
    mapper = EntityMapper(solid)

    # --- 面片提取（复用 preprocess.patches）--------------------------------
    patch_list = []
    for face in faces:
        try:
            patch_list.append(extract_face_patches(face, num_samples=num_samples, rng=rng))
        except Exception:
            patch_list.append(None)
    kept_face_idx = [i for i, p in enumerate(patch_list) if p is not None]
    if not kept_face_idx:
        return None
    # 统一索引空间：EntityMapper 的 face_index 与 solid.faces() 遍历序一致
    old2new_face = {mapper.face_index(faces[i]): new for new, i in enumerate(kept_face_idx)}

    # --- 环扫描 + 去重边（顺序与 preprocess.topology 一致）-------------------
    edge_map = TopTools_IndexedMapOfShape()
    topexp_MapShapes(solid.topods_shape(), TopAbs_EDGE, edge_map)
    face_wires: list[list[list]] = []  # 每面：每环的有序边对象列表
    edge_objects: list = []  # 去重后的实体边（首次出现序）
    compound2global: dict[int, int] = {}
    for face in faces:
        wire_lists = []
        for wire in face.wires():
            wire_lists.append([e for e in wire.ordered_edges() if e.has_curve()])
        face_wires.append(wire_lists)
    for wire_lists in face_wires:
        for edges in wire_lists:
            for edge in edges:
                cid = edge_map.FindIndex(edge.topods_shape())
                if cid not in compound2global:
                    compound2global[cid] = len(edge_objects)
                    edge_objects.append(edge)
    if not edge_objects:
        return None

    # --- 边控制点 + 邻接面（复用 preprocess.topology / occwl）----------------
    edge_cps, edge_adj, kept_edge_idx = [], [], []
    for ei, edge in enumerate(edge_objects):
        try:
            cps = edge_bezier_control_points(edge)
        except Exception:
            continue
        adj = [new for f in solid.faces_from_edge(edge) if (new := old2new_face.get(mapper.face_index(f))) is not None]
        adj = list(dict.fromkeys(adj))
        if not adj:
            continue
        # 非流形边（邻接面 > 2）无法由单条图边（仅两个端点）表达，第 3 个面产生的
        # line-graph 配对会违反 dual_encoder 的共面不变量（models/dual_encoder.py:365），丢弃。
        if len(adj) > 2:
            continue
        edge_cps.append((cps, _edge_uv_points(edge)))
        edge_adj.append(adj)
        kept_edge_idx.append(ei)
    if not edge_cps:
        return None
    old2new_edge = {old: new for new, old in enumerate(kept_edge_idx)}
    num_faces = len(kept_face_idx)
    num_edges = len(edge_cps)

    # --- face 文件 ------------------------------------------------------------
    face_dict = {
        "nodes": [patch_list[i].nodes for i in kept_face_idx],
        "in_mask": [patch_list[i].trimmed_mask for i in kept_face_idx],
        "tri_normals": [patch_list[i].patch_features for i in kept_face_idx],
        "points": np.stack([_sanitize_points(patch_list[i].points, faces[i]) for i in kept_face_idx]),
        "uv_face_points": np.stack([_face_uv_grid(faces[i]) for i in kept_face_idx]),
    }

    # --- 拓扑索引（wire_index / edge_index / adj_face_index）------------------
    wire_index, edge_index = [], []
    for old_fi in kept_face_idx:
        wids = []
        for edges in face_wires[old_fi]:
            eids = [
                old2new_edge[compound2global[edge_map.FindIndex(e.topods_shape())]]
                for e in edges
                if edge_map.FindIndex(e.topods_shape()) in compound2global
                and compound2global[edge_map.FindIndex(e.topods_shape())] in old2new_edge
            ]
            if eids:
                wids.append(len(edge_index))
                edge_index.append(eids)
        wire_index.append(wids)

    topo_dict = {
        "edge": [cps for cps, _uv in edge_cps],
        "uv_edge_points": np.stack([uv for _cps, uv in edge_cps]),
        "edge_index": edge_index,
        "wire_index": wire_index,
        "adj_face_index": edge_adj,
    }

    # --- DGL 图：节点=面，边=B-rep 边（连接其邻接面）---------------------------
    src = torch.tensor([adj[0] for adj in edge_adj], dtype=torch.int64)
    dst = torch.tensor([adj[1] if len(adj) > 1 else adj[0] for adj in edge_adj], dtype=torch.int64)
    graph = dgl.graph((src, dst), num_nodes=num_faces)

    # --- DGL 对偶图：节点=B-rep 边，边=同面共边对 -----------------------------
    line_src, line_dst = [], []
    for old_fi in kept_face_idx:
        face_edge_ids = list(
            dict.fromkeys(
                old2new_edge[compound2global[edge_map.FindIndex(e.topods_shape())]]
                for edges in face_wires[old_fi]
                for e in edges
                if edge_map.FindIndex(e.topods_shape()) in compound2global
                and compound2global[edge_map.FindIndex(e.topods_shape())] in old2new_edge
            )
        )
        n = len(face_edge_ids)
        if n < 2:
            continue
        if n <= LINE_GRAPH_FULL_PAIRS:
            for a in range(n):
                for b in range(n):
                    if a != b:
                        line_src.append(face_edge_ids[a])
                        line_dst.append(face_edge_ids[b])
        else:
            for a in range(n):
                b = (a + 1) % n
                line_src += [face_edge_ids[a], face_edge_ids[b]]
                line_dst += [face_edge_ids[b], face_edge_ids[a]]
    line_graph = dgl.graph(
        (
            torch.tensor(line_src, dtype=torch.int64) if line_src else torch.zeros(0, dtype=torch.int64),
            torch.tensor(line_dst, dtype=torch.int64) if line_dst else torch.zeros(0, dtype=torch.int64),
        ),
        num_nodes=num_edges,
    )

    return {"face": face_dict, "topo": topo_dict, "graph": graph, "line_graph": line_graph}


def _sanitize_points(points: np.ndarray, face) -> np.ndarray:
    """把采样点中的非有限值替换为参数域网格点（退化面求值可能产生 NaN）。"""
    if np.isfinite(points).all():
        return points.astype(np.float32)
    finite = np.isfinite(points).all(axis=1)
    fallback = _face_uv_grid(face).reshape(-1, 3)
    points = points.copy()
    k = 0
    for i in range(len(points)):
        if not finite[i]:
            points[i] = fallback[k % len(fallback)]
            k += 1
    return np.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# ---------------------------------------------------------------------------
# 文件级驱动
# ---------------------------------------------------------------------------


def _sample_name(path: pathlib.Path) -> str:
    stem = re.sub(r"[^0-9A-Za-z_.-]+", "_", path.stem).strip("_") or "model"
    digest = hashlib.md5(str(path.resolve()).encode()).hexdigest()[:8]
    return f"{stem}_{digest}"


def _atomic_save(save_fn, path: pathlib.Path) -> None:
    """先写临时文件再原子替换，避免 worker 超时被杀时留下半成品输出。"""
    tmp = path.with_name(path.name + ".tmp")
    save_fn(tmp)
    os.replace(tmp, path)


def process_step_file(args):
    """Worker 入口：一个 STEP 文件 -> 一个样本（官方管线仅取主实体）。"""
    path, out_dir, max_faces, num_samples = args
    path = pathlib.Path(path)
    out = pathlib.Path(out_dir)

    # ---------- 跳过已处理的文件 ----------
    name = _sample_name(path)
    face_path = out / "features" / "triangles" / f"{name}.pt"
    topo_path = out / "features" / "topology" / f"{name}.pt"
    graph_path = out / "graphs" / f"{name}.bin"
    line_graph_path = out / "line_graphs" / f"{name}.bin"
    if face_path.exists() and topo_path.exists() and graph_path.exists() and line_graph_path.exists():
        print(f"[skip] {path}: already processed", flush=True)
        return (path, [name], None)  # 当作成功处理，但实际未处理，manifest 会保留该样本
    # -----------------------------------------

    start = time.monotonic()
    try:
        compound, solid, _ = load_step_model(path)
        rng = np.random.default_rng(zlib.crc32(str(path.resolve()).encode("utf-8")))
        sample = process_solid(compound, solid, max_faces, num_samples, rng)
        if sample is None:
            return (path, [], "no processable solid")
        _atomic_save(lambda p: torch.save(sample["face"], p), face_path)
        _atomic_save(lambda p: torch.save(sample["topo"], p), topo_path)
        _atomic_save(lambda p: dgl.save_graphs(str(p), [sample["graph"]]), graph_path)
        _atomic_save(lambda p: dgl.save_graphs(str(p), [sample["line_graph"]]), line_graph_path)
        elapsed = time.monotonic() - start
        if elapsed > 30:
            print(f"[slow] {path}: {elapsed:.1f}s", flush=True)
        return (path, [name], None)
    except Exception as exc:
        return (path, [], f"{type(exc).__name__}: {exc}")


def collect_step_files(input_dirs, exclude_dir=None):
    exclude = pathlib.Path(exclude_dir).resolve() if exclude_dir else None
    files = []
    for d in input_dirs:
        root = pathlib.Path(d)
        if root.is_file():
            files.append(root)
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower().endswith((".stp", ".step")):
                    p = pathlib.Path(dirpath) / fn
                    if exclude is not None and exclude in p.resolve().parents:
                        continue
                    files.append(p)
    seen, unique = set(), []
    for f in files:
        r = str(f.resolve())
        if r not in seen:
            seen.add(r)
            unique.append(f)
    return unique


def _limit_threads() -> None:
    """Worker 初始化：单线程运行，避免 workers 个进程各自再占满全部核。"""
    torch.set_num_threads(1)


def _kill_pool(pool: ProcessPoolExecutor) -> None:
    """杀掉池内所有 worker（超时路径专用；卡死的 OCC C++ 调用无法优雅中断）。"""
    procs = list(getattr(pool, "_processes", {}).values())
    pool.shutdown(wait=False, cancel_futures=True)
    for proc in procs:
        proc.kill()
    for proc in procs:
        proc.join()


def preprocess_files(files, out_dir, workers=8, max_faces=256, num_samples=64, timeout=300):
    out = pathlib.Path(out_dir)
    for sub in ("features/triangles", "features/topology", "graphs", "line_graphs"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    manifest, skipped = {}, []
    pending = [(f, out, max_faces, num_samples) for f in files]
    # 滑窗提交：同时在飞的任务最多 2*workers 个。个别卡死任务超时后只需
    # 杀掉当前池重建，不会让后面上千个任务全部陪跑超时。
    window = max(2 * workers, workers + 2)
    timeout_arg = timeout if timeout > 0 else None

    while pending:
        inflight = pending[:window]
        pending = pending[window:]
        pool = ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=64, initializer=_limit_threads)
        futures = [(job, pool.submit(process_step_file, job)) for job in inflight]
        killed = False
        for i, (job, future) in enumerate(futures):
            path = job[0]
            try:
                _, names, error = future.result(timeout=timeout_arg)
            except FutureTimeoutError:
                skipped.append({"file": str(path), "error": f"timeout after {timeout} seconds"})
                print(f"[timeout] {path}: exceeded {timeout}s, killing worker pool", flush=True)
                # 本窗口内尚未取到结果的任务重新入队（已完成者由 skip 检查秒过）
                pending = [j for j, _ in futures[i + 1 :]] + pending
                _kill_pool(pool)
                killed = True
                break
            except Exception as e:  # 捕获其他意外异常，避免崩溃
                skipped.append({"file": str(path), "error": f"unexpected error: {e}"})
                print(f"[error] {path}: {e}", flush=True)
                continue
            if error is not None:
                skipped.append({"file": str(path), "error": error})
                print(f"[skip] {path}: {error}", flush=True)
            else:
                for name in names:
                    manifest[name] = str(path.resolve())
        if not killed:
            pool.shutdown(wait=True)

    return manifest, skipped


def write_datasplit(out_dir, manifest, val_ratio=0.05, seed=0, splits=("train", "val", "test")):
    """写 datasplit.json（默认 train/val/test，test 同 val；``splits=("test",)`` 时全部进 test）。"""
    import random

    names = sorted(manifest)
    if len(splits) == 1 and splits[0] == "test":
        train, val = [], names
    else:
        rng = random.Random(seed)
        rng.shuffle(names)
        n_val = max(1, int(len(names) * val_ratio)) if len(names) > 1 else len(names)
        val, train = names[:n_val], names[n_val:] or names

    def item(name):
        return {
            "face": f"features/triangles/{name}.pt",
            "topo": f"features/topology/{name}.pt",
            "graph": f"graphs/{name}.bin",
            "line_graph": f"line_graphs/{name}.bin",
        }

    split = {"train": [item(n) for n in train], "val": [item(n) for n in val], "test": [item(n) for n in val]}
    with open(pathlib.Path(out_dir) / "datasplit.json", "w", encoding="utf-8") as f:
        json.dump(split, f, indent=1)
    return split


def write_meta(out_dir, manifest, skipped):
    out = pathlib.Path(out_dir)
    with open(out / "source_index.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
    with open(out / "skipped.json", "w", encoding="utf-8") as f:
        json.dump(skipped, f, indent=1, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", action="append", required=True, help="STEP 文件或目录（可重复，递归遍历）")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4))
    parser.add_argument("--max_faces", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=64, help="每面采样点数（用于归一化，BRT 默认 256）")
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--timeout", type=float, default=300.0, help="每个文件的处理超时时间（秒），<=0 表示不设超时")
    args = parser.parse_args()

    log_level = os.environ.get("LOG_LEVEL", "INFO")
    logging.basicConfig(level=log_level, format="%(levelname)s %(name)s: %(message)s")
    logging.captureWarnings(True)

    files = collect_step_files(args.input_dir, exclude_dir=args.output_dir)
    # 按文件大小升序处理：小文件先完成产出，少数超大装配体排在队尾，
    # 它们导致的超时/杀池不会拖累前面上千个正常文件。
    files.sort(key=lambda f: f.stat().st_size)
    print(f"Found {len(files)} STEP files", flush=True)
    if not files:
        return

    manifest, skipped = preprocess_files(
        files,
        args.output_dir,
        workers=args.workers,
        max_faces=args.max_faces,
        num_samples=args.num_samples,
        timeout=args.timeout,
    )
    write_datasplit(args.output_dir, manifest, val_ratio=args.val_ratio)
    write_meta(args.output_dir, manifest, skipped)
    print(
        f"Done: {len(manifest)} samples, {len(skipped)} skipped -> {pathlib.Path(args.output_dir) / 'datasplit.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
