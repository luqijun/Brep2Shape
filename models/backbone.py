from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class EncoderOutput:
    """Face-, solid-, and optional edge-level encoder representations."""

    # shape: [num_faces, face_emb_dim]
    face: torch.Tensor
    # shape: [batch_size, graph_emb_dim]
    solid: torch.Tensor
    # shape: [num_edges, edge_emb_dim] | None
    edge: torch.Tensor | None = None


def _run_module(module: nn.Module, *args, use_checkpoint: bool):
    if use_checkpoint:
        return checkpoint(module, *args, use_reentrant=False)
    return module(*args)


def encode_brep(
    batch: Mapping[str, Any],
    *,
    curve_layer: nn.Module,
    surface_layer: nn.Module,
    graph_layer: nn.Module,
    use_checkpoint: bool,
) -> EncoderOutput:
    """Run the shared curve, surface, and graph encoding pipeline."""
    try:
        graph = batch["graph"]  # DGLGraph, batched solids
        line_graph = batch["line_graph"]  # DGLGraph, line graph of batched solids
        # shape: [num_edges, num_primitives, num_points, 4*11]
        edge = graph.edata["edge"]
        # shape: [num_edges, num_primitives]
        edge_padding_mask = graph.edata["edge_padding_mask"]
        # shape: [num_faces, num_primitives, num_points, 28*4+1+7]
        face = graph.ndata["face"]
        # shape: [num_faces, num_primitives, 7]
        tri_normal = graph.ndata["tri_normal"]
        # shape: [num_faces, num_primitives]
        face_vis_mask = graph.ndata["face_vis_mask"]
        # shape: [num_faces, num_primitives]
        face_padding_mask = graph.ndata["face_padding_mask"]
    except KeyError as exc:
        raise KeyError(f"Model batch is missing required feature {exc.args[0]!r}") from exc

    # shape: [num_edges, curve_emb_dim]
    edge_embedding = _run_module(
        curve_layer,
        edge,
        edge_padding_mask,
        use_checkpoint=use_checkpoint,
    )
    # shape: [num_faces, surface_emb_dim]
    face_embedding = _run_module(
        surface_layer,
        face,
        tri_normal,
        face_vis_mask,
        face_padding_mask,
        use_checkpoint=use_checkpoint,
    )
    # graph_outputs: (face_output, solid_output) or (face_output, solid_output, edge_output)
    # face_output: [num_faces, graph_emb_dim], solid_output: [batch_size, graph_emb_dim]
    graph_outputs = _run_module(
        graph_layer,
        graph,
        face_embedding,
        edge_embedding,
        line_graph,
        use_checkpoint=use_checkpoint,
    )
    if not isinstance(graph_outputs, tuple) or len(graph_outputs) not in {2, 3}:
        raise TypeError("Graph encoder must return two or three tensors")

    face_output, solid_output = graph_outputs[:2]
    edge_output = graph_outputs[2] if len(graph_outputs) == 3 else None
    return EncoderOutput(face=face_output, solid=solid_output, edge=edge_output)
