"""PyG Graph Neural Network Foundation Encoders (GIN, GCN, GAT, GraphSAGE)."""

from __future__ import annotations

import logging
from typing import Any, List, Optional
import numpy as np
import torch
import torch.nn as nn

try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import GINConv, GCNConv, GATConv, SAGEConv, global_mean_pool, global_add_pool
except ImportError:
    Data = None
    Batch = None

from .base import BaseEncoder

logger = logging.getLogger(__name__)


def _build_batch(graph_dicts: List[Any], device: torch.device) -> Any:
    data_list = []
    for g in graph_dicts:
        if isinstance(g, Data):
            # Native PyG Data object (loaded from .pt): edge_index is already
            # in the canonical [2, E] layout and must NOT be transposed again.
            num_nodes = g.num_nodes
            x = g.x.float() if g.x is not None and g.x.numel() > 0 else torch.ones((num_nodes, 16), dtype=torch.float32)
            edge_index = g.edge_index if g.edge_index is not None else torch.empty((2, 0), dtype=torch.long)
        else:
            # Dict format (loaded from JSON): edge_index is a list of [src, dst]
            # pairs and must be transposed into PyG's [2, E] layout.
            num_nodes = g.get("num_nodes", 1)
            raw_x = g.get("x")
            if raw_x is not None and len(raw_x) > 0:
                x = torch.tensor(raw_x, dtype=torch.float32)
            else:
                # Fallback node degree / constant feature if no attributes
                x = torch.ones((num_nodes, 16), dtype=torch.float32)

            edge_list = g.get("edge_index", [])
            if len(edge_list) > 0:
                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)
        data_list.append(data)

    batch = Batch.from_data_list(data_list).to(device)
    return batch


class PyGGINEncoder(BaseEncoder):
    """Graph Isomorphism Network (GIN) encoder with global sum pooling."""

    def __init__(self, embedding_dim: int = 256, num_layers: int = 3, device: str = "cpu"):
        super().__init__(encoder_id="pyg_gin", domain="graphs", device=device); self.embedding_dim = embedding_dim
        torch.manual_seed(42)
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        in_dim = None  # Lazily initialized upon first forward

    def _init_layers(self, in_features: int):
        torch.manual_seed(42)
        self.convs = nn.ModuleList()
        h_dim = self.embedding_dim
        mlp1 = nn.Sequential(nn.Linear(in_features, h_dim), nn.ReLU(), nn.Linear(h_dim, h_dim))
        self.convs.append(GINConv(mlp1))
        for _ in range(self.num_layers - 1):
            mlp = nn.Sequential(nn.Linear(h_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, h_dim))
            self.convs.append(GINConv(mlp))
        self.convs.to(self.device)
        self.convs.eval()

    def encode(self, inputs: List[Any], batch_size: int = 64) -> np.ndarray:
        all_embs = []
        dev = torch.device(self.device)

        with torch.no_grad():
            for i in range(0, len(inputs), batch_size):
                chunk = inputs[i : i + batch_size]
                batch = _build_batch(chunk, dev)
                if len(self.convs) == 0:
                    self._init_layers(batch.x.size(1))

                # If input dimension changed between datasets, re-project node features
                if batch.x.size(1) != self.convs[0].nn[0].in_features:
                    self._init_layers(batch.x.size(1))

                x, edge_index = batch.x, batch.edge_index
                for conv in self.convs:
                    x = torch.relu(conv(x, edge_index))
                pooled = global_add_pool(x, batch.batch)
                all_embs.append(pooled.cpu().numpy())

        return np.vstack(all_embs).astype(np.float32)


class PyGGCNEncoder(BaseEncoder):
    """Graph Convolutional Network (GCN) encoder with global mean pooling."""

    def __init__(self, embedding_dim: int = 256, num_layers: int = 3, device: str = "cpu"):
        super().__init__(encoder_id="pyg_gcn", domain="graphs", device=device); self.embedding_dim = embedding_dim
        torch.manual_seed(43)
        self.num_layers = num_layers
        self.convs = nn.ModuleList()

    def _init_layers(self, in_features: int):
        torch.manual_seed(43)
        self.convs = nn.ModuleList()
        h_dim = self.embedding_dim
        self.convs.append(GCNConv(in_features, h_dim))
        for _ in range(self.num_layers - 1):
            self.convs.append(GCNConv(h_dim, h_dim))
        self.convs.to(self.device)
        self.convs.eval()

    def encode(self, inputs: List[Any], batch_size: int = 64) -> np.ndarray:
        all_embs = []
        dev = torch.device(self.device)

        with torch.no_grad():
            for i in range(0, len(inputs), batch_size):
                chunk = inputs[i : i + batch_size]
                batch = _build_batch(chunk, dev)
                if len(self.convs) == 0 or batch.x.size(1) != self.convs[0].in_channels:
                    self._init_layers(batch.x.size(1))

                x, edge_index = batch.x, batch.edge_index
                for conv in self.convs:
                    x = torch.relu(conv(x, edge_index))
                pooled = global_mean_pool(x, batch.batch)
                all_embs.append(pooled.cpu().numpy())

        return np.vstack(all_embs).astype(np.float32)


class PyGGATEncoder(BaseEncoder):
    """Graph Attention Network (GAT) encoder."""

    def __init__(self, embedding_dim: int = 256, heads: int = 4, device: str = "cpu"):
        super().__init__(encoder_id="pyg_gat", domain="graphs", device=device); self.embedding_dim = embedding_dim
        self.heads = heads
        self.convs = nn.ModuleList()

    def _init_layers(self, in_features: int):
        torch.manual_seed(44)
        self.convs = nn.ModuleList()
        h_dim = self.embedding_dim // self.heads
        self.convs.append(GATConv(in_features, h_dim, heads=self.heads))
        self.convs.append(GATConv(h_dim * self.heads, h_dim, heads=self.heads))
        self.convs.to(self.device)
        self.convs.eval()

    def encode(self, inputs: List[Any], batch_size: int = 64) -> np.ndarray:
        all_embs = []
        dev = torch.device(self.device)

        with torch.no_grad():
            for i in range(0, len(inputs), batch_size):
                chunk = inputs[i : i + batch_size]
                batch = _build_batch(chunk, dev)
                if len(self.convs) == 0 or batch.x.size(1) != self.convs[0].in_channels:
                    self._init_layers(batch.x.size(1))

                x, edge_index = batch.x, batch.edge_index
                for conv in self.convs:
                    x = torch.nn.functional.elu(conv(x, edge_index))
                pooled = global_mean_pool(x, batch.batch)
                all_embs.append(pooled.cpu().numpy())

        return np.vstack(all_embs).astype(np.float32)


class PyGSAGEEncoder(BaseEncoder):
    """GraphSAGE encoder with mean aggregation."""

    def __init__(self, embedding_dim: int = 256, device: str = "cpu"):
        super().__init__(encoder_id="pyg_sage", domain="graphs", device=device); self.embedding_dim = embedding_dim
        self.convs = nn.ModuleList()

    def _init_layers(self, in_features: int):
        torch.manual_seed(45)
        self.convs = nn.ModuleList()
        h_dim = self.embedding_dim
        self.convs.append(SAGEConv(in_features, h_dim))
        self.convs.append(SAGEConv(h_dim, h_dim))
        self.convs.to(self.device)
        self.convs.eval()

    def encode(self, inputs: List[Any], batch_size: int = 64) -> np.ndarray:
        all_embs = []
        dev = torch.device(self.device)

        with torch.no_grad():
            for i in range(0, len(inputs), batch_size):
                chunk = inputs[i : i + batch_size]
                batch = _build_batch(chunk, dev)
                if len(self.convs) == 0 or batch.x.size(1) != self.convs[0].in_channels:
                    self._init_layers(batch.x.size(1))

                x, edge_index = batch.x, batch.edge_index
                for conv in self.convs:
                    x = torch.relu(conv(x, edge_index))
                pooled = global_mean_pool(x, batch.batch)
                all_embs.append(pooled.cpu().numpy())

        return np.vstack(all_embs).astype(np.float32)
