"""Load semantic/spatial graph caches and batch disconnected WSI graphs without padding."""
from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset

from graphs import branch_inputs

BRANCHES = ('semantic', 'spatial')


def _classes(data):
    classes = sorted(data['metadata']['classes'], key=lambda c: c['id'])
    if [c['id'] for c in classes] != list(range(len(classes))) or not classes:
        raise ValueError('Semantic class IDs must be contiguous from zero')
    return tuple(c['name'] for c in classes)


def compact_sample(data, *, label, cache_path=''):
    """Keep raw nodes once; each branch carries compact edges and a raw-node lookup."""
    features, coords, semantic = data['features'], data['coords'], data['semantic_label']
    classes = _classes(data)
    n = len(features)
    if features.ndim != 2 or not n or not features.is_floating_point():
        raise ValueError('Expected nonempty floating features[N,D]')
    if coords.shape != (n, 2) or coords.dtype != torch.long:
        raise ValueError('coords must be int64[N,2]')
    if semantic.shape != (n,) or semantic.dtype != torch.long:
        raise ValueError('semantic_label must be int64[N]')
    if not torch.isfinite(features).all() or not torch.all((semantic >= 0) & (semantic < len(classes))):
        raise ValueError('Invalid features or semantic labels')
    if not torch.equal(data['feature_row'], torch.arange(n)):
        raise ValueError('Cache feature rows have been reordered')
    if label not in (0, 1):
        raise ValueError('WSI label must be 0 or 1')
    if 'label' in data and int(data['label']) != label:
        raise ValueError('Manifest label differs from cache label')
    branches = {}
    for branch in BRANCHES:
        ids, mask = data[branch + '_graph_id'], data[branch + '_node_mask']
        edges, attrs = data[branch + '_edge_index'], data[branch + '_edge_attr']
        if ids.shape != (n,) or ids.dtype != torch.long or not torch.all(ids >= -1):
            raise ValueError(f'{branch}: invalid global graph IDs')
        if mask.dtype != torch.bool or not torch.equal(mask, ids >= 0):
            raise ValueError(f'{branch}: node mask differs from graph IDs')
        if edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype != torch.long:
            raise ValueError(f'{branch}: invalid edge_index')
        if edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= n):
            raise ValueError(f'{branch}: global edge index outside feature rows')
        if attrs.shape != (edges.shape[1], 3) or not torch.isfinite(attrs).all():
            raise ValueError(f'{branch}: invalid edge attributes')
        if not mask[edges].all() or not torch.equal(ids[edges[0]], ids[edges[1]]):
            raise ValueError(f'{branch}: discarded-node or cross-graph edge')
        unique = ids[mask].unique()
        if not torch.equal(unique, torch.arange(len(unique))):
            raise ValueError(f'{branch}: graph IDs must be dense')
        loops = edges[0] == edges[1]
        if not torch.equal(edges[0, loops].sort().values, mask.nonzero().flatten()):
            raise ValueError(f'{branch}: expected exactly one self-loop per retained node')
        if attrs[loops].any():
            raise ValueError(f'{branch}: self-loop attributes must be zero')
        compact = branch_inputs(data, branch)
        branches[branch] = {
            'node_index': compact['original_node_index'],
            'graph_id': compact['graph_id'], 'edge_index': compact['edge_index'],
            'edge_attr': compact['edge_attr'].float(), 'num_graphs': len(unique)}
    return {'slide_id': data['slide_id'], 'cache_path': str(cache_path), 'label': int(label),
            'features': features, 'coords': coords, 'semantic_label': semantic,
            'class_names': classes, 'branches': branches}


class WSIGraphDataset(Dataset):
    """Use a cleaned master manifest or a fold manifest with split=train/val/test."""

    def __init__(self, manifest, split=None, *, cache_samples=False):
        self._sample_cache = {} if cache_samples else None
        self.manifest = Path(manifest).expanduser().resolve()
        with self.manifest.open() as stream:
            records = list(csv.DictReader(stream))
        if not records:
            raise ValueError('Empty manifest')
        if split is not None:
            available = {r['split'] for r in records}
            if split not in available:
                raise ValueError(f'Split {split!r} absent; available={sorted(available)}')
            records = [r for r in records if r['split'] == split]
        self.records = []
        seen = set()
        for row in records:
            slide_id = row.get('slide_id') or row['case_name']
            identity = (row['class_name'], slide_id)
            if identity in seen or row['status'] != 'PASS' or int(row['label']) not in (0, 1):
                raise ValueError(f'Duplicate or invalid manifest sample: {identity}')
            seen.add(identity)
            path = Path(row['cache_path']).expanduser()
            path = path.resolve() if path.is_absolute() else (self.manifest.parent / path).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            self.records.append({**row, 'slide_id': slide_id, 'cache_path': str(path)})
        sample = self._load(0)
        self.input_dim = sample['features'].shape[1]
        self.class_names = sample['class_names']
        self.num_semantic_classes = len(self.class_names)

    def __len__(self):
        return len(self.records)

    def _load(self, index):
        if self._sample_cache is not None and index in self._sample_cache:
            return self._sample_cache[index]
        row = self.records[index]
        data = torch.load(row['cache_path'], map_location='cpu', weights_only=True)
        if data['slide_id'] != row['slide_id']:
            raise ValueError('Manifest slide ID differs from cache')
        sample = compact_sample(data, label=int(row['label']), cache_path=row['cache_path'])
        if self._sample_cache is not None:
            self._sample_cache[index] = sample
        return sample

    def __getitem__(self, index):
        sample = self._load(index)
        if sample['features'].shape[1] != self.input_dim or sample['class_names'] != self.class_names:
            raise ValueError('Inconsistent feature dimension or semantic class mapping')
        return sample


def collate_wsi_graphs(samples):
    """Offset raw nodes, retained branch nodes and graph IDs independently."""
    if not samples:
        raise ValueError('Cannot collate an empty WSI batch')
    classes = samples[0]['class_names']
    width = samples[0]['features'].shape[1]
    if any(s['class_names'] != classes or s['features'].shape[1] != width for s in samples):
        raise ValueError('Incompatible feature dimensions or class mappings')
    node_counts = [len(s['features']) for s in samples]
    node_ptr = torch.tensor([0, *node_counts]).cumsum(0)
    result = {'features': torch.cat([s['features'].float() for s in samples]),
              'coords': torch.cat([s['coords'] for s in samples]),
              'semantic_label': torch.cat([s['semantic_label'] for s in samples]),
              'labels': torch.tensor([s['label'] for s in samples], dtype=torch.float32),
              'slide_ids': [s['slide_id'] for s in samples],
              'cache_paths': [s['cache_path'] for s in samples],
              'class_names': classes, 'num_slides': len(samples), 'node_ptr': node_ptr,
              'node_slide_index': torch.repeat_interleave(torch.arange(len(samples)),
                                                         torch.tensor(node_counts)), 'branches': {}}
    for branch in BRANCHES:
        groups = [s['branches'][branch] for s in samples]
        branch_ptr = torch.tensor([0, *[len(g['node_index']) for g in groups]]).cumsum(0)
        graph_ptr = torch.tensor([0, *[g['num_graphs'] for g in groups]]).cumsum(0)
        result['branches'][branch] = {
            'node_index': torch.cat([g['node_index'] + node_ptr[i] for i, g in enumerate(groups)]),
            'original_node_index': torch.cat([g['node_index'] for g in groups]),
            'graph_id': torch.cat([g['graph_id'] + graph_ptr[i] for i, g in enumerate(groups)]),
            'edge_index': torch.cat([g['edge_index'] + branch_ptr[i] for i, g in enumerate(groups)], dim=1),
            'edge_attr': torch.cat([g['edge_attr'] for g in groups]),
            'node_ptr': branch_ptr, 'graph_ptr': graph_ptr, 'num_graphs': int(graph_ptr[-1]),
            'graph_slide_index': torch.repeat_interleave(torch.arange(len(samples)),
                                                        torch.tensor([g['num_graphs'] for g in groups])),
            'graph_local_id': torch.cat([torch.arange(g['num_graphs']) for g in groups])}
    return result


def move_graph_batch(batch, device):
    """Move tensor fields recursively; preserve file paths and Python metadata."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: move_graph_batch(v, device) for k, v in batch.items()}
    return batch
