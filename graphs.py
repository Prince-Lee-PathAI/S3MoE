#!/usr/bin/env python3
"""Deterministic semantic and spatial subgraph construction."""
from collections import deque
from copy import deepcopy
import numpy as np
import torch

NEIGHBORS = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]


def components(nodes, adjacency):
    remaining = set(int(i) for i in nodes)
    found = []
    for root in sorted(remaining):
        if root not in remaining:
            continue
        remaining.remove(root)
        queue, group = deque([root]), []
        while queue:
            node = queue.popleft()
            group.append(node)
            for neighbor in adjacency[node]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
        found.append(sorted(group))
    return found


def edge_attributes(coords, edges):
    displacement = (coords[edges[1]] - coords[edges[0]]).float()
    return torch.cat((displacement, displacement.norm(dim=1, keepdim=True)), dim=1)


def build_graphs(coords: torch.Tensor, labels: torch.Tensor, semantic_max_nodes=256,
                 grid_rows=8, grid_cols=8, semantic_radius=1):
    if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) == 0:
        raise ValueError('coords must be nonempty [N, 2]')
    if len(labels) != len(coords) or min(semantic_max_nodes, grid_rows, grid_cols) <= 0:
        raise ValueError('Invalid labels or graph parameters')
    if not isinstance(semantic_radius, int) or semantic_radius < 1:
        raise ValueError('semantic_radius must be a positive integer in patch grid units')
    xy = coords.cpu().numpy().astype(np.int64)
    y = labels.cpu().numpy()
    lookup = {tuple(value): i for i, value in enumerate(xy)}
    if len(lookup) != len(coords):
        raise ValueError('Duplicate coordinates must be resolved before graph construction')
    sources, targets, semantic_sources, semantic_targets = [], [], [], []
    adjacency = [[] for _ in xy]
    semantic_neighbors = [(dr, dc) for dr in range(-semantic_radius, semantic_radius + 1)
                          for dc in range(-semantic_radius, semantic_radius + 1) if dr or dc]
    for i, (row, col) in enumerate(xy):
        for dr, dc in NEIGHBORS:
            j = lookup.get((int(row + dr), int(col + dc)))
            if j is not None:
                sources.append(i); targets.append(j)
        for dr, dc in semantic_neighbors:
            j = lookup.get((int(row + dr), int(col + dc)))
            if j is not None and y[i] == y[j]:
                semantic_sources.append(i); semantic_targets.append(j)
                adjacency[i].append(j)
    source, target = np.asarray(sources, dtype=np.int64), np.asarray(targets, dtype=np.int64)
    initial = components(range(len(xy)), adjacency)
    queue, final, splits = deque(initial), [], []
    while queue:
        group = queue.popleft()
        if len(group) <= semantic_max_nodes:
            final.append(group)
            continue
        positions = xy[group]
        span = np.ptp(positions, axis=0)
        axis = int(np.argmax(span))
        values = np.unique(positions[:, axis])
        if len(values) < 2:
            raise RuntimeError('Oversized component cannot be spatially split')
        # Threshold lies between integer grid lines. Both sides are nonempty.
        boundary = int(values[len(values) // 2])
        left = [i for i in group if xy[i, axis] < boundary]
        right = [i for i in group if xy[i, axis] >= boundary]
        if not left or not right:
            raise RuntimeError('Empty side in semantic component split')
        subgroups = components(left, adjacency) + components(right, adjacency)
        queue.extend(subgroups)
        splits.append({'parent_nodes': len(group), 'axis': ['row', 'col'][axis],
                       'threshold': boundary - 0.5, 'child_sizes': [len(g) for g in subgroups]})
    final.sort(key=lambda group: min(group))
    semantic_ids = np.empty(len(xy), dtype=np.int64)
    for graph_id, group in enumerate(final):
        semantic_ids[group] = graph_id
    minimum, maximum = xy.min(0), xy.max(0)
    height, width = maximum - minimum + 1
    window_rows = ((xy[:, 0] - minimum[0]) * grid_rows) // height
    window_cols = ((xy[:, 1] - minimum[1]) * grid_cols) // width
    window_ids = window_rows * grid_cols + window_cols
    nonempty_windows, spatial_ids = np.unique(window_ids, return_inverse=True)
    def filtered_edges(ids, source, target):
        mask = ids[source] == ids[target]
        src = np.concatenate([source[mask], np.arange(len(xy))])
        dst = np.concatenate([target[mask], np.arange(len(xy))])
        order = np.lexsort((dst, src))
        return torch.from_numpy(np.stack([src[order], dst[order]]))
    semantic_edges = filtered_edges(semantic_ids, np.asarray(semantic_sources, dtype=np.int64),
                                    np.asarray(semantic_targets, dtype=np.int64))
    spatial_edges = filtered_edges(spatial_ids, source, target)
    result = {'semantic_graph_id': torch.from_numpy(semantic_ids),
              'spatial_graph_id': torch.from_numpy(spatial_ids.astype(np.int64)),
              'semantic_edge_index': semantic_edges, 'spatial_edge_index': spatial_edges,
              'semantic_edge_attr': edge_attributes(coords, semantic_edges),
              'spatial_edge_attr': edge_attributes(coords, spatial_edges)}
    metadata = {'semantic_max_nodes': semantic_max_nodes, 'grid_rows': grid_rows,
                'semantic_radius': semantic_radius, 'spatial_radius': 1,
                'semantic_neighbor_rule': 'same label, 0 < max(abs(delta_row),abs(delta_col)) <= semantic_radius',
                'semantic_gap_policy': 'radius edges may span missing or differently labeled intermediate positions; endpoints must exist and have the same label; no synthetic nodes',
                'grid_cols': grid_cols, 'initial_semantic_components': len(initial),
                'semantic_graph_count': len(final), 'spatial_graph_count': len(nonempty_windows),
                'retained_bbox_min_row_col': minimum.tolist(),
                'retained_bbox_max_row_col': maximum.tolist(),
                'nonempty_window_ids': nonempty_windows.tolist(), 'semantic_splits': splits,
                'self_loops': 'one explicit self-loop for every node in each branch',
                'edge_direction': 'source=edge_index[0], target=edge_index[1]',
                'edge_attr_definition': '[target_row-source_row, target_col-source_col, Euclidean grid distance]'}
    return result, metadata


def prune_small_graphs(graphs, metadata, min_nodes=2):
    """Filter each branch independently without changing global feature indices."""
    if not isinstance(min_nodes, int) or min_nodes < 1:
        raise ValueError('min_nodes must be a positive integer')
    result, info = dict(graphs), deepcopy(metadata)
    info['min_graph_nodes'] = min_nodes
    info['small_graph_filter'] = {}
    for branch in ['semantic', 'spatial']:
        original_ids = graphs[branch + '_graph_id']
        if (original_ids < 0).any():
            raise ValueError('Pruning expects the complete, unfiltered graph cache')
        sizes = torch.bincount(original_ids)
        kept = (sizes >= min_nodes).nonzero().flatten()
        remap = torch.full((len(sizes),), -1, dtype=torch.long)
        remap[kept] = torch.arange(len(kept))
        ids = remap[original_ids]
        edges = graphs[branch + '_edge_index']
        edge_mask = (ids[edges[0]] >= 0) & (ids[edges[1]] >= 0)
        result[branch + '_graph_id'] = ids
        result[branch + '_node_mask'] = ids >= 0
        result[branch + '_edge_index'] = edges[:, edge_mask]
        result[branch + '_edge_attr'] = graphs[branch + '_edge_attr'][edge_mask]
        info[branch + '_graph_count'] = len(kept)
        info['small_graph_filter'][branch] = {
            'candidate_graphs': len(sizes), 'kept_graphs': len(kept),
            'discarded_graphs': int((sizes < min_nodes).sum()),
            'discarded_nodes': int((ids < 0).sum()), 'kept_original_graph_ids': kept.tolist()}
        if branch == 'spatial':
            info['nonempty_window_ids'] = [metadata['nonempty_window_ids'][i] for i in kept.tolist()]
    info['self_loops'] = 'one explicit self-loop per retained branch node; discarded nodes have no edges or loops'
    info['discarded_node_contract'] = 'graph_id=-1 and node_mask=False in that branch; global feature/coordinate/label arrays remain aligned. Exclude these nodes before branch GNN and pooling using branch_inputs.'
    return result, info


def validate_graphs(coords, labels, graphs, metadata):
    n = len(coords)
    results = {}
    for branch in ['semantic', 'spatial']:
        ids, edges = graphs[branch + '_graph_id'], graphs[branch + '_edge_index']
        attrs = graphs[branch + '_edge_attr']
        assert ids.shape == (n,) and ids.dtype == torch.long
        assert torch.all(ids >= -1)
        active = ids >= 0
        if branch + '_node_mask' in graphs:
            assert torch.equal(graphs[branch + '_node_mask'], active)
        assert edges.ndim == 2 and edges.shape[0] == 2 and edges.dtype == torch.long
        if edges.numel():
            assert int(edges.min()) >= 0 and int(edges.max()) < n
        assert active[edges.flatten()].all()
        assert torch.equal(ids[edges[0]], ids[edges[1]])
        steps = coords[edges[1]] - coords[edges[0]]
        loops = edges[0] == edges[1]
        assert int(loops.sum()) == int(active.sum())
        assert torch.equal(edges[0, loops].sort().values, active.nonzero().flatten())
        distances = steps[~loops].abs().amax(1)
        radius = metadata.get('semantic_radius', 1) if branch == 'semantic' else 1
        assert torch.all((distances >= 1) & (distances <= radius))
        assert torch.equal(attrs, edge_attributes(coords, edges))
        assert not attrs[loops].any()
        # Every directed edge has its reverse, no cross-gap edges are created.
        pairs = set(zip(edges[0].tolist(), edges[1].tolist()))
        assert len(pairs) == edges.shape[1]
        assert all((j, i) in pairs for i, j in pairs)
        sizes = torch.bincount(ids[active])
        assert torch.all(sizes >= metadata.get('min_graph_nodes', 1))
        if branch == 'semantic':
            assert not len(sizes) or int(sizes.max()) <= metadata['semantic_max_nodes']
            assert torch.equal(labels[edges[0]], labels[edges[1]])
            adjacency = [[] for _ in range(n)]
            for i, j in pairs:
                if i != j: adjacency[i].append(j)
            assert len(components(active.nonzero().flatten().tolist(), adjacency)) == len(sizes)
            assert all(len(labels[ids == i].unique()) == 1 for i in range(len(sizes)))
        else:
            assert len(sizes) <= metadata['grid_rows'] * metadata['grid_cols']
        results[branch] = {'graphs': len(sizes), 'edges_including_self_loops': edges.shape[1],
                           'min_nodes': int(sizes.min()) if len(sizes) else 0, 'max_nodes': int(sizes.max()) if len(sizes) else 0,
                           'singleton_graphs': int((sizes == 1).sum()),
                           'retained_nodes': int(active.sum()), 'discarded_nodes': int((~active).sum()),
                           'cross_semantic_directed_edges': int((labels[edges[0]] != labels[edges[1]]).sum()),
                           'invariants': 'PASS'}
    return results


def branch_inputs(data, branch):
    """Compact a retained branch and remap global edges for GNN/pooling input."""
    if branch not in ('semantic', 'spatial'):
        raise ValueError('branch must be semantic or spatial')
    nodes = (data[branch + '_graph_id'] >= 0).nonzero().flatten()
    remap = torch.full((len(data['coords']),), -1, dtype=torch.long)
    remap[nodes] = torch.arange(len(nodes))
    edges = remap[data[branch + '_edge_index']]
    if (edges < 0).any():
        raise ValueError('Discarded node appears in edges')
    return {'features': data['features'][nodes], 'coords': data['coords'][nodes],
            'semantic_label': data['semantic_label'][nodes], 'original_node_index': nodes,
            'graph_id': data[branch + '_graph_id'][nodes], 'edge_index': edges,
            'edge_attr': data[branch + '_edge_attr']}
