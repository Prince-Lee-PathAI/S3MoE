"""S3MoE: semantic/spatial GATv2 encoders, expert-choice routing and global residual.

GATv2 uses incoming-edge attention with explicit self-loops; no PyG is required.
The parameter names match the complete dual-branch training implementation.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

BRANCHES = ('semantic', 'spatial')


def segment_softmax(logits, index, segments):
    """Stable softmax over each graph or incoming-neighbor set, per column."""
    if logits.shape[0] != index.numel():
        raise ValueError('Segment index must match logits')
    if not index.numel():
        return logits.clone()
    # The shift is numerically stabilizing and need not carry a gradient.
    work = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
    expanded = index.reshape(-1, *([1] * (work.ndim - 1))).expand_as(work)
    maxima = work.new_full((segments, *work.shape[1:]), -torch.inf)
    maxima.scatter_reduce_(0, expanded, work.detach(), reduce='amax', include_self=True)
    exp = (work - maxima[index]).exp()
    sums = work.new_zeros((segments, *work.shape[1:])).index_add(0, index, exp)
    return (exp / sums[index]).to(logits.dtype)


class GATv2Conv(nn.Module):
    """alpha(src->dst)=softmax_dst(a·LeakyReLU(Wsrc*xsrc+Wdst*xdst+We*e))."""

    def __init__(self, hidden_dim=256, heads=4, edge_dim=3, dropout=0.1):
        super().__init__()
        if hidden_dim <= 0 or heads <= 0 or hidden_dim % heads:
            raise ValueError('hidden_dim must be a positive multiple of heads')
        self.heads, self.channels = heads, hidden_dim // heads
        self.source = nn.Linear(hidden_dim, hidden_dim)
        self.target = nn.Linear(hidden_dim, hidden_dim)
        self.edge = nn.Linear(edge_dim, hidden_dim, bias=False)
        self.attention = nn.Parameter(torch.empty(heads, self.channels))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.dropout = nn.Dropout(dropout)
        for linear in (self.source, self.target, self.edge):
            nn.init.xavier_uniform_(linear.weight)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)
        nn.init.xavier_uniform_(self.attention)

    def forward(self, x, edge_index, edge_attr):
        n = len(x)
        if not n:
            return x
        source, target = edge_index
        values = self.source(x).reshape(n, self.heads, self.channels)
        queries = self.target(x).reshape(n, self.heads, self.channels)
        edge = self.edge(edge_attr).reshape(-1, self.heads, self.channels)
        joint = values[source] + queries[target] + edge
        logits = (F.leaky_relu(joint, negative_slope=.2) * self.attention).sum(-1)
        alpha = self.dropout(segment_softmax(logits, target, n))
        messages = values[source] * alpha.unsqueeze(-1)
        output = values.new_zeros((n, self.heads, self.channels)).index_add(0, target, messages)
        return output.reshape(n, -1) + self.bias


class GraphBlock(nn.Module):
    def __init__(self, hidden_dim, heads, dropout):
        super().__init__()
        self.conv = GATv2Conv(hidden_dim, heads, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edges, attrs):
        return self.norm(x + self.dropout(F.gelu(self.conv(x, edges, attrs))))


class GatedGraphPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.v, self.u = nn.Linear(hidden_dim, hidden_dim), nn.Linear(hidden_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, nodes, graph_id, num_graphs):
        logits = self.score(torch.tanh(self.v(nodes)) * torch.sigmoid(self.u(nodes))).squeeze(-1)
        weights = segment_softmax(logits, graph_id, num_graphs)
        pooled = nodes.new_zeros((num_graphs, nodes.shape[1])).index_add(
            0, graph_id, nodes * weights.unsqueeze(-1))
        return pooled, weights


class GraphBranchEncoder(nn.Module):
    def __init__(self, hidden_dim, layers, heads, dropout):
        super().__init__()
        self.layers = nn.ModuleList([GraphBlock(hidden_dim, heads, dropout) for _ in range(layers)])
        self.pool = GatedGraphPooling(hidden_dim)

    def forward(self, nodes, branch):
        for layer in self.layers:
            nodes = layer(nodes, branch['edge_index'], branch['edge_attr'])
        pooled, attention = self.pool(nodes, branch['graph_id'], branch['num_graphs'])
        return pooled, attention, nodes


@dataclass(frozen=True)
class GraphEncoderConfig:
    input_dim: int
    num_semantic_classes: int
    semantic_embedding_dim: int = 16
    hidden_dim: int = 256
    gnn_layers: int = 2
    gat_heads: int = 4
    branch_embedding_dim: int = 16
    dropout: float = 0.1


class DualBranchGraphEncoder(nn.Module):
    """Return branch graph tokens and ownership; no slide pooling or classifier."""

    def __init__(self, config: GraphEncoderConfig):
        super().__init__()
        if min(config.input_dim, config.num_semantic_classes, config.semantic_embedding_dim,
               config.hidden_dim, config.gnn_layers, config.gat_heads, config.branch_embedding_dim) <= 0:
            raise ValueError('Encoder dimensions and layer count must be positive')
        if config.hidden_dim % config.gat_heads or not 0 <= config.dropout < 1:
            raise ValueError('Invalid head count or dropout')
        self.config = config
        self.semantic_embedding = nn.Embedding(config.num_semantic_classes, config.semantic_embedding_dim)
        self.input_projection = nn.Sequential(
            nn.Linear(config.input_dim + config.semantic_embedding_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim), nn.GELU(), nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim))
        self.branches = nn.ModuleDict({b: GraphBranchEncoder(config.hidden_dim, config.gnn_layers,
                                       config.gat_heads, config.dropout) for b in ('semantic', 'spatial')})
        self.branch_embedding = nn.Embedding(2, config.branch_embedding_dim)
        token_width = config.hidden_dim + config.branch_embedding_dim + config.num_semantic_classes + 1
        self.token_projection = nn.Sequential(nn.Linear(token_width, config.hidden_dim), nn.GELU(),
                                               nn.Dropout(config.dropout),
                                               nn.Linear(config.hidden_dim, config.hidden_dim))

    def forward(self, batch, return_node_details=False):
        if len(batch['class_names']) != self.config.num_semantic_classes:
            raise ValueError('Batch semantic class count differs from encoder configuration')
        x = batch['features']
        if x.ndim != 2 or x.shape[1] != self.config.input_dim:
            raise ValueError('Batch feature width differs from encoder configuration')
        semantic = batch['semantic_label']
        projected = self.input_projection(torch.cat((x, self.semantic_embedding(semantic)), -1))
        output = {'num_slides': batch['num_slides'], 'branches': {}}
        for branch_index, name in enumerate(('semantic', 'spatial')):
            branch = batch['branches'][name]
            indices, ids, count = branch['node_index'], branch['graph_id'], branch['num_graphs']
            pooled, attention, nodes = self.branches[name](projected[indices], branch)
            sizes = torch.bincount(ids, minlength=count)
            semantic_counts = pooled.new_zeros((count, self.config.num_semantic_classes)).index_add(
                0, ids, F.one_hot(semantic[indices], self.config.num_semantic_classes).to(pooled.dtype))
            proportions = semantic_counts / sizes.clamp_min(1).unsqueeze(-1).to(pooled.dtype)
            branch_ids = torch.full((count,), branch_index, dtype=torch.long, device=x.device)
            token_input = torch.cat((pooled, self.branch_embedding(branch_ids), proportions,
                                     sizes.to(pooled.dtype).log1p().unsqueeze(-1)), -1)
            result = {'tokens': self.token_projection(token_input), 'pooled': pooled,
                      'semantic_proportions': proportions, 'node_counts': sizes,
                      'graph_slide_index': branch['graph_slide_index'],
                      'graph_local_id': branch['graph_local_id'], 'graph_ptr': branch['graph_ptr']}
            if return_node_details:
                result.update(node_attention=attention, node_embeddings=nodes,
                              node_index=indices, original_node_index=branch['original_node_index'])
            output['branches'][name] = result
        return output

    def configuration(self):
        return asdict(self.config)


@dataclass(frozen=True)
class GraphMoEConfig:
    hidden_dim: int = 256
    num_experts: int = 4
    capacity_factor: float = 1.0
    dense_router_warmup_epochs: int = 5
    dropout: float = 0.1
    routing_mode: str = 'expert_choice'
    routing_seed: int = 20261007


def branch_capacity(num_graphs, num_experts, capacity_factor):
    """Sparse capacity per expert, calculated independently for each branch."""
    if num_graphs == 0:
        return 0
    return min(num_graphs, max(1, math.ceil(capacity_factor * num_graphs / num_experts)))


class ResidualExpert(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, 2 * hidden_dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(2 * hidden_dim, hidden_dim),
                                 nn.Dropout(dropout))

    def forward(self, tokens):
        return tokens + self.mlp(tokens)


def selection_statistics(indices, num_graphs):
    """Unique coverage and pairwise Jaccard of expert selections; local indices."""
    experts, capacity = indices.shape
    masks = torch.zeros((experts, num_graphs), dtype=torch.bool, device=indices.device)
    masks.scatter_(1, indices, True)
    counts = masks.sum(0)
    intersection = masks.float() @ masks.float().T
    sizes = masks.sum(1)
    union = sizes[:, None] + sizes[None, :] - intersection
    return {'selection_counts': sizes, 'selected_unique': (counts > 0).sum(),
            'unselected_count': (counts == 0).sum(), 'token_selection_counts': counts,
            'pairwise_jaccard': intersection / union.clamp_min(1),
            'total_assignments': experts * capacity}


class GraphExpertChoiceMoE(nn.Module):
    """Shared experts, branch-specific per-slide capacity, and binary slide logit.

    epoch is zero-based. Training uses dense routing during warm-up, then
    per-expert top-k selection. Evaluation always uses sparse expert choice.
    Selected weights retain router probabilities without renormalization.
    """

    def __init__(self, config=GraphMoEConfig()):
        super().__init__()
        if config.hidden_dim <= 0 or config.num_experts <= 0:
            raise ValueError('Hidden dimension and expert count must be positive')
        if not math.isfinite(config.capacity_factor) or config.capacity_factor <= 0:
            raise ValueError('Capacity factor must be finite and positive')
        if config.dense_router_warmup_epochs < 0 or not 0 <= config.dropout < 1:
            raise ValueError('Invalid warmup or dropout')
        if config.routing_mode != 'expert_choice':
            raise ValueError('Unknown routing mode')
        if config.routing_seed < 0:
            raise ValueError('Routing seed must be nonnegative')
        self.config = config
        h = config.hidden_dim
        self.router = nn.Linear(h, config.num_experts)
        self.experts = nn.ModuleList([ResidualExpert(h, config.dropout) for _ in range(config.num_experts)])
        self.fusion = nn.Sequential(nn.Linear(2 * h, h), nn.GELU(), nn.Dropout(config.dropout),
                                    nn.Linear(h, h))
        self.expert_attention = nn.Sequential(nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1, bias=False))
        self.classifier = nn.Linear(h, 1)

    def _validate(self, encoded):
        slides = encoded['num_slides']
        if slides <= 0:
            raise ValueError('At least one WSI is required')
        pointers, prototype = {}, encoded['branches']['semantic']['tokens']
        for name in BRANCHES:
            branch = encoded['branches'][name]
            tokens, owners, ids = branch['tokens'], branch['graph_slide_index'], branch['graph_local_id']
            ptr = branch['graph_ptr']
            if tokens.ndim != 2 or tokens.shape[1] != self.config.hidden_dim:
                raise ValueError(f'{name}: wrong token width')
            if tokens.device != prototype.device or tokens.dtype != prototype.dtype:
                raise ValueError('Both branches must use the same device and dtype')
            if ptr.shape != (slides + 1,) or ptr.dtype != torch.long:
                raise ValueError(f'{name}: invalid graph pointers')
            values = ptr.tolist()
            if values[0] != 0 or values[-1] != len(tokens) or any(a > b for a, b in zip(values, values[1:])):
                raise ValueError(f'{name}: graph pointers do not partition tokens')
            if owners.shape != (len(tokens),) or owners.dtype != torch.long:
                raise ValueError(f'{name}: invalid WSI ownership')
            expected = torch.repeat_interleave(torch.arange(slides, device=owners.device), ptr.diff())
            if not torch.equal(owners, expected):
                raise ValueError(f'{name}: token ownership differs from graph pointers')
            if ids.shape != (len(tokens),) or ids.dtype != torch.long:
                raise ValueError(f'{name}: invalid graph IDs')
            pointers[name] = values
        return pointers

    def forward(self, encoded, *, epoch=0, return_routing=False, return_probabilities=False):
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError('epoch must be a nonnegative zero-based integer')
        pointers = self._validate(encoded)
        dense = self.training and epoch < self.config.dense_router_warmup_epochs
        e, h = self.config.num_experts, self.config.hidden_dim
        prototype = encoded['branches']['semantic']['tokens']
        scores = {}
        for name in BRANCHES:
            tokens = encoded['branches'][name]['tokens']
            if len(tokens):
                logits = self.router(tokens)
                work = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
                scores[name] = work.softmax(-1).to(logits.dtype)
            else:
                scores[name] = tokens.new_empty((0, e))
        embeddings, missing_masks, slide_info = [], [], []
        for slide in range(encoded['num_slides']):
            summaries, missing, branch_info = [], [], {}
            for name in BRANCHES:
                left, right = pointers[name][slide:slide + 2]
                tokens = encoded['branches'][name]['tokens'][left:right]
                weights = scores[name][left:right]
                m = right - left
                k = m if dense else branch_capacity(m, e, self.config.capacity_factor)
                missing.append(m == 0)
                if dense:
                    chosen = torch.arange(m, device=tokens.device).expand(e, m)
                else:
                    # Stable order resolves equal scores by the original token order.
                    chosen = torch.argsort(weights.T, dim=1, descending=True, stable=True)[:, :k]
                selected_weights = weights.T.gather(1, chosen)
                if k:
                    vectors = torch.stack([
                        (expert(tokens[chosen[i]]) * selected_weights[i, :, None]).sum(0) / k
                        for i, expert in enumerate(self.experts)])
                else:
                    vectors = prototype.new_zeros((e, h))
                summaries.append(vectors)
                if return_routing:
                    ids = encoded['branches'][name]['graph_local_id'][left:right]
                    branch_info[name] = {
                        'num_graphs': m, 'capacity': k,
                        'selected_token_indices': chosen + left,
                        'selected_graph_ids': ids[chosen], 'selected_weights': selected_weights,
                        'router_probabilities': weights, 'expert_branch_embeddings': vectors,
                        **selection_statistics(chosen, m)}
            fused = self.fusion(torch.cat(summaries, -1))
            # With no retained graphs, use a zero slide representation rather than
            # treating fusion biases as observed tissue evidence.
            if all(missing):
                fused = fused * 0
            attention = self.expert_attention(fused).squeeze(-1).softmax(0)
            embeddings.append((fused * attention[:, None]).sum(0))
            missing_masks.append(missing)
            if return_routing:
                slide_info.append({'slide_index': slide, 'branches': branch_info,
                                   'expert_embeddings': fused, 'expert_attention': attention})
        slide_embedding = torch.stack(embeddings)
        result = {'slide_embedding': slide_embedding,
                  'logits': self.classifier(slide_embedding).squeeze(-1),
                  'branch_missing_mask': torch.tensor(missing_masks, dtype=torch.bool, device=prototype.device)}
        if return_probabilities:
            result['probabilities'] = result['logits'].sigmoid()
        if return_routing:
            result['routing_info'] = {'mode': 'dense' if dense else 'sparse', 'epoch': epoch,
                                      'slides': slide_info}
        return result

    def configuration(self):
        return asdict(self.config)


class S3MoE(nn.Module):
    def __init__(self, encoder_config: GraphEncoderConfig, moe_config: GraphMoEConfig | None = None,
                 *, global_alpha_init=.1):
        super().__init__()
        moe_config = moe_config or GraphMoEConfig(hidden_dim=encoder_config.hidden_dim,
                                                 dropout=encoder_config.dropout)
        if encoder_config.hidden_dim != moe_config.hidden_dim:
            raise ValueError('Encoder and MoE hidden dimensions differ')
        self.encoder = DualBranchGraphEncoder(encoder_config)
        self.moe = GraphExpertChoiceMoE(moe_config)
        self.global_residual = True
        self.global_alpha_init = global_alpha_init
        if not 0 < global_alpha_init < 1:
            raise ValueError('Global residual alpha initialization must be in (0, 1)')
        hidden = encoder_config.hidden_dim
        # All raw visual patches participate, including discarded graph singletons.
        # This branch receives neither semantic labels nor graph assignments.
        self.global_projection = nn.Sequential(
            nn.Linear(encoder_config.input_dim, hidden), nn.LayerNorm(hidden),
            nn.GELU(), nn.Dropout(encoder_config.dropout))
        self.global_pool = GatedGraphPooling(hidden)
        self.residual_alpha_logit = nn.Parameter(torch.tensor(
            math.log(global_alpha_init / (1 - global_alpha_init))))
        self.fusion_norm = nn.LayerNorm(hidden)
    def forward(self, batch, *, epoch=0, return_routing=False, return_probabilities=False,
                return_node_attention=False, return_graph_tokens=False):
        encoded = self.encoder(batch, return_node_details=return_node_attention)
        encoded['slide_ids'] = batch['slide_ids']
        result = self.moe(encoded, epoch=epoch, return_routing=return_routing,
                          return_probabilities=return_probabilities)
        global_embedding, global_attention = self.global_pool(
            self.global_projection(batch['features']), batch['node_slide_index'], batch['num_slides'])
        alpha = self.residual_alpha_logit.sigmoid()
        result['moe_slide_embedding'] = result['slide_embedding']
        result['global_slide_embedding'] = global_embedding
        result['residual_alpha'] = alpha
        result['slide_embedding'] = self.fusion_norm(global_embedding + alpha * result['slide_embedding'])
        # Reuse the existing slide classifier; no unused classification head.
        result['logits'] = self.moe.classifier(result['slide_embedding']).squeeze(-1)
        if return_probabilities:
            result['probabilities'] = result['logits'].sigmoid()
        if return_node_attention:
            result['global_node_attention'] = global_attention
        if return_routing:
            result['routing_info']['slide_ids'] = batch['slide_ids']
        if return_node_attention:
            result['node_attention'] = {
                name: {'weights': branch['node_attention'], 'node_index': branch['node_index'],
                       'original_node_index': branch['original_node_index'],
                       'node_slide_index': batch['node_slide_index'][branch['node_index']]}
                for name, branch in encoded['branches'].items()}
        if return_graph_tokens:
            result['graph_tokens'] = encoded
        return result

    def configuration(self):
        return {'encoder': self.encoder.configuration(), 'moe': self.moe.configuration(),
                'global_residual': {'enabled': True, 'alpha_init': self.global_alpha_init}}

    @classmethod
    def from_configuration(cls, config):
        if (not config.get('global_residual', {}).get('enabled', False)
                or config.get('graph_output_norm', False) or config.get('disabled_branch') is not None):
            raise ValueError('This release supports the complete S3MoE architecture only')
        return cls(GraphEncoderConfig(**config['encoder']), GraphMoEConfig(**config['moe']),
                   global_alpha_init=config['global_residual'].get('alpha_init', .1))
