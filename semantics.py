"""Frozen CPath-CLIP semantic assignment and portable per-WSI graph export."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from graphs import build_graphs, prune_small_graphs, validate_graphs


class CPathTextEncoder(nn.Module):
    def __init__(self, config_path: Path, checkpoint: Path):
        super().__init__()
        from transformers import AutoConfig
        from llm2vec.models import Qwen2BiModel

        config = AutoConfig.from_pretrained(str(config_path), local_files_only=True)
        config._attn_implementation = 'sdpa'
        config.use_cache = False
        state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
        if 'state_dict' in state:
            state = state['state_dict']
        state = {k.removeprefix('module.'): v for k, v in state.items()}
        projection = state['proj.weight']
        self.cls_dim = int(state['visual2.cls_token'].shape[-1])
        clip_dim = int(state['visual.proj'].shape[-1])
        if projection.shape[0] != clip_dim + 2 * self.cls_dim:
            raise ValueError('Checkpoint does not match CLIP + Virchow2 CLS + token mean layout')
        with torch.device('meta'):
            self.text = Qwen2BiModel(config)
            self.proj = nn.Linear(int(projection.shape[1]), int(projection.shape[0]))
        text_state = {k[len('text.'):]: v for k, v in state.items() if k.startswith('text.')}
        proj_state = {k[len('proj.'):]: v for k, v in state.items() if k.startswith('proj.')}
        self.text.load_state_dict(text_state, strict=True, assign=True)
        self.proj.load_state_dict(proj_state, strict=True, assign=True)
        # Rotary frequency/cache buffers are nonpersistent and thus absent
        # from the checkpoint. Materialize them after meta-device loading.
        for layer in self.text.layers:
            rotary = layer.self_attn.rotary_emb
            layer.self_attn.rotary_emb = type(rotary)(
                rotary.dim, max_position_embeddings=rotary.max_position_embeddings,
                base=rotary.base, device='cpu')
        if any(buffer.is_meta for buffer in self.buffers()):
            raise RuntimeError('Unmaterialized buffer after loading the text encoder')
        self.load_info = {'text_keys': len(text_state), 'projection_keys': len(proj_state),
                          'strict_load': True, 'clip_dim': clip_dim, 'cls_dim': self.cls_dim,
                          'total_dim': int(projection.shape[0])}
        self.requires_grad_(False).eval()

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        mask = batch['attention_mask']
        # Explicit noncausal key-padding mask: this installed transformers
        # version otherwise constructs a causal mask for padded Qwen batches.
        # Qwen2BiModel is intended to attend bidirectionally.
        length = mask.shape[1]
        additive_mask = torch.zeros((len(mask), 1, length, length),
                                   device=mask.device, dtype=self.text.embed_tokens.weight.dtype)
        additive_mask.masked_fill_(~mask[:, None, None, :].bool(),
                                  torch.finfo(additive_mask.dtype).min)
        hidden = self.text(input_ids=batch['input_ids'], attention_mask=additive_mask,
                           use_cache=False).last_hidden_state
        weights = mask.unsqueeze(-1).float()
        pooled = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)
        full = self.proj(pooled)
        text_cls = full[..., -2 * self.cls_dim:][..., :self.cls_dim]
        return F.normalize(text_cls.float(), dim=-1)


@torch.inference_mode()
def encode_prompts(args, cfg: dict) -> tuple[dict, dict, dict]:
    from transformers import AutoTokenizer

    model = CPathTextEncoder(args.tokenizer_dir, args.checkpoint).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_dir), local_files_only=True,
                                            padding_side='right')
    names = cfg['prototype_names'] if 'prototype_names' in cfg else list(cfg['descriptions'])
    prompt_records = []
    if 'prompt_records' in cfg:
        for record in cfg['prompt_records']:
            if (record.get('prototype') not in names or not isinstance(record.get('text'), str)
                    or not record['text'].strip()):
                raise ValueError(f'Invalid explicit prompt record: {record}')
            prompt_records.append((record['prototype'], record['text']))
        if set(name for name, _ in prompt_records) != set(names):
            raise ValueError('Explicit prompts must cover every prototype')
    else:
        for name in names:
            descriptions = cfg['descriptions'][name]
            if isinstance(descriptions, str):
                descriptions = [descriptions]
            for description in descriptions:
                for template in cfg['templates']:
                    if description.lstrip().startswith('It '):
                        # User-provided complete sentences follow a neutral image
                        # context rather than becoming "An image of It ...".
                        context = cfg.get('sentence_description_context', 'a histopathology tissue region')
                        prompt = template.format(context) + ' ' + description
                    else:
                        prompt = template.format(description)
                    prompt_records.append((name, prompt))
    vectors = {name: [] for name in names}
    for start in range(0, len(prompt_records), args.text_batch_size):
        chunk = prompt_records[start:start + args.text_batch_size]
        batch = tokenizer([text for _, text in chunk], return_tensors='pt', padding=True,
                          truncation=True, max_length=512)
        batch = {k: v.to(args.device) for k, v in batch.items()}
        output = model(batch).cpu()
        if not torch.isfinite(output).all():
            raise ValueError('Text encoder produced nonfinite vectors')
        for (name, _), vector in zip(chunk, output):
            vectors[name].append(vector)
        print(f'Text prompts: {min(start + len(chunk), len(prompt_records))}/{len(prompt_records)}', flush=True)
    prototypes = {name: F.normalize(torch.stack(values).mean(0), dim=0)
                  for name, values in vectors.items()}
    # Verify padding/batching invariance. Also diagnose the native path:
    # transformers 4.44 constructs causal attention despite this installed
    # llm2vec model's is_causal=False, so native equality is not required.
    one = tokenizer([prompt_records[0][1]], return_tensors='pt', padding=False, max_length=512, truncation=True)
    one = {k: v.to(args.device) for k, v in one.items()}
    hidden = model.text(**one, use_cache=False).last_hidden_state
    mask = one['attention_mask'][..., None].float()
    pooled = (hidden * mask).sum(1) / mask.sum(1)
    native = model.proj(pooled)[..., -2 * model.cls_dim:][..., :model.cls_dim]
    native = F.normalize(native.float(), dim=-1)
    explicit = model(one)
    comparison = {'native_unpadded_vs_bidirectional_max_abs_error': float((native-explicit).abs().max()),
                  'batched_vs_single_max_abs_error': float((vectors[prompt_records[0][0]][0]-explicit[0].cpu()).abs().max())}
    if comparison['batched_vs_single_max_abs_error'] > 1e-4:
        raise AssertionError(f'Text encoding equivalence check failed: {comparison}')
    info = {**model.load_info, **comparison,
            'attention': 'bidirectional SDPA with explicit noncausal padding mask',
            'compatibility_note': 'Native transformers 4.44 Qwen attention can remain causal despite Qwen2BiModel; explicit bidirectional mask enforces intended text-encoder architecture',
            'pooling': 'attention-mask-weighted mean of last_hidden_state, then proj',
            'text_slice': 'last 2560 dimensions, then first 1280; normalize after slicing'}
    return prototypes, {'records': [{'prototype': name, 'text': text} for name,text in prompt_records]}, info


def classify_ovarian(features: torch.Tensor, prototypes: dict, cfg: dict) -> dict:
    raw = {name: features @ vector for name, vector in prototypes.items()}
    raw['other'] = torch.stack([raw[name] for name in cfg['other_components']], dim=1).amax(1)
    coarse = torch.stack([raw[name] for name in cfg['coarse_groups']], dim=1)
    fine = torch.stack([raw[name] for name in cfg['fine_groups']], dim=1)
    coarse_top = coarse.topk(2, dim=1)
    fine_top = fine.topk(2, dim=1)
    class_id = {item['name']: item['id'] for item in cfg['classes']}
    predictions, coarse_winners = [], []
    for i in range(len(features)):
        winner = cfg['coarse_groups'][int(coarse_top.indices[i, 0])]
        coarse_winners.append(winner)
        if winner == 'set_tumor':
            # Inadequate SET support goes to Other, never to the SET fallback.
            if float(raw['set_tumor'][i]) < cfg['set_min_cosine']:
                final = 'other'
            elif float(fine_top.values[i, 0] - fine_top.values[i, 1]) < cfg['fine_margin_threshold']:
                final = cfg['fallback_name']
            else:
                final = cfg['fine_groups'][int(fine_top.indices[i, 0])]
        else:
            final = winner
        predictions.append(class_id[final])
    return {'semantic_label': torch.tensor(predictions), 'raw_cosines': raw,
            'coarse_cosines': coarse, 'fine_cosines': fine,
            'coarse_softmax_uncalibrated': (coarse * cfg['display_softmax_scale']).softmax(1),
            'coarse_winners': coarse_winners,
            'coarse_margin': coarse_top.values[:, 0] - coarse_top.values[:, 1],
            'fine_margin': fine_top.values[:, 0] - fine_top.values[:, 1]}


def classify_breast(features, prototypes, cfg):
    """Five-class cosine argmax; benign/background prototypes compete via their max."""
    names = [c['name'] for c in cfg['classes']]
    raw = {name: features @ prototypes[name] for name in cfg['prototype_names']}
    raw['other'] = torch.stack([raw[name] for name in cfg['other_components']], dim=1).amax(1)
    scores = torch.stack([raw[name] for name in names], dim=1)
    return {'semantic_label': scores.argmax(1), 'class_cosines': scores}


def assign_semantics(features, prototypes, cfg, stain=None):
    """Normalize a copy for cosine assignment; the exported visual features stay raw."""
    if features.ndim != 2 or not len(features) or not torch.isfinite(features).all():
        raise ValueError('Expected finite, nonempty features[N,D]')
    if (features.norm(dim=1) == 0).any():
        raise ValueError('Zero feature vectors cannot be classified by cosine similarity')
    features = F.normalize(features.float(), dim=1)
    if 'stain_records' in cfg:
        if stain not in cfg['stain_records']:
            raise ValueError('Breast assignment requires an explicit HE or IHC stain')
        prototypes = {name: prototypes[f'{stain}::{name}'] for name in cfg['prototype_names']}
    for vector in prototypes.values():
        if vector.shape != (features.shape[1],) or not torch.isfinite(vector).all():
            raise ValueError('Text prototype dimension/value mismatch')
        if not torch.allclose(vector.float().norm(), torch.tensor(1.), atol=1e-5):
            raise ValueError('Text prototypes must be L2-normalized')
    if cfg['assignment'] == 'flat_argmax_cosine':
        return classify_breast(features, prototypes, cfg)
    if cfg['assignment'] == 'hierarchical_cosine':
        return classify_ovarian(features, prototypes, cfg)
    raise ValueError('Unknown semantic assignment rule')


def load_features(path, layout='auto'):
    x = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if not isinstance(x, torch.Tensor) or x.ndim != 2 or not x.is_floating_point():
        raise ValueError('Features must be a floating Tensor[N,D]')
    layouts = {'virchow2_cls': (1280, 0), 'virchow2_cls_mean': (2560, 0),
               'clip_virchow2': (3328, 768)}
    if layout == 'auto':
        layout = {width: name for name, (width, _) in layouts.items()}.get(x.shape[1])
    if layout not in layouts or x.shape[1] != layouts[layout][0]:
        raise ValueError('Supported input widths: 1280, 2560, 3328')
    start = layouts[layout][1]
    return x[:, start:start + 1280].clone()


def load_coords(args, count):
    if args.coords:
        coords = torch.load(args.coords, map_location='cpu', weights_only=True)
        if not isinstance(coords, torch.Tensor) or coords.dtype != torch.long:
            raise ValueError('Coordinates must be an int64 Tensor[N,2], in feature-row order')
        patch_ids = [f'patch_{i}' for i in range(count)]
    else:
        # This convention matches the original extractor. Use --coords for other orders.
        import re
        paths = sorted(args.patch_dir.glob('*.jpg'), key=lambda p: [
            int(v) if v.isdigit() else v.lower() for v in re.split(r'(\d+)', p.name)])
        if len(paths) != count:
            raise ValueError('Patch filename count differs from the number of feature rows')
        positions, grids, instances = [], set(), set()
        for path in paths:
            match = re.fullmatch(r'(\d+)_(\d+)_(\d+)_(\d+)x(\d+)\.jpg', path.name)
            if not match:
                raise ValueError('Patch names must follow instance_row_col_heightxwidth.jpg')
            instance, row, col, height, width = map(int, match.groups())
            if not (0 <= row < height and 0 <= col < width):
                raise ValueError('Patch coordinates lie outside the filename grid')
            positions.append([row, col]); grids.add((height, width)); instances.add(instance)
        if len(grids) != 1 or len(instances) != 1:
            raise ValueError('Patch filenames contain inconsistent grid or instance metadata')
        coords, patch_ids = torch.tensor(positions, dtype=torch.long), [p.name for p in paths]
    if coords.shape != (count, 2) or (coords < 0).any() or len(coords.unique(dim=0)) != count:
        raise ValueError('Coordinates must be unique, nonnegative, and aligned with features')
    return coords, patch_ids


def config_digest(cfg):
    import hashlib
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    encode = sub.add_parser('encode', help='Encode fixed text prototypes once per semantic config')
    encode.add_argument('--config', type=Path, required=True)
    encode.add_argument('--checkpoint', type=Path, required=True, help='Merged CPath-CLIP checkpoint')
    encode.add_argument('--tokenizer-dir', type=Path, required=True, help='Qwen encoder config/tokenizer')
    encode.add_argument('--device', default='cpu')
    encode.add_argument('--text-batch-size', type=int, default=2)
    encode.add_argument('--output', type=Path, required=True)
    prepare = sub.add_parser('prepare', help='Assign one WSI and export its two graph branches')
    prepare.add_argument('--config', type=Path, required=True)
    prepare.add_argument('--prototypes', type=Path, required=True)
    prepare.add_argument('--features', type=Path, required=True)
    prepare.add_argument('--feature-layout', default='auto',
                         choices=['auto', 'virchow2_cls', 'virchow2_cls_mean', 'clip_virchow2'])
    coords = prepare.add_mutually_exclusive_group(required=True)
    coords.add_argument('--coords', type=Path, help='Int64 Tensor[N,2] in original feature-row order')
    coords.add_argument('--patch-dir', type=Path, help='Naturally sorted coordinate-bearing JPG filenames')
    prepare.add_argument('--slide-id', required=True)
    prepare.add_argument('--stain', choices=['HE', 'IHC'])
    prepare.add_argument('--semantic-radius', type=int, default=2)
    prepare.add_argument('--semantic-max-nodes', type=int, default=256)
    prepare.add_argument('--grid-rows', type=int, default=8)
    prepare.add_argument('--grid-cols', type=int, default=8)
    prepare.add_argument('--min-graph-nodes', type=int, default=2)
    prepare.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new file')
    cfg = json.loads(args.config.read_text())
    if [c['id'] for c in cfg['classes']] != list(range(len(cfg['classes']))):
        raise ValueError('Semantic class IDs must be contiguous from zero')
    if args.command == 'encode':
        if args.text_batch_size < 1:
            parser.error('text-batch-size must be positive')
        text_cfg = cfg
        if 'stain_records' in cfg:
            text_cfg = {'prototype_names': [f'{s}::{n}' for s in cfg['stain_records']
                                            for n in cfg['prototype_names']],
                        'prompt_records': [{'prototype': f'{s}::{r["prototype"]}', 'text': r['text']}
                                           for s, rows in cfg['stain_records'].items() for r in rows]}
        vectors, _, info = encode_prompts(SimpleNamespace(**vars(args)), text_cfg)
        artifact = {'prototypes': vectors, 'config_digest': config_digest(cfg), 'text_encoder': info}
    else:
        text = torch.load(args.prototypes, map_location='cpu', weights_only=True)
        if text['config_digest'] != config_digest(cfg):
            raise ValueError('Text prototypes were generated with a different config')
        features = load_features(args.features, args.feature_layout)
        coords, patch_ids = load_coords(args, len(features))
        predictions = assign_semantics(features, text['prototypes'], cfg, args.stain)
        labels = predictions['semantic_label']
        graphs, metadata = build_graphs(coords, labels, args.semantic_max_nodes,
                                        args.grid_rows, args.grid_cols, args.semantic_radius)
        graphs, metadata = prune_small_graphs(graphs, metadata, args.min_graph_nodes)
        validation = validate_graphs(coords, labels, graphs, metadata)
        artifact = {'slide_id': args.slide_id, 'features': features, 'coords': coords,
                    'feature_row': torch.arange(len(features)), 'patch_id': patch_ids,
                    'semantic_label': labels, **graphs,
                    'metadata': {'classes': cfg['classes'], 'graphs': metadata,
                                 'graph_validation': validation, 'stain': args.stain,
                                 'semantic_config_digest': config_digest(cfg)}}
        print(json.dumps(validation, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output)


if __name__ == '__main__':
    main()
