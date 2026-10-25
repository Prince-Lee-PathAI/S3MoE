"""Training, validation-only selection and resumable epoch checkpoints."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from model import S3MoE, GraphEncoderConfig, GraphMoEConfig
from data import WSIGraphDataset, collate_wsi_graphs, move_graph_batch

LOGGER = logging.getLogger('wsi_training')
METRIC_PROTOCOL = 'logit_ranking_float64_probability_v2'


def atomic_write(path, value, *, tensor=False):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    if tensor:
        torch.save(value, temporary)
    else:
        temporary.write_text(value)
    os.replace(temporary, path)


def write_json(path, value):
    atomic_write(path, json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    if not rows:
        raise ValueError('Cannot export empty prediction/history table')
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(path, stream.getvalue())


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_protocol(directory):
    """Verify fixed test membership, disjoint splits and five-fold coverage."""
    directory = Path(directory)
    folds, validation_sets, pool_reference, test_reference, identities = [], [], None, None, {}
    for fold in range(1, 6):
        path = directory / f'fold_{fold}_cache_manifest.csv'
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        groups = {name: {} for name in ['train', 'val', 'test']}
        for row in rows:
            identity = (row['class_name'], row.get('slide_id') or row['case_name'])
            split = row['split']
            if split not in groups or row['status'] != 'PASS' or int(row['label']) not in (0, 1):
                raise ValueError(f'Invalid fold record: {row}')
            if any(identity in group for group in groups.values()):
                raise ValueError(f'Duplicate WSI in fold {fold}: {identity}')
            cache = Path(row['cache_path']).expanduser()
            cache = cache.resolve() if cache.is_absolute() else (directory / cache).resolve()
            if not cache.is_file():
                raise FileNotFoundError(cache)
            signature = (int(row['label']), str(cache))
            if identity in identities and identities[identity] != signature:
                raise ValueError(f'Inconsistent WSI label/cache across folds: {identity}')
            identities[identity] = signature
            groups[split][identity] = signature
        for split, group in groups.items():
            if not group or {v[0] for v in group.values()} != {0, 1}:
                raise ValueError(f'Fold {fold} {split} must contain both labels')
        pool = groups['train'] | groups['val']
        if pool_reference is not None and (pool != pool_reference or groups['test'] != test_reference):
            raise ValueError('Training pool or fixed test set changes between folds')
        pool_reference, test_reference = pool, groups['test']
        validation_sets.append(set(groups['val']))
        folds.append({'fold': fold, 'counts': {name: len(group) for name, group in groups.items()},
                      'sha256': file_sha256(path)})
    all_validation = [identity for group in validation_sets for identity in group]
    if len(all_validation) != len(set(all_validation)) or set(all_validation) != set(pool_reference):
        raise ValueError('Validation folds must partition the training pool exactly once')
    return {'status': 'PASS', 'training_pool': len(pool_reference), 'fixed_test': len(test_reference), 'folds': folds}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(_worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def capture_rng():
    state = np.random.get_state()
    return {'python': random.getstate(), 'numpy': [state[0], state[1].tolist(), state[2], state[3], state[4]],
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] and torch.cuda.is_available():
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA device count changed; cannot restore CUDA random states exactly')
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def binary_metrics(labels, probabilities, threshold=.5, *, ranking_scores=None):
    labels, probabilities = np.asarray(labels, dtype=int), np.asarray(probabilities, dtype=float)
    if labels.size == 0 or labels.shape != probabilities.shape or not np.isfinite(probabilities).all():
        raise ValueError('Invalid binary predictions')
    scores = probabilities if ranking_scores is None else np.asarray(ranking_scores, dtype=float)
    if scores.shape != labels.shape or not np.isfinite(scores).all():
        raise ValueError('Invalid ranking scores')
    predicted = probabilities >= threshold
    both = len(np.unique(labels)) == 2
    return {'auroc': float(roc_auc_score(labels, scores)) if both else None,
            'auprc': float(average_precision_score(labels, scores)) if both else None,
            'balanced_accuracy': float(balanced_accuracy_score(labels, predicted)) if both else None,
            'accuracy': float(np.mean(labels == predicted)), 'threshold': float(threshold)}


def select_validation_threshold(labels, probabilities):
    """Maximize balanced accuracy on validation only; ties prefer proximity to .5."""
    y, p = np.asarray(labels, dtype=int), np.asarray(probabilities, dtype=float)
    if set(y.tolist()) != {0, 1} or not np.isfinite(p).all():
        raise ValueError('Threshold selection requires finite validation scores and both labels')
    candidates = np.unique(np.r_[0., .5, p, np.nextafter(p.max(), np.inf)])
    ranked = []
    for threshold in candidates:
        prediction = p >= threshold
        score = .5 * (prediction[y == 1].mean() + (~prediction[y == 0]).mean())
        ranked.append((-float(score), abs(float(threshold) - .5), float(threshold)))
    return min(ranked)[2]


@dataclass
class AUROCEarlyStopping:
    patience: int = 20
    min_delta: float = 0.
    best_score: float = -1.
    best_epoch: int = -1
    patience_anchor: float = -1.
    bad_epochs: int = 0

    def update(self, score, epoch):
        if not math.isfinite(score):
            raise ValueError('Validation AUROC is undefined or nonfinite')
        improved = score > self.best_score
        if improved:
            self.best_score, self.best_epoch = float(score), epoch
        # Save every actual maximum; min_delta controls patience reset only.
        if self.patience_anchor < 0 or score > self.patience_anchor + self.min_delta:
            self.patience_anchor, self.bad_epochs = float(score), 0
        else:
            self.bad_epochs += 1
        return improved

    @property
    def stopped(self):
        return self.bad_epochs >= self.patience


def train_epoch(model, loader, optimizer, *, epoch, device, accumulation_steps=1,
                gradient_clip=5., pos_weight=None, log_every=10, label_smoothing=0.):
    if not 0 <= label_smoothing < 1:
        raise ValueError('label_smoothing must be in [0, 1)')
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss, objective_sum, total_samples, bucket_samples, steps = 0., 0., 0, 0, 0
    labels, probabilities, logits = [], [], []
    started = time.monotonic()
    for index, batch in enumerate(loader):
        batch = move_graph_batch(batch, device)
        output = model(batch, epoch=epoch)
        targets = batch['labels'] * (1 - label_smoothing) + .5 * label_smoothing
        loss = F.binary_cross_entropy_with_logits(output['logits'], targets,
                                                  reduction='sum', pos_weight=pos_weight)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite training loss: {batch["slide_ids"]}')
        loss.backward()
        count = batch['labels'].numel()
        raw_loss = loss.detach() if label_smoothing == 0 else F.binary_cross_entropy_with_logits(
            output['logits'].detach(), batch['labels'], reduction='sum', pos_weight=pos_weight)
        total_loss += float(raw_loss)
        objective_sum += float(loss.detach())
        total_samples += count
        bucket_samples += count
        labels.extend(batch['labels'].detach().cpu().tolist())
        values = output['logits'].detach().cpu()
        logits.extend(values.tolist())
        probabilities.extend(values.double().sigmoid().tolist())
        if (index + 1) % accumulation_steps == 0 or index + 1 == len(loader):
            # Average by actual WSI count, including incomplete final buckets.
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(bucket_samples)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip or float('inf'),
                                                  error_if_nonfinite=True)
            if not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite gradient norm')
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            bucket_samples, steps = 0, steps + 1
        if (index + 1) % log_every == 0 or index + 1 == len(loader):
            LOGGER.info('  train epoch=%d batches=%d/%d WSIs=%d loss=%.5f elapsed=%.1fs',
                        epoch + 1, index + 1, len(loader), total_samples, total_loss / total_samples,
                        time.monotonic() - started)
    return {'loss': total_loss / total_samples, 'objective_loss': objective_sum / total_samples,
            **binary_metrics(labels, probabilities, ranking_scores=logits), 'optimizer_steps': steps}


@torch.no_grad()
def evaluate(model, loader, *, device, threshold=None):
    model.eval()  # Follow the saved routing mode; warm-up never changes evaluation.
    rows, loss_sum = [], 0.
    for batch in loader:
        batch = move_graph_batch(batch, device)
        output = model(batch)
        loss = F.binary_cross_entropy_with_logits(output['logits'], batch['labels'], reduction='sum')
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite validation/test loss')
        loss_sum += float(loss)
        for slide_id, path, label, logit, probability in zip(
                batch['slide_ids'], batch['cache_paths'], batch['labels'].cpu().tolist(),
                output['logits'].cpu().tolist(), output['logits'].cpu().double().sigmoid().tolist()):
            rows.append({'slide_id': slide_id, 'cache_path': path, 'label': int(label),
                         'logit': logit, 'probability': probability})
    labels, probabilities = [r['label'] for r in rows], [r['probability'] for r in rows]
    if threshold is None:
        threshold = select_validation_threshold(labels, probabilities)
    for row in rows:
        row.update(threshold=threshold, prediction=int(row['probability'] >= threshold))
    return {'loss': loss_sum / len(rows),
            **binary_metrics(labels, probabilities, threshold, ranking_scores=[r['logit'] for r in rows])}, rows


def build_model(config, input_dim, classes):
    encoder = GraphEncoderConfig(input_dim, classes, semantic_embedding_dim=config['semantic_embedding_dim'],
                                 hidden_dim=config['hidden_dim'], gnn_layers=config['gnn_layers'],
                                 gat_heads=config['gat_heads'], branch_embedding_dim=config['branch_embedding_dim'],
                                 dropout=config['dropout'])
    moe = GraphMoEConfig(hidden_dim=config['hidden_dim'], num_experts=config['num_experts'],
                        capacity_factor=config['capacity_factor'],
                        dense_router_warmup_epochs=config['warmup_epochs'], dropout=config['dropout'])
    return S3MoE(encoder, moe, global_alpha_init=config['global_alpha_init'])


def balanced_smoke_subset(dataset, per_class):
    indices = []
    for label in [0, 1]:
        matching = [i for i, row in enumerate(dataset.records) if int(row['label']) == label]
        indices.extend(sorted(matching, key=lambda i: dataset.records[i]['slide_id'])[:per_class])
    return Subset(dataset, indices)


def fold_datasets(manifest, smoke=False, *, cache_graph_samples=False):
    train, val = (WSIGraphDataset(manifest, split, cache_samples=cache_graph_samples)
                  for split in ['train', 'val'])
    if train.class_names != val.class_names or train.input_dim != val.input_dim:
        raise ValueError('Training and validation feature/class mappings differ')
    return train, val, balanced_smoke_subset(train, 2) if smoke else train, balanced_smoke_subset(val, 1) if smoke else val


def best_export(checkpoint, fold, directory):
    best = checkpoint['best']
    atomic_write(directory / 'best.pt', {'format_version': 1, 'fold': fold,
                 'metric_protocol': METRIC_PROTOCOL,
                 'model_config': checkpoint['model_config'], 'class_names': checkpoint['class_names'],
                 'model_state': best['model_state'], 'epoch': best['epoch'],
                 'validation_metrics': best['metrics'], 'threshold': best['metrics']['threshold'],
                 'run_signature': checkpoint['run_signature']}, tensor=True)
    write_csv(directory / 'best_validation_predictions.csv', best['predictions'])
    write_csv(directory / 'history.csv', checkpoint['history'])


def fit_fold(config, fold, manifest, directory, signature, *, resume=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=resume)
    set_seed(config['seed'] + fold)
    train_base, val_base, train_data, val_data = fold_datasets(manifest, config['smoke_test'],
        cache_graph_samples=config.get('cache_graph_samples', False))
    model = build_model(config,
                        train_base.input_dim, train_base.num_semantic_classes).to(config['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    generator = torch.Generator().manual_seed(config['seed'] + fold)
    loader_options = dict(batch_size=config['batch_size'], num_workers=config['num_workers'],
                          collate_fn=collate_wsi_graphs,
                          worker_init_fn=seed_worker)
    train_loader = DataLoader(train_data, shuffle=True, generator=generator, **loader_options)
    val_loader = DataLoader(val_data, shuffle=False,
                            generator=torch.Generator().manual_seed(config['seed'] + 100 + fold), **loader_options)
    stopping = AUROCEarlyStopping(config['patience'], config['min_delta'])
    start, history, best = 0, [], None
    checkpoint_path = directory / 'last.pt'
    if resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if checkpoint.get('metric_protocol') != METRIC_PROTOCOL:
            raise ValueError('Legacy probability-ranked checkpoints require a fresh training run')
        if checkpoint['run_signature'] != signature or checkpoint['fold'] != fold:
            raise ValueError('Checkpoint protocol/config signature differs')
        if checkpoint['model_config'] != model.configuration() or checkpoint['class_names'] != train_base.class_names:
            raise ValueError('Checkpoint feature/class configuration differs')
        model.load_state_dict(checkpoint['model_state'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        stopping = AUROCEarlyStopping(**checkpoint['early_stopping'])
        history, best, start = checkpoint['history'], checkpoint['best'], checkpoint['epoch'] + 1
        generator.set_state(checkpoint['loader_generator_state'])
        restore_rng(checkpoint['rng_state'])
        best_export(checkpoint, fold, directory)  # Repair exports after an interrupted atomic commit.
        LOGGER.info('Fold %d resumed after epoch %d, best AUROC=%.5f', fold, start, stopping.best_score)
    train_labels = [int(train_base.records[i]['label']) for i in train_data.indices] if isinstance(train_data, Subset) else [int(r['label']) for r in train_base.records]
    positive_weight = None
    if config['pos_weight'] == 'balanced':
        positive_weight = torch.tensor(train_labels.count(0) / train_labels.count(1), device=config['device'])
    LOGGER.info('Fold %d train=%d val=%d input=%d classes=%d parameters=%s', fold, len(train_data), len(val_data),
                train_base.input_dim, train_base.num_semantic_classes, f'{sum(p.numel() for p in model.parameters()):,}')
    steps = sum(row['optimizer_steps'] for row in history)
    for epoch in range(start, config['max_epochs']):
        if stopping.stopped:
            break
        started = time.monotonic()
        training = train_epoch(model, train_loader, optimizer, epoch=epoch, device=config['device'],
                                accumulation_steps=config['accumulation_steps'], gradient_clip=config['gradient_clip'],
                                pos_weight=positive_weight, log_every=config['log_every'],
                                label_smoothing=config.get('label_smoothing', 0.))
        validation, predictions = evaluate(model, val_loader, device=config['device'],
                                            threshold=config.get('classification_threshold'))
        improved = stopping.update(validation['auroc'], epoch)
        if improved:
            best = {'epoch': epoch, 'metrics': validation, 'predictions': predictions,
                    'model_state': {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}}
        steps += training['optimizer_steps']
        mode = 'dense' if epoch < config['warmup_epochs'] else 'sparse'
        history.append({'epoch': epoch + 1, 'train_routing': mode, 'train_loss': training['loss'],
                        'train_auroc': training['auroc'], 'val_loss': validation['loss'],
                        'val_auroc': validation['auroc'], 'val_auprc': validation['auprc'],
                        'val_balanced_accuracy': validation['balanced_accuracy'],
                        'val_threshold': validation['threshold'], 'best_epoch': stopping.best_epoch + 1,
                        'best_val_auroc': stopping.best_score, 'bad_epochs': stopping.bad_epochs,
                        'optimizer_steps': training['optimizer_steps'], 'elapsed_seconds': time.monotonic() - started})
        if config.get('label_smoothing', 0.) > 0:
            history[-1]['train_objective_loss'] = training['objective_loss']
        checkpoint = {'format_version': 1, 'fold': fold, 'epoch': epoch, 'run_signature': signature,
                      'metric_protocol': METRIC_PROTOCOL,
                      'model_config': model.configuration(), 'class_names': train_base.class_names,
                      'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
                      'early_stopping': asdict(stopping), 'rng_state': capture_rng(),
                      'loader_generator_state': generator.get_state(), 'history': history, 'best': best}
        # last.pt is the atomic source of truth and includes the best model.
        atomic_write(checkpoint_path, checkpoint, tensor=True)
        if improved:
            best_export(checkpoint, fold, directory)
        else:
            write_csv(directory / 'history.csv', history)
        LOGGER.info('Fold %d epoch=%d/%d %s train_loss=%.5f val_AUROC=%.5f best=%.5f@%d patience=%d/%d%s',
                    fold, epoch + 1, config['max_epochs'], mode, training['loss'], validation['auroc'],
                    stopping.best_score, stopping.best_epoch + 1, stopping.bad_epochs, config['patience'],
                    ' [best saved]' if improved else '')
    if best is None:
        raise RuntimeError('No checkpoint was produced')
    result = {'fold': fold, 'status': 'SMOKE_PASS' if config['smoke_test'] else 'COMPLETE',
              'train_samples': len(train_data), 'val_samples': len(val_data),
              'epochs_completed': len(history), 'best_epoch': best['epoch'] + 1,
              'validation_metrics': best['metrics'], 'early_stopped': stopping.stopped,
              'optimizer_steps': steps, 'positive_weight': float(positive_weight) if positive_weight is not None else None,
              'best_checkpoint': str(directory / 'best.pt'), 'last_checkpoint': str(checkpoint_path),
              'test_evaluated': False}
    write_json(directory / 'summary.json', result)
    return result


DEFAULTS = dict(folds=[1, 2, 3, 4, 5], seed=20261007, max_epochs=200, patience=10, min_delta=0.,
                learning_rate=2e-4, weight_decay=1e-4, dropout=.25, hidden_dim=256, gnn_layers=2,
                gat_heads=4, semantic_embedding_dim=16, branch_embedding_dim=16, num_experts=4,
                capacity_factor=1., warmup_epochs=5, batch_size=1, accumulation_steps=4,
                num_workers=0, threads=8, log_every=10, gradient_clip=5., pos_weight='none',
                smoke_test=False, global_alpha_init=.1, label_smoothing=0.,
                cache_graph_samples=False, classification_threshold=None)
RUNTIME_OPTIONS = {'device', 'threads', 'num_workers', 'log_every', 'max_epochs'}


def snapshot_manifests(source, destination):
    destination.mkdir()
    for fold in range(1, 6):
        path = source / f'fold_{fold}_cache_manifest.csv'
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            cache = Path(row['cache_path']).expanduser()
            cache = cache.resolve() if cache.is_absolute() else (source / cache).resolve()
            row['cache_path'] = os.path.relpath(cache, destination)
        write_csv(destination / path.name, rows)


def run_signature(config, protocol):
    protected = {k: v for k, v in config.items() if k not in RUNTIME_OPTIONS}
    payload = json.dumps({'config': protected, 'fold_sha256': [r['sha256'] for r in protocol['folds']]}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_config(config):
    for key in ['max_epochs', 'patience', 'hidden_dim', 'gnn_layers', 'gat_heads', 'semantic_embedding_dim',
                'branch_embedding_dim', 'num_experts', 'batch_size', 'accumulation_steps', 'threads', 'log_every']:
        if config[key] <= 0:
            raise ValueError(f'{key} must be positive')
    if min(config['num_workers'], config['warmup_epochs'], config['seed']) < 0:
        raise ValueError('Workers, warmup and seed must be nonnegative')
    for key in ['min_delta', 'weight_decay', 'gradient_clip']:
        if not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f'{key} must be finite and nonnegative')
    for key in ['learning_rate', 'capacity_factor']:
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if not 0 <= config['dropout'] < 1 or config['hidden_dim'] % config['gat_heads']:
        raise ValueError('Invalid dropout or hidden dimension / head count')
    if not 0 < config['global_alpha_init'] < 1 or not 0 <= config['label_smoothing'] < 1:
        raise ValueError('Invalid global residual alpha or label smoothing')
    threshold = config['classification_threshold']
    if threshold is not None and (not math.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError('Classification threshold must be between zero and one')
    if len(set(config['folds'])) != len(config['folds']):
        raise ValueError('Folds must be distinct')
    device = torch.device(config['device'])
    if device.type not in ('cpu', 'cuda') or (device.type == 'cuda' and (
            not torch.cuda.is_available() or (device.index or 0) >= torch.cuda.device_count())):
        raise ValueError('Unavailable training device')


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train S3MoE using five-fold manifests; select by validation AUROC.')
    parser.add_argument('--output', type=Path, required=True, help='New run directory, or an existing run with --resume')
    parser.add_argument('--split-directory', type=Path, help='Contains fold_1_cache_manifest.csv through fold_5_cache_manifest.csv')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--folds', type=int, nargs='+', choices=range(1, 6), default=argparse.SUPPRESS)
    for key, default in DEFAULTS.items():
        if key == 'folds':
            continue
        name = '--' + key.replace('_', '-')
        if isinstance(default, bool):
            parser.add_argument(name, action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
        elif key == 'pos_weight':
            parser.add_argument(name, choices=['none', 'balanced'], default=argparse.SUPPRESS)
        else:
            parser.add_argument(name, type=float if default is None else type(default), default=argparse.SUPPRESS)
    parser.add_argument('--device', default=argparse.SUPPRESS)
    args = vars(parser.parse_args())
    run = args.pop('output').expanduser().resolve()
    resume, source = args.pop('resume'), args.pop('split_directory')
    if resume:
        saved = json.loads((run / 'config.json').read_text())
        if source is not None:
            parser.error('Resume uses the frozen manifests; omit --split-directory')
        for key, value in args.items():
            if key not in RUNTIME_OPTIONS and value != saved['training'][key]:
                parser.error(f'Cannot change {key} during resume')
            if key == 'max_epochs' and value < saved['training'][key]:
                parser.error('Resume may extend max-epochs, not reduce it')
        config = {**saved['training'], **args}
        protocol = validate_protocol(run / 'splits')
        signature = run_signature(config, protocol)
        if signature != saved['run_signature']:
            raise ValueError('Frozen split/config fingerprint differs from training')
    else:
        if source is None:
            parser.error('A new run requires --split-directory')
        config = {**DEFAULTS, 'device': 'cuda:0' if torch.cuda.is_available() else 'cpu', **args}
        if config['smoke_test']:
            config['max_epochs'] = min(config['max_epochs'], 2)
            config['warmup_epochs'] = min(config['warmup_epochs'], 1)
        validate_config(config)
        source = source.expanduser().resolve()
        validate_protocol(source)
        run.mkdir(parents=True, exist_ok=False)
        snapshot_manifests(source, run / 'splits')
        protocol = validate_protocol(run / 'splits')
        signature = run_signature(config, protocol)
    validate_config(config)
    torch.set_num_threads(config['threads'])
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(run / 'training.log')])
    write_json(run / 'config.json', {'training': config, 'run_signature': signature})
    results = [fit_fold(config, fold, run / 'splits' / f'fold_{fold}_cache_manifest.csv',
                        run / f'fold_{fold}', signature, resume=resume) for fold in config['folds']]
    write_json(run / 'summary.json', {'status': 'SMOKE_PASS' if config['smoke_test'] else 'COMPLETE',
                                    'selection_metric': 'validation AUROC', 'test_used_for_selection': False,
                                    'folds': results})


if __name__ == '__main__':
    main()
