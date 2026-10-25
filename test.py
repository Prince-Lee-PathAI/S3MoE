"""Evaluate a validation-selected S3MoE checkpoint with its saved threshold."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

from model import S3MoE
from data import WSIGraphDataset, collate_wsi_graphs
from train import evaluate, file_sha256, write_csv, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True, help='Validation-selected best.pt')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--split', default='test', choices=['val', 'test'])
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    if min(args.batch_size, args.threads) < 1 or args.num_workers < 0:
        parser.error('Batch size/threads must be positive and workers nonnegative')
    if args.output.exists():
        parser.error('Output already exists; choose a new directory')
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    threshold = checkpoint['threshold']
    if threshold is None or not math.isfinite(threshold):
        raise ValueError('A finite saved validation threshold is required')
    dataset = WSIGraphDataset(args.manifest, args.split)
    model = S3MoE.from_configuration(checkpoint['model_config'])
    if tuple(checkpoint['class_names']) != dataset.class_names or model.encoder.config.input_dim != dataset.input_dim:
        raise ValueError('Checkpoint semantic classes/feature dimension differ from the data')
    model.load_state_dict(checkpoint['model_state'], strict=True)
    model.to(args.device)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                        shuffle=False, collate_fn=collate_wsi_graphs)
    metrics, rows = evaluate(model, loader, device=args.device, threshold=threshold)
    labels, predictions = [r['label'] for r in rows], [r['prediction'] for r in rows]
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary', zero_division=0)
    metrics.update(precision=float(precision), recall=float(recall), f1=float(f1),
                   confusion_matrix=confusion_matrix(labels, predictions, labels=[0, 1]).tolist())
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / 'predictions.csv', rows)
    write_json(args.output / 'metrics.json', {'split': args.split, 'n_wsi': len(rows),
                                             'checkpoint_sha256': file_sha256(args.checkpoint),
                                             'metrics': metrics})
    print(metrics)


if __name__ == '__main__':
    main()
