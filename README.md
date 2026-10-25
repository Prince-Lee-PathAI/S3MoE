# S³MoE

A compact implementation of semantic/spatial subgraph mixture-of-experts learning for WSI-level binary treatment response prediction. The complete model combines two GATv2 graph encoders, per-WSI expert-choice routing, attention across experts, and a gated global visual residual.

## Files

| File | Role |
| --- | --- |
| `model.py` | Complete S³MoE architecture, graph attention, pooling and expert routing |
| `semantics.py` | Frozen CPath-CLIP text encoding, semantic assignment and graph-cache export |
| `graphs.py` | Semantic/spatial graph construction, filtering and validation |
| `data.py` | Graph-cache dataset and batching |
| `train.py` | Five-fold training, validation selection, early stopping and resume |
| `test.py` | Checkpoint evaluation using the saved classification threshold |
| `configs/ovarian.json` | Fixed ovarian semantic definitions and prompts |
| `configs/breast.json` | Fixed breast definitions and separate HE/IHC prompts |

No cohort data or pretrained weights are bundled. Obtain CPath-CLIP and its Qwen text-encoder config/tokenizer separately from their original providers. This code assumes pre-extracted Virchow2 patch features; WSI tiling and foundation-model feature extraction are outside this release. Third-party models remain subject to their respective licenses.

## Environment

Python 3.10 was used for verification. Install a suitable PyTorch 2.5.1 build for your device, then:

```bash
pip install -r requirements.txt
```

`transformers` and `llm2vec` are required only to encode text prototypes. Graph preparation with saved prototypes, training, and testing need only PyTorch, NumPy, and scikit-learn. The pinned text-encoder versions preserve the bidirectional attention and rotary-buffer interfaces used by this implementation. No PyG or custom scatter extension is required.

Run commands from this directory. All data, weight and output locations are explicit command-line inputs.

## 1. Encode semantic prototypes once

```bash
python semantics.py encode \
  --config configs/ovarian.json \
  --checkpoint weights/cpath_clip.pt \
  --tokenizer-dir weights/Qwen-encoder-1.5B \
  --output data/ovarian_prototypes.pt
```

Use `configs/breast.json` to generate breast prototypes instead. HE and IHC prototypes are encoded separately and never averaged across stains. Text encoding averages normalized prompt embeddings per prototype, then normalizes the result. It uses masked mean pooling with an explicit bidirectional padding mask and the projected Virchow2 CLS channels.

## 2. Assign patches and construct graphs

```bash
python semantics.py prepare \
  --config configs/ovarian.json \
  --prototypes data/ovarian_prototypes.pt \
  --features data/features/slide_001.pt \
  --coords data/coords/slide_001.pt \
  --slide-id slide_001 \
  --output data/graphs/slide_001.pt
```

Inputs are a floating-point feature tensor `[N,D]` and an `int64` coordinate tensor `[N,2]`. Coordinates are nonnegative `(row, col)` indices on the original patch grid, **in exactly the same row order as the features**, with no duplicate positions. Supported feature widths are 1280 (Virchow2 CLS), 2560 (CLS + patch-token mean), and 3328 (CLIP 768 + CLS 1280 + patch-token mean 1280). The raw CLS block is retained for the downstream model; only a copy is normalized for semantic assignment.

Alternatively, replace `--coords` with `--patch-dir data/patches/slide_001` for naturally sorted JPG names of the form `instance_row_col_heightxwidth.jpg`. This option assumes that feature extraction used the same natural filename order; it checks metadata/counts but does not numerically re-encode patches to verify correspondence. Use explicit coordinates for other extraction orders.

For breast WSIs, use the breast config and prototypes and specify `--stain HE` or `--stain IHC`.

Graph defaults reproduce the full-model preprocessing: semantic radius 2 in Chebyshev grid distance, at most 256 nodes per semantic graph, an 8 × 8 spatial partition of the retained patch bounding box, and a minimum of 2 nodes per graph. Same-label semantic connected components are split recursively if oversized; radius-2 edges may span missing or differently labeled intermediate positions. Spatial graphs use immediate eight-neighbor edges within each window. Both branches have directed reverse edges and explicit self-loops. Graphs are filtered independently by branch; all original patches remain available to the global visual path.

Ovarian assignment first separates SET tumor, inflammation, necrosis, stroma and Other, then resolves SET morphology by cosine margin. The internal `nuclear_atypia` class denotes **SET ambiguity**, not an independently detected nuclear-atypia phenotype. Breast assignment uses five-class cosine argmax. Other is the maximum similarity over benign/background component prototypes. Scores are not calibrated probabilities.

The exported cache contains raw features, coordinates, semantic labels, branch graph IDs/masks, edges, edge attributes and class metadata, compatible with `data.py`. It stores no absolute source paths. Graph preprocessing is label-free and does not use treatment response labels.

## 3. Train

Provide a split directory containing `fold_1_cache_manifest.csv` through `fold_5_cache_manifest.csv`. Each CSV needs these columns:

```csv
slide_id,class_name,label,split,status,cache_path
slide_001,negative,0,train,PASS,../graphs/slide_001.pt
slide_002,positive,1,val,PASS,../graphs/slide_002.pt
slide_003,negative,0,test,PASS,../graphs/slide_003.pt
```

The rows above illustrate the schema only. Each complete fold must have both labels in train, val and test; these splits must be disjoint. The training pool and test set must be identical across the five manifests, while validation folds partition the training pool exactly once. Relative cache paths are resolved against the manifest directory. Use consistent class names and semantic mappings across all samples.

```bash
python train.py \
  --split-directory data/splits \
  --output runs/s3moe \
  --device cuda:0
```

The default model has 256 hidden dimensions, two GATv2 layers, four heads, four experts, capacity factor 1.0, dropout 0.25, and global residual alpha initialized to 0.1. Training uses AdamW, learning rate 2e-4, weight decay 1e-4, batch size 1, gradient accumulation over 4 WSIs, 5 dense-routing warm-up epochs, at most 200 epochs, and early-stopping patience 10. Expert capacity is computed separately for each WSI and branch as `min(M, max(1, ceil(capacity_factor * M / E)))` for nonempty branches. Routing probabilities are not renormalized after top-k selection; experts can select overlapping subgraphs.

These are configurable starting settings. The cohort runs used the following learning rates, patience and threshold policies:

| Cohort | Learning rate | Patience | Classification threshold |
| --- | --- | --- | --- |
| O.B.R | 2e-4 | 10 | Fixed 0.5 (`--classification-threshold 0.5`) |
| PTRC-HGSOC | 2e-4 | 10 | Validation balanced accuracy |
| TCGA-OV | 5e-5 | 20 | Validation balanced accuracy |
| IMPRESS-HER2+ | 5e-5 | 20 | Validation balanced accuracy |
| IMPRESS-TNBC | 2e-4 | 10 | Validation balanced accuracy |

Checkpoints are selected strictly by validation AUROC. Without a fixed threshold, the classification threshold maximizes validation balanced accuracy, with ties preferring proximity to 0.5. Test data are not evaluated during training. AUROC/AUPRC use logits for ranking to avoid sigmoid saturation; exported probabilities use float64 sigmoid.

Each fold saves `best.pt`, `last.pt`, validation predictions, training history and a summary. Resume an interrupted run with:

```bash
python train.py --output runs/s3moe --resume --device cuda:0
```

`--folds 1` trains one fold; the complete five-fold manifest protocol is still checked. `--smoke-test` limits training to a small balanced subset and at most two epochs. Use `--help` for configurable model/training options.

## 4. Test

```bash
python test.py \
  --checkpoint runs/s3moe/fold_1/best.pt \
  --manifest runs/s3moe/splits/fold_1_cache_manifest.csv \
  --output outputs/fold_1_test \
  --device cuda:0
```

Repeat for the other folds as needed. Testing exports per-WSI predictions and AUROC, AUPRC, accuracy, balanced accuracy, precision, recall, F1 and a confusion matrix. It uses the saved threshold without tuning on test labels. Existing full-model state dictionaries can be loaded strictly through `S3MoE.from_configuration(checkpoint['model_config'])`; ablation architectures are intentionally excluded.

Generated data, weights and run outputs are excluded by `.gitignore`. This source-only release does not include a license grant; the authors should add their chosen license before public distribution.
