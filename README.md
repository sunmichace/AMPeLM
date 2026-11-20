# Peptide Sequence Classification

These Python scripts train and evaluate binary classifiers for peptide sequences. The lightweight CNN works directly with amino acid tokens. The ESM2-based classifiers use a pretrained backbone with an MLP, CNN, or LSTM classification head.

## Lightweight CNN

- `COMDEL_train.py` trains a 1D CNN from labeled CSVs with `id`, `seq`, and `label` columns. It uses validation loss to select the best checkpoint and stop training early, saves the final weights as `final_model.pt`, and evaluates the model on the configured validation and test sets.
- `COMDEL_pred.py` loads only `final_model.pt`. It runs inference without ESM2 or an ensemble. It uses the three test CSVs configured in `TEST_CSVs`. Each must contain `seq` (or `sequence`) and a binary `label` (`0` or `1`); when `id` is absent, the script assigns one.

```sh
python COMDEL_pred.py
```

Inference truncates or pads sequences to 256 residues, uses batches of 64, and preserves `DROPOUT_RATE = 0.1772219731237647`. The script applies softmax to get the class probabilities and selects the class with the higher probability. Each test set produces a CSV with `comdel_cnn_pred` and `comdel_pos_prob`, and the script prints accuracy, F1, precision, recall, and ROC AUC. The `_xgboost_result.csv` filename suffix is retained for compatibility; the predictions come from the CNN alone.

## ESM2 Training

| Script | What it does |
| --- | --- |
| `pretrain_uniprot_bionemo_256.py` | Pretrains an ESM2 backbone using masked language modeling and exports weights at selected training steps. |
| `finetune_classifier_rawdata_256MLP.py` | Fine-tunes a 6M ESM2 backbone with a three-layer MLP head at multiple checkpoint steps. |
| `finetune_classifier_rawdata_256CNN.py` | Fine-tunes the step-250 6M ESM2 checkpoint with a three-layer 1D CNN head. |
| `finetune_classifier_rawdata_256LSTM.py` | Fine-tunes the step-250 6M ESM2 checkpoint with a three-layer unidirectional LSTM head. |

The fine-tuning scripts take labeled CSVs through `--train-csv`, `--val-csv`, and `--test-csv`. Each CSV must have `id`, `seq`, and `label` columns.

## ESM2 and Ensemble Inference

- `CNN_pred.py` runs the ESM2-based CNN from `CNN.ckpt`. This is a separate model from the lightweight CNN.
- `4_models_pred.py` runs the ESM2-based LSTM, CNN, and MLP alongside the lightweight CNN. The combined prediction is positive only when all four models predict class `1`.

Both scripts require `id` and `seq` in the input CSV. They write `predictions.csv` and write `metrics.csv` only when the input includes `label`.

```sh
python CNN_pred.py --input-csv input.csv
python 4_models_pred.py --input-csv input.csv
```
