#!/usr/bin/env python
import argparse
from pathlib import Path
import warnings
import gc

import lightning as L
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from esme.alphabet import Alphabet3, tokenize_unpad
from esme.esm import ESM2

warnings.filterwarnings("ignore")

# Use the same pretrained safetensors weights as the training script.
PRETRAIN_ESM_SAFETENSORS = "./model/pretrain_uniprot_bionemo_256_rawdata/step_safetensors/step-250/model.safetensors"


class SequenceClassificationDataset(Dataset):
    def __init__(self, df, max_residues):
        df = df[["id", "seq"]].copy()
        df["seq"] = df["seq"].astype(str).str.upper().str[:max_residues]
        df = df[df["seq"].str.len() > 0].reset_index(drop=True)
        self.ids = df["id"].astype(str).tolist()
        self.sequences = df["seq"].tolist()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.ids[idx], self.sequences[idx]


def collate_batch_infer(batch):
    ids, sequences = zip(*batch)
    tokens, _, cu_lens, max_len = tokenize_unpad(list(sequences), alphabet=Alphabet3)
    return list(ids), tokens, (cu_lens, max_len)


# CNN classifier used by this standalone inference script.
class ESM2CNNClassifier(L.LightningModule):
    def __init__(
        self,
        backbone,
        embed_dim,
        lr,
        weight_decay,
        freeze_backbone=False,
        dropout=0.1,
        hidden_dim=256,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.backbone = backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.dropout_rate = dropout
        self.cnn1 = nn.Conv1d(in_channels=embed_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.cnn2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.cnn3 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.final_linear = nn.Linear(hidden_dim, 2)

    def forward(self, tokens, pad_args):
        reps = self.backbone.forward_representation(tokens, pad_args)
        cu_lens, batch_max_len = pad_args
        cu_lens = cu_lens.to(reps.device)
        batch_size = int(cu_lens.shape[0]) - 1
        batch_list = []
        for i in range(batch_size):
            start = int(cu_lens[i].item())
            end = int(cu_lens[i + 1].item())
            seq_feat = reps[start:end, :]
            pad_len = batch_max_len - seq_feat.size(0)
            padded = F.pad(seq_feat, (0, 0, 0, pad_len), value=0.0)
            batch_list.append(padded)
        x = torch.stack(batch_list, dim=0)
        x = x.permute(0, 2, 1)
        x = self.cnn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.cnn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.cnn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x_pool = self.pool(x)
        graph_emb = x_pool.squeeze(-1)
        logits = self.final_linear(graph_emb)
        return logits


# Evaluation metrics.
def calc_metrics(y_true, y_pred, y_prob):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


# Checkpoint loading.
def load_pl_ckpt(ckpt_path, model_cls, device):
    backbone = ESM2.from_pretrained(PRETRAIN_ESM_SAFETENSORS, device="cpu")
    model = model_cls.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        backbone=backbone,
        map_location=device
    )
    model.eval()
    model.to(device)
    return model


# Batched CNN inference.
@torch.no_grad()
def run_cnn_infer(df, cnn_model, device, max_tokens=258, batch_size=64, num_workers=0):
    max_residues = max_tokens - 2
    ds = SequenceClassificationDataset(df, max_residues)
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=collate_batch_infer, pin_memory=False
    )
    rec = []
    for ids, tokens, pad_args in dl:
        tokens = tokens.to(device)
        cu_lens, max_len = pad_args
        pad_args_dev = (cu_lens.to(device), max_len)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
            logits_cnn = cnn_model(tokens, pad_args_dev)

        def get_pred_prob(logits):
            prob = torch.softmax(logits.float(), dim=-1)[:,1].cpu()
            pred = torch.argmax(logits, dim=-1).cpu()
            return pred.numpy(), prob.numpy()

        cnn_pred, cnn_p1 = get_pred_prob(logits_cnn)
        for idx, sid in enumerate(ids):
            rec.append({
                "id": sid,
                "cnn_pred": int(cnn_pred[idx]),
                "cnn_prob1": float(cnn_p1[idx]),
            })
    return pd.DataFrame(rec)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True, help="Input CSV with required id and seq columns and an optional label column")
    parser.add_argument("--output-dir", default="./result/cnn_pred")
    parser.add_argument("--cnn-ckpt", default="./model/CNN.ckpt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size-esme", type=int, default=64)
    parser.add_argument("--max-tokens-per-seq", type=int, default=258)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"Using device: {device}")

    df_input = pd.read_csv(args.input_csv)
    assert {"id", "seq"}.issubset(set(df_input.columns)), "The input CSV must contain id and seq columns."
    has_label = "label" in df_input.columns
    print(f"Input samples: {len(df_input)}, label column present: {has_label}")

    # Load the CNN checkpoint.
    print("\nLoading the CNN checkpoint...")
    model_cnn = load_pl_ckpt(args.cnn_ckpt, ESM2CNNClassifier, device)
    print("Model loaded")

    print("\nRunning ESME-CNN inference...")
    df_cnn = run_cnn_infer(
        df_input, model_cnn,
        device=device, max_tokens=args.max_tokens_per_seq,
        batch_size=args.batch_size_esme, num_workers=args.num_workers
    )

    df_out = df_input.merge(df_cnn, on="id", how="left")

    # Write the CNN predictions to predictions.csv.
    out_cols = ["id", "seq"]
    if has_label:
        out_cols.append("label")
    out_cols += ["cnn_pred", "cnn_prob1"]
    df_out[out_cols].to_csv(out_dir / "predictions.csv", index=False)
    print(f"\nPredictions saved to {out_dir / 'predictions.csv'}")

    # Compute metrics only for labeled input.
    if has_label:
        y_true = df_out["label"].to_numpy()
        metric_list = []
        models_info = [
            ("cnn", df_out["cnn_pred"].values, df_out["cnn_prob1"].values),
        ]
        for m_name, y_pred, y_prob in models_info:
            met = calc_metrics(y_true, y_pred, y_prob)
            met["model_name"] = m_name
            metric_list.append(met)
        df_metric = pd.DataFrame(metric_list)[["model_name", "accuracy", "precision", "recall", "f1", "roc_auc"]]
        df_metric.to_csv(out_dir / "metrics.csv", index=False)
        print("Evaluation metrics:")
        print(df_metric.to_string(index=False))
        print(f"Metrics saved to {out_dir / 'metrics.csv'}")

    del model_cnn
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("\nInference complete")


if __name__ == "__main__":
    main()
