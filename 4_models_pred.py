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
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from esme.alphabet import Alphabet3, tokenize_unpad
from esme.esm import ESM2

warnings.filterwarnings("ignore")

# Use the same pretrained safetensors weights as the training scripts.
PRETRAIN_ESM_SAFETENSORS = "./model/pretrain_uniprot_bionemo_256_rawdata/step_safetensors/step-250/model.safetensors"

# Lightning model definitions shared with training.
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


class ESM2LSTMClassifier(L.LightningModule):
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
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=3,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )
        self.final_linear = nn.Linear(hidden_dim, 2)

    def forward(self, tokens, pad_args):
        reps = self.backbone.forward_representation(tokens, pad_args)
        cu_lens, max_len = pad_args
        cu_lens = cu_lens.to(reps.device)
        batch_size = int(cu_lens.shape[0]) - 1
        seq_list = []
        lengths = []
        for i in range(batch_size):
            start = int(cu_lens[i].item())
            end = int(cu_lens[i+1].item())
            seq_list.append(reps[start:end, :])
            lengths.append(end - start)
        padded_x = pad_sequence(seq_list, batch_first=True)
        lengths_cpu = torch.tensor(lengths, dtype=torch.long).cpu()
        packed_x = pack_padded_sequence(padded_x, lengths_cpu, batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(packed_x)
        graph_emb = hn[-1, :, :]
        logits = self.final_linear(graph_emb)
        return logits


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


class ESM2MLPClassifier(L.LightningModule):
    def __init__(
        self,
        backbone,
        embed_dim,
        lr,
        weight_decay,
        freeze_backbone=False,
        dropout=0.1,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.backbone = backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        # Match the checkpoint: the first layer outputs 256 features and the second preserves that size.
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2)
        )

    def forward(self, tokens, pad_args):
        reps = self.backbone.forward_representation(tokens, pad_args)
        cls_positions = pad_args[0][:-1].long().to(reps.device)
        cls_reps = reps.index_select(0, cls_positions)
        return self.classifier(cls_reps)


# Standalone CNN for final_model.pt; it uses the training architecture without ESM2.
DROPOUT_RATE = 0.1772219731237647
MAX_SEQ_LEN = 256

class AminoAcidTokenizer:
    def __init__(self):
        self.amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        self.special_tokens = {"[PAD]": 0, "[UNK]": 1}
        self.token_to_id = self.special_tokens.copy()
        for i, aa in enumerate(self.amino_acids):
            self.token_to_id[aa] = i + len(self.special_tokens)
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.pad_token_id = 0
        self.vocab_size = len(self.token_to_id)

    def encode_batch(self, sequences, max_length=256):
        encoded = []
        for seq in sequences:
            seq_ids = []
            for char in seq[:max_length]:
                char = char.upper()
                seq_ids.append(self.token_to_id.get(char, self.token_to_id["[UNK]"]))
            encoded.append(seq_ids)
        return encoded


class LightweightModel(nn.Module):
    def __init__(self, vocab_size=22, embed_dim=192, hidden_dim=256, max_seq_len=256):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(DROPOUT_RATE)
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.AdaptiveMaxPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(64, 2)
        )

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.embed_dropout(x)
        x = x.transpose(1, 2)
        x = self.feature_extractor(x)
        x = x.squeeze(-1)
        return self.classifier(x)


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
    """Load an LSTM, CNN, or MLP Lightning checkpoint."""
    # Build the ESM2 backbone first; the checkpoint restores its weights.
    backbone = ESM2.from_pretrained(PRETRAIN_ESM_SAFETENSORS, device="cpu")
    model = model_cls.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        backbone=backbone,
        map_location=device
    )
    model.eval()
    model.to(device)
    return model


def load_final_cnn_pt(pt_path, device):
    tokenizer = AminoAcidTokenizer()
    model = LightweightModel(vocab_size=tokenizer.vocab_size, embed_dim=192, hidden_dim=256, max_seq_len=MAX_SEQ_LEN)
    ckpt = torch.load(pt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    return model, tokenizer


# Inference helpers.
@torch.no_grad()
def run_esme_models_infer(df, lstm_model, cnn_model, mlp_model, device, max_tokens=258, batch_size=64, num_workers=2):
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
            logits_lstm = lstm_model(tokens, pad_args_dev)
            logits_cnn = cnn_model(tokens, pad_args_dev)
            logits_mlp = mlp_model(tokens, pad_args_dev)

        def get_pred_prob(logits):
            prob = torch.softmax(logits.float(), dim=-1)[:,1].cpu()
            pred = torch.argmax(logits, dim=-1).cpu()
            return pred.numpy(), prob.numpy()

        lstm_pred, lstm_p1 = get_pred_prob(logits_lstm)
        cnn_pred, cnn_p1 = get_pred_prob(logits_cnn)
        mlp_pred, mlp_p1 = get_pred_prob(logits_mlp)

        for idx, sid in enumerate(ids):
            rec.append({
                "id": sid,
                "lstm_pred": int(lstm_pred[idx]),
                "lstm_prob1": float(lstm_p1[idx]),
                "cnn_pred": int(cnn_pred[idx]),
                "cnn_prob1": float(cnn_p1[idx]),
                "mlp_pred": int(mlp_pred[idx]),
                "mlp_prob1": float(mlp_p1[idx]),
            })
    return pd.DataFrame(rec)


@torch.no_grad()
def run_final_cnn_infer(df, model, tokenizer, device, batch_size=1024):
    seq_list = df["seq"].astype(str).str.upper().tolist()
    id_list = df["id"].astype(str).tolist()
    encoded = tokenizer.encode_batch(seq_list, max_length=MAX_SEQ_LEN)
    rec = []
    for start in range(0, len(id_list), batch_size):
        end = start + batch_size
        batch_ids = id_list[start:end]
        batch_enc = encoded[start:end]
        max_len = min(max(len(x) for x in batch_enc), MAX_SEQ_LEN)
        import numpy as np
        padded = np.zeros((len(batch_enc), max_len), dtype=np.int64)
        for i, s in enumerate(batch_enc):
            sl = min(len(s), max_len)
            padded[i,:sl] = s[:sl]
        input_t = torch.from_numpy(padded).to(device)
        logits = model(input_t)
        prob = torch.softmax(logits.float(), dim=-1)[:,1].cpu()
        pred = torch.argmax(logits, dim=-1).cpu()
        for i, sid in enumerate(batch_ids):
            rec.append({
                "id": sid,
                "final_cnn_pred": int(pred[i]),
                "final_cnn_prob1": float(prob[i])
            })
    return pd.DataFrame(rec)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True, help="Input CSV with required id and seq columns and an optional label column")
    parser.add_argument("--output-dir", default="./result/ensemble")
    parser.add_argument("--lstm-ckpt", default="./model/LSTM.ckpt")
    parser.add_argument("--cnn-ckpt", default="./model/CNN.ckpt")
    parser.add_argument("--mlp-ckpt", default="./model/MLP.ckpt")
    parser.add_argument("--final-pt", default="./model/final_model.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size-esme", type=int, default=64)
    parser.add_argument("--batch-size-final", type=int, default=1024)
    parser.add_argument("--max-tokens-per-seq", type=int, default=258)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Read the input CSV.
    df_input = pd.read_csv(args.input_csv)
    assert {"id", "seq"}.issubset(set(df_input.columns)), "The input CSV must contain id and seq columns."
    has_label = "label" in df_input.columns
    print(f"Input samples: {len(df_input)}, label column present: {has_label}")

    # Load the four checkpoints.
    print("\nLoading the LSTM checkpoint...")
    model_lstm = load_pl_ckpt(args.lstm_ckpt, ESM2LSTMClassifier, device)
    print("Loading the CNN checkpoint...")
    model_cnn = load_pl_ckpt(args.cnn_ckpt, ESM2CNNClassifier, device)
    print("Loading the MLP checkpoint...")
    model_mlp = load_pl_ckpt(args.mlp_ckpt, ESM2MLPClassifier, device)
    print("Loading final_model.pt...")
    model_final_cnn, tokenizer_final = load_final_cnn_pt(args.final_pt, device)
    print("All models loaded")

    # Run inference with each model.
    print("\nRunning LSTM, CNN, and MLP inference...")
    df_esme = run_esme_models_infer(
        df_input, model_lstm, model_cnn, model_mlp,
        device=device, max_tokens=args.max_tokens_per_seq,
        batch_size=args.batch_size_esme, num_workers=args.num_workers
    )
    print("Running standalone CNN inference...")
    df_final = run_final_cnn_infer(df_input, model_final_cnn, tokenizer_final, device, batch_size=args.batch_size_final)

    # Combine the individual predictions.
    df_out = df_input.merge(df_esme, on="id", how="left").merge(df_final, on="id", how="left")
    # Mark a sample positive only if all four models predict class 1.
    df_out["ensemble_pred"] = ((df_out["lstm_pred"] == 1)
                              & (df_out["cnn_pred"] == 1)
                              & (df_out["mlp_pred"] == 1)
                              & (df_out["final_cnn_pred"] == 1)).astype(int)

    # Write predictions.csv.
    out_cols = ["id", "seq"]
    if has_label:
        out_cols.append("label")
    out_cols += [
        "lstm_pred", "lstm_prob1",
        "cnn_pred", "cnn_prob1",
        "mlp_pred", "mlp_prob1",
        "final_cnn_pred", "final_cnn_prob1",
        "ensemble_pred"
    ]
    df_out[out_cols].to_csv(out_dir / "predictions.csv", index=False)
    print(f"\nPredictions saved to {out_dir / 'predictions.csv'}")

    # Compute metrics only for labeled input.
    if has_label:
        y_true = df_out["label"].to_numpy()
        metric_list = []
        # Evaluate each model separately.
        models_info = [
            ("lstm", df_out["lstm_pred"].values, df_out["lstm_prob1"].values),
            ("cnn", df_out["cnn_pred"].values, df_out["cnn_prob1"].values),
            ("mlp", df_out["mlp_pred"].values, df_out["mlp_prob1"].values),
            ("final_cnn", df_out["final_cnn_pred"].values, df_out["final_cnn_prob1"].values),
            ("ensemble", df_out["ensemble_pred"].values, None)
        ]
        for m_name, y_pred, y_prob in models_info:
            met = calc_metrics(y_true, y_pred, y_prob if y_prob is not None else y_pred)
            met["model_name"] = m_name
            metric_list.append(met)
        df_metric = pd.DataFrame(metric_list)[["model_name", "accuracy", "precision", "recall", "f1", "roc_auc"]]
        df_metric.to_csv(out_dir / "metrics.csv", index=False)
        print("Evaluation metrics:")
        print(df_metric.to_string(index=False))
        print(f"Metrics saved to {out_dir / 'metrics.csv'}")

    # Release GPU memory.
    del model_lstm, model_cnn, model_mlp, model_final_cnn
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("\nInference complete")


if __name__ == "__main__":
    main()
