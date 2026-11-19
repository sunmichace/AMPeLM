#!/usr/bin/env python
import argparse
from pathlib import Path

import lightning as L
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

from esme.alphabet import Alphabet3, tokenize_unpad
from esme.esm import ESM2


# Root directory for the pretrained 6M backbone checkpoints.
CHECKPOINT_ROOT = Path("./model/pretrain_uniprot_bionemo_256_rawdata")
RESULT_ROOT = Path("./result/finetune/CNN")


class SequenceClassificationDataset(Dataset):
    def __init__(self, csv_path, max_residues, limit=None):
        df = pd.read_csv(csv_path)
        required = {"id", "seq", "label"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
        df = df[["id", "seq", "label"]].dropna()
        df["seq"] = df["seq"].astype(str).str.upper().str[:max_residues]
        df = df[df["seq"].str.len() > 0]
        df["label"] = df["label"].astype(int)
        if limit is not None:
            df = df.head(limit)
        self.ids = df["id"].astype(str).tolist()
        self.sequences = df["seq"].tolist()
        self.labels = df["label"].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.ids[idx], self.sequences[idx], self.labels[idx]


def collate_batch(batch):
    ids, sequences, labels = zip(*batch)
    tokens, _, cu_lens, max_len = tokenize_unpad(list(sequences), alphabet=Alphabet3)
    labels = torch.tensor(labels, dtype=torch.long)
    return list(ids), tokens, (cu_lens, max_len), labels


# Three-layer 1D CNN classification head.
class ESM2SequenceClassifier(L.LightningModule):
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
        # Use padding of 1 with kernel size 3 to preserve the sequence length.
        self.cnn1 = nn.Conv1d(in_channels=embed_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.cnn2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.cnn3 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)

        self.pool = nn.AdaptiveAvgPool1d(output_size=1)  # Pool each sequence to one feature vector.
        self.final_linear = nn.Linear(hidden_dim, 2)

    def forward(self, tokens, pad_args):
        reps = self.backbone.forward_representation(tokens, pad_args)
        cu_lens, batch_max_len = pad_args
        cu_lens = cu_lens.to(reps.device)

        batch_size = int(cu_lens.shape[0]) - 1
        embed_dim = reps.size(-1)

        # Restore the per-sequence features before applying Conv1d.
        batch_list = []
        for i in range(batch_size):
            start = int(cu_lens[i].item())
            end = int(cu_lens[i + 1].item())
            seq_feat = reps[start:end, :]  # [seq_len, embed_dim]
            pad_len = batch_max_len - seq_feat.size(0)
            padded = F.pad(seq_feat, (0, 0, 0, pad_len), value=0.0)
            batch_list.append(padded)

        x = torch.stack(batch_list, dim=0)  # [B, L, embed_dim]
        x = x.permute(0, 2, 1)  # Conv1d expects [batch, channels, length].

        # Apply convolution, ReLU, and dropout in each block.
        x = self.cnn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.cnn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.cnn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # Pool over the sequence dimension.
        x_pool = self.pool(x)  # [B, hidden_dim, 1]
        graph_emb = x_pool.squeeze(-1)  # [B, hidden_dim]
        logits = self.final_linear(graph_emb)
        return logits

    def training_step(self, batch, batch_idx):
        _, tokens, pad_args, labels = batch
        logits = self(tokens, pad_args)
        loss = F.cross_entropy(logits, labels.to(logits.device))
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=labels.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        _, tokens, pad_args, labels = batch
        logits = self(tokens, pad_args)
        labels = labels.to(logits.device)
        loss = F.cross_entropy(logits, labels)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True, batch_size=labels.size(0))

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)


def make_dataloader(csv_path, max_residues, batch_size, num_workers, shuffle, limit=None):
    dataset = SequenceClassificationDataset(csv_path, max_residues=max_residues, limit=limit)
    if len(dataset) == 0:
        raise ValueError(f"No usable rows in {csv_path}; check sequence length and labels.")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_batch,
        pin_memory=True,
    )


def compute_metrics(labels, preds, probs):
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(labels, probs)
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


def predict(pl_model, dataloader, device):
    pl_model.eval().to(device)
    all_ids, all_labels, all_preds, all_probs = [], [], [], []
    with torch.no_grad():
        for ids, tokens, pad_args, labels in dataloader:
            tokens = tokens.to(device)
            pad_args = (pad_args[0].to(device), pad_args[1])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
                logits = pl_model(tokens, pad_args)
            probs = torch.softmax(logits.float(), dim=-1)[:, 1].cpu()
            preds = torch.argmax(logits, dim=-1).cpu()
            all_ids.extend(ids)
            all_labels.extend(labels.tolist())
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())
    return all_ids, all_labels, all_preds, all_probs


def write_eval_outputs(pl_model, val_dl, test_dl, output_dir, device):
    output_dir = Path(output_dir)
    rows = []
    for split, dl in [("val", val_dl), ("test", test_dl)]:
        ids, labels, preds, probs = predict(pl_model, dl, device)
        metrics = compute_metrics(labels, preds, probs)
        rows.append({"split": split, **metrics})
        pd.DataFrame({
            "id": ids,
            "label": labels,
            "pred": preds,
            "prob_positive": probs,
        }).to_csv(output_dir / f"{split}_predictions.csv", index=False)
        if split == "test":
            cm = confusion_matrix(labels, preds, labels=[0, 1])
            pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(
                output_dir / "test_confusion_matrix.csv"
            )
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    print(metrics_df.to_string(index=False))
    print(f"Saved metrics and predictions to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default="./data/rawdata_train.csv")
    parser.add_argument("--val-csv", default="./data/rawdata_val.csv")
    parser.add_argument("--test-csv", default="./data/rawdata_test.csv")
    parser.add_argument("--checkpoint-root", default="./model/pretrain_uniprot_bionemo_256_rawdata")
    parser.add_argument("--result-root", default="./result/finetune/CNN")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    # Allow 256 residues plus two special tokens.
    parser.add_argument("--max-tokens-per-seq", type=int, default=258)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=256, help="CNN hidden channel dimension")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-val", type=int)
    parser.add_argument("--limit-test", type=int)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_float32_matmul_precision("medium")
    L.seed_everything(args.seed, workers=True)
    max_residues = args.max_tokens_per_seq - 2

    # Fine-tune only the step-250 checkpoint.
    step_folder_list = ["step-250"]

    # Fine-tune each pretrained checkpoint independently.
    for step_tag in step_folder_list:
        print(f"\n====== Fine-tuning 6M {step_tag} ======")
        pretrained = Path(args.checkpoint_root) / "step_safetensors" / step_tag / "model.safetensors"
        if not pretrained.is_file():
            print(f"Warning: skipping missing checkpoint {pretrained}")
            continue

        # Store each run under result_root/6m/step-xxx.
        output_dir = Path(args.result_root) / "6m" / step_tag
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build fresh data loaders for this run.
        train_dl = make_dataloader(args.train_csv, max_residues, args.batch_size, args.num_workers, True, args.limit_train)
        val_dl = make_dataloader(args.val_csv, max_residues, args.batch_size, args.num_workers, False, args.limit_val)
        test_dl = make_dataloader(args.test_csv, max_residues, args.batch_size, args.num_workers, False, args.limit_test)

        backbone = ESM2.from_pretrained(str(pretrained), device="cpu")
        pl_model = ESM2SequenceClassifier(
            backbone=backbone,
            embed_dim=backbone.embed_dim,
            lr=args.lr,
            weight_decay=args.weight_decay,
            freeze_backbone=args.freeze_backbone,
            dropout=args.dropout,
            hidden_dim=args.hidden_dim,
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath=Path("./model/finetuned_cnn") / "6m" / step_tag / "checkpoints",
            filename="{epoch}-{step}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        )
        trainer = L.Trainer(
            accelerator="gpu",
            devices=args.devices,
            strategy="ddp" if args.devices > 1 else "auto",
            precision=args.precision,
            max_epochs=args.epochs,
            logger=CSVLogger(save_dir=output_dir, name="logs"),
            callbacks=[checkpoint_callback],
            log_every_n_steps=20,
        )
        trainer.fit(pl_model, train_dataloaders=train_dl, val_dataloaders=val_dl)

        if trainer.is_global_zero:
            if checkpoint_callback.best_model_path:
                pl_model = ESM2SequenceClassifier.load_from_checkpoint(
                    checkpoint_callback.best_model_path,
                    backbone=ESM2.from_pretrained(str(pretrained), device="cpu"),
                )
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            write_eval_outputs(pl_model, val_dl, test_dl, output_dir, device)

        # Release GPU memory before fine-tuning the next checkpoint.
        del trainer, pl_model, backbone, train_dl, val_dl, test_dl
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
