#!/usr/bin/env python
import argparse
import json
import math
import os
import random
import sqlite3
from pathlib import Path

import lightning as L
import polars as pl
import torch
from lightning.pytorch.loggers import CSVLogger
from safetensors.torch import save_model
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from esme.alphabet import Alphabet3, mask_tokens, tokenize_unpad
from esme.esm import ESM2
from esme.loss import cross_entropy


# Backbone configurations for the 6M and 8M models.
ESM2_CONFIGS = {
    "6m": dict(num_layers=5, embed_dim=288, attention_heads=12)
}


class StepSafetensorsCheckpoint(L.Callback):
    # Checkpoint export steps.
    TARGET_SAVE_STEPS = {125, 250, 500, 1000, 2000, 3000, 5000}

    def __init__(self, output_dir, max_tokens_per_seq):
        self.output_dir = Path(output_dir) / "step_safetensors"
        self.max_tokens_per_seq = max_tokens_per_seq
        self._saved_steps = set()

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        current_step = trainer.global_step
        # Export weights only at the configured steps.
        if current_step <= 0 or current_step in self._saved_steps or current_step not in self.TARGET_SAVE_STEPS:
            return
        self._saved_steps.add(current_step)
        if trainer.is_global_zero:
            step_dir = self.output_dir / f"step-{current_step}"
            save_safetensors(pl_module, step_dir, self.max_tokens_per_seq)
            save_metrics(trainer, step_dir, current_step)


class SQLiteSequenceStore:
    def __init__(self, db_path):
        self.db_path = str(db_path)
        self.connection = None

    def connect(self):
        if self.connection is None:
            self.connection = sqlite3.connect(
                f"file:{self.db_path}?mode=ro",
                uri=True,
                timeout=60,
                check_same_thread=False,
            )
        return self.connection

    def get_many(self, ids):
        ids = list(ids)
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        rows = self.connect().execute(
            f"SELECT id, sequence FROM protein WHERE id IN ({placeholders})", ids
        ).fetchall()
        return dict(rows)

    def iter_sequences(self, chunk_size=8192):
        cur = self.connect().execute("SELECT id, sequence FROM protein")
        while True:
            rows = cur.fetchmany(chunk_size)
            if not rows:
                break
            for _, sequence in rows:
                yield sequence


class UniRefClusterIterableDataset(IterableDataset):
    def __init__(
        self,
        clusters_path,
        db_path,
        token_per_batch,
        max_residues,
        parquet_chunk_size=4096,
        query_chunk_size=512,
        limit_clusters=None,
        seed=0,
    ):
        self.clusters_path = str(clusters_path)
        self.store = SQLiteSequenceStore(db_path)
        self.token_per_batch = token_per_batch
        self.max_residues = max_residues
        self.parquet_chunk_size = parquet_chunk_size
        self.query_chunk_size = query_chunk_size
        self.limit_clusters = limit_clusters
        self.seed = seed

    def _worker_shard(self):
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        num_workers = worker.num_workers if worker else 1
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        global_worker_id = rank * num_workers + worker_id
        global_workers = world_size * num_workers
        return global_worker_id, global_workers

    def _cluster_chunks(self, global_worker_id, global_workers):
        scan = pl.scan_parquet(self.clusters_path).with_row_index("row_idx")
        if self.limit_clusters is not None:
            scan = scan.filter(pl.col("row_idx") < self.limit_clusters)
        scan = scan.filter((pl.col("row_idx") % global_workers) == global_worker_id)

        offset = 0
        while True:
            chunk = scan.slice(offset, self.parquet_chunk_size).collect()
            if chunk.is_empty():
                break
            yield chunk
            offset += self.parquet_chunk_size

    def __iter__(self):
        global_worker_id, global_workers = self._worker_shard()
        rng = random.Random(self.seed + global_worker_id)
        pending_ids = []
        batch = []
        token_count = 0

        def add_sequence(sequence):
            nonlocal batch, token_count
            if len(sequence) > self.max_residues:
                return None
            seq_tokens = len(sequence) + 2
            if batch and token_count + seq_tokens > self.token_per_batch:
                output = collate_mlm_sequences(batch)
                batch = [sequence]
                token_count = seq_tokens
                return output
            batch.append(sequence)
            token_count += seq_tokens
            return None

        def flush_pending_ids():
            nonlocal pending_ids
            seqs_by_id = self.store.get_many(pending_ids)
            ids = pending_ids
            pending_ids = []
            for seq_id in ids:
                sequence = seqs_by_id.get(seq_id)
                if sequence is None:
                    continue
                output = add_sequence(sequence)
                if output is not None:
                    yield output

        for chunk in self._cluster_chunks(global_worker_id, global_workers):
            ur90_id_lists = chunk["ur90_id"].to_list()
            rng.shuffle(ur90_id_lists)
            for ur90_ids in ur90_id_lists:
                if not ur90_ids:
                    continue
                pending_ids.append(rng.choice(ur90_ids))
                if len(pending_ids) >= self.query_chunk_size:
                    yield from flush_pending_ids()

        if pending_ids:
            yield from flush_pending_ids()
        if batch:
            yield collate_mlm_sequences(batch)


class SQLiteValidationIterableDataset(IterableDataset):
    def __init__(self, db_path, token_per_batch, max_residues, limit_sequences=None):
        self.store = SQLiteSequenceStore(db_path)
        self.token_per_batch = token_per_batch
        self.max_residues = max_residues
        self.limit_sequences = limit_sequences

    def _worker_shard(self):
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        num_workers = worker.num_workers if worker else 1
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        global_worker_id = rank * num_workers + worker_id
        global_workers = world_size * num_workers
        return global_worker_id, global_workers

    def __iter__(self):
        global_worker_id, global_workers = self._worker_shard()
        count = 0
        seen = 0
        batch = []
        token_count = 0
        for sequence in self.store.iter_sequences():
            row_idx = seen
            seen += 1
            if row_idx % global_workers != global_worker_id:
                continue
            if len(sequence) > self.max_residues:
                continue
            seq_tokens = len(sequence) + 2
            if batch and token_count + seq_tokens > self.token_per_batch:
                yield collate_mlm_sequences(batch)
                batch = []
                token_count = 0
            batch.append(sequence)
            token_count += seq_tokens
            count += 1
            if self.limit_sequences is not None and count >= self.limit_sequences:
                break
        if batch:
            yield collate_mlm_sequences(batch)


def collate_mlm_sequences(sequences):
    tokens, _, cu_lens, max_len = tokenize_unpad(sequences, alphabet=Alphabet3)
    masked_tokens, mask = mask_tokens(tokens, alphabet=Alphabet3)
    return tokens, (cu_lens, max_len), masked_tokens, mask


class ShortPeptideMLM(L.LightningModule):
    def __init__(self, model, lr, weight_decay, warmup_steps, max_steps):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

    def training_step(self, batch, batch_idx):
        target, pad_args, token, mask = batch
        logits = self.model(token, pad_args)
        loss = cross_entropy(logits, target, mask, alphabet=Alphabet3)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        target, pad_args, token, mask = batch
        logits = self.model(token, pad_args)
        loss = cross_entropy(logits, target, mask, alphabet=Alphabet3)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        if self.warmup_steps <= 0 or self.max_steps <= 0:
            return optimizer

        def lr_lambda(step):
            if step < self.warmup_steps:
                return max(step, 1) / self.warmup_steps
            progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
                "interval": "step",
            },
        }


def build_model(args):
    if args.model_size not in ESM2_CONFIGS:
        raise ValueError(f"Unsupported model size: {args.model_size}")
    return ESM2(
        **ESM2_CONFIGS[args.model_size],
        checkpointing=args.checkpointing,
        dtype=torch.bfloat16,
    )


def make_dataloaders(args):
    data_dir = Path(args.data_dir)
    max_residues = args.max_tokens_per_seq - 2
    train_ds = UniRefClusterIterableDataset(
        clusters_path=data_dir / "train_clusters.parquet",
        db_path=data_dir / "train.db",
        token_per_batch=args.token_per_batch,
        max_residues=max_residues,
        limit_clusters=args.limit_train_clusters,
        parquet_chunk_size=args.parquet_chunk_size,
        query_chunk_size=args.query_chunk_size,
        seed=args.seed,
    )
    val_ds = SQLiteValidationIterableDataset(
        db_path=data_dir / "validation.db",
        token_per_batch=args.token_per_batch,
        max_residues=max_residues,
        limit_sequences=args.limit_val_sequences,
    )
    val_workers = 0 if args.num_workers == 0 else max(1, args.num_workers // 2)
    train_dl = DataLoader(train_ds, batch_size=None, num_workers=args.num_workers)
    val_dl = DataLoader(val_ds, batch_size=None, num_workers=val_workers)
    return train_dl, val_dl


def dry_run(train_dl, val_dl, num_batches):
    for name, dl in [("train", train_dl), ("val", val_dl)]:
        print(f"{name} batches")
        for idx, batch in zip(range(num_batches), dl):
            target, (cu_lens, max_len), token, mask = batch
            print(
                f"  batch={idx} tokens={target.numel()} seqs={len(cu_lens) - 1} "
                f"max_len={max_len} masked={int(mask.sum())}"
            )


def save_safetensors(pl_model, output_dir, max_tokens_per_seq):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = pl_model.model
    metadata = {
        "format": "pt",
        "name": "esm2",
        "num_layers": str(model.num_layers),
        "embed_dim": str(model.embed_dim),
        "attention_heads": str(model.attention_heads),
        "max_tokens_per_seq": str(max_tokens_per_seq),
    }
    save_model(model, output_dir / "model.safetensors", metadata=metadata)


def save_metrics(trainer, output_dir, step):
    output_dir = Path(output_dir)
    metrics = {"step": step}
    for name in ("train_loss_step", "train_loss_epoch", "val_loss"):
        value = trainer.callback_metrics.get(name)
        if value is not None:
            metrics[name] = float(value.detach().cpu())
    with (output_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data/uniprot_pretrain")
    parser.add_argument("--output-dir", default="./model/pretrain_uniprot_bionemo_256_rawdata")
    parser.add_argument("--model-size", default="6m", choices=sorted(ESM2_CONFIGS))
    parser.add_argument("--devices", type=int, default=2)
    # Allow 256 residues plus two special tokens.
    parser.add_argument("--max-tokens-per-seq", type=int, default=258)
    parser.add_argument("--token-per-batch", type=int, default=50_000)
    parser.add_argument("--accumulate-grad-batches", type=int, default=8)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-epochs", type=int, default=-1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--val-check-interval", type=int, default=125)
    parser.add_argument("--limit-train-clusters", type=int)
    parser.add_argument("--limit-val-sequences", type=int)
    parser.add_argument("--parquet-chunk-size", type=int, default=4096)
    parser.add_argument("--query-chunk-size", type=int, default=512)
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--dry-run-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_tokens_per_seq != 258:
        print(f"Using max_tokens_per_seq={args.max_tokens_per_seq}; short peptide default is 258.")

    torch.set_float32_matmul_precision("medium")
    L.seed_everything(args.seed, workers=True)

    train_dl, val_dl = make_dataloaders(args)
    if args.dry_run_batches > 0:
        dry_run(train_dl, val_dl, args.dry_run_batches)
        return

    model = build_model(args)
    pl_model = ShortPeptideMLM(
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
    )

    output_dir = Path(args.output_dir)
    checkpoint_callback = StepSafetensorsCheckpoint(
        output_dir=output_dir,
        max_tokens_per_seq=args.max_tokens_per_seq,
    )
    logger = CSVLogger(save_dir="./result/pretrain_uniprot_bionemo_256_rawdata", name="logs")

    trainer = L.Trainer(
        accelerator="gpu",
        devices=args.devices,
        strategy="ddp" if args.devices > 1 else "auto",
        precision=args.precision,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        val_check_interval=args.val_check_interval,
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
        reload_dataloaders_every_n_epochs=1,
    )
    trainer.fit(pl_model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    if trainer.is_global_zero:
        save_safetensors(pl_model, output_dir, args.max_tokens_per_seq)
        print(f"Saved final model to {output_dir / 'model.safetensors'}")


if __name__ == "__main__":
    main()
