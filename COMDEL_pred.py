import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

ROOT_DIR = "./"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

TEST_CSVs = {
    "t1": os.path.join(ROOT_DIR, "data/bagel4_1_val.csv"),
    "t2": os.path.join(ROOT_DIR, "data/bagel4_2_val.csv"),
    "t3": os.path.join(ROOT_DIR, "data/bagel4_3_val.csv")
}
COMDEL_MODEL_PATH = os.path.join(ROOT_DIR, "model/final_model.pt")
# Preserve the existing output location for compatibility.
OUTPUT_DIR = os.path.join(ROOT_DIR, "result/xgboost_ensemble")

# Match the architecture and preprocessing used by the existing checkpoint.
DROPOUT_RATE = 0.1772219731237647
MAX_SEQ_LEN = 256
BATCH_SIZE = 64


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

    def batch_encode(self, sequences, max_length=256):
        encoded = []
        for seq in sequences:
            seq = seq.strip().upper()
            seq_ids = []
            for char in seq[:max_length]:
                char = char.upper()
                seq_ids.append(self.token_to_id.get(char, self.token_to_id["[UNK]"]))
            encoded.append(seq_ids)
        padded_list = []
        for ids in encoded:
            if len(ids) < max_length:
                ids = ids + [self.pad_token_id] * (max_length - len(ids))
            else:
                ids = ids[:max_length]
            padded_list.append(ids)
        return torch.tensor(padded_list, dtype=torch.long)


class LightweightCNN(nn.Module):
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
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.embed_dropout(x)
        x = x.transpose(1, 2)
        x = self.feature_extractor(x)
        x = x.squeeze(-1)
        logits = self.classifier(x)
        return logits


def load_csv_data(csv_path):
    """Return IDs, sequences, and binary labels after filtering invalid rows."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)

    if "seq" in df.columns:
        seq_col = "seq"
    elif "sequence" in df.columns:
        seq_col = "sequence"
    else:
        raise ValueError("Expected a 'seq' or 'sequence' column")

    if "label" not in df.columns:
        raise ValueError("Missing required 'label' column")

    if "id" not in df.columns:
        df = df.copy()
        df["id"] = np.arange(1, len(df) + 1)

    ids, seqs, labels = [], [], []
    for _, row in df.iterrows():
        seq = str(row[seq_col]).strip().upper()
        if len(seq) == 0:
            print(f"Skipping empty sequence (id: {row['id']})")
            continue
        try:
            label = int(row['label'])
            if label not in (0, 1):
                raise ValueError("Label must be 0 or 1")
        except (TypeError, ValueError, OverflowError):
            print(f"Skipping invalid label (id: {row['id']}, label: {row['label']})")
            continue
        ids.append(row['id'])
        seqs.append(seq)
        labels.append(label)

    total, pos = len(labels), sum(labels)
    if total == 0:
        raise ValueError(f"No valid rows in CSV file: {csv_path}")
    ratio = pos / total * 100
    print(f"Loaded {os.path.basename(csv_path)}: {total} valid sequences ({pos} positive, {ratio:.2f}%)")
    return ids, seqs, labels


def load_comdel_model(checkpoint_path, device):
    """Load the standalone CNN and its amino acid tokenizer."""
    tokenizer = AminoAcidTokenizer()
    model = LightweightCNN(vocab_size=tokenizer.vocab_size).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()
    return model, tokenizer


def predict_comdel(sequences, model, tokenizer, device):
    """Return CNN labels and positive-class probabilities in input order."""
    input_ids = tokenizer.batch_encode(sequences, max_length=MAX_SEQ_LEN)
    preds_list = []
    prob_list = []
    with torch.no_grad():
        for i in range(0, len(input_ids), BATCH_SIZE):
            batch_input = input_ids[i:i + BATCH_SIZE].to(device)
            logits = model(batch_input)
            prob = F.softmax(logits, dim=1)
            pred = torch.argmax(prob, dim=1)
            prob_list.extend(prob[:, 1].cpu().numpy())
            preds_list.extend(pred.cpu().numpy())
    return np.array(preds_list), np.array(prob_list)


def main():
    """Evaluate final_model.pt on each configured test set."""
    print(f"Starting final_model.pt inference | Device: {device}")
    model, tokenizer = load_comdel_model(COMDEL_MODEL_PATH, device)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for test_name, test_csv in TEST_CSVs.items():
        print(f"\n========== Test set: {test_name} ==========")
        ids, seqs, labels = load_csv_data(test_csv)
        preds, probs = predict_comdel(seqs, model, tokenizer, device)

        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, zero_division=0)
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)
        try:
            auc = roc_auc_score(labels, probs)
        except ValueError:
            # ROC AUC is undefined when the test set contains only one class.
            auc = 0.0

        result_df = pd.DataFrame({
            "id": ids,
            "sequence": seqs,
            "true_label": labels,
            "comdel_cnn_pred": preds,
            "comdel_pos_prob": np.round(probs, 6),
        })
        result_path = os.path.join(OUTPUT_DIR, f"{test_name}_xgboost_result.csv")
        result_df.to_csv(result_path, index=False, encoding="utf-8-sig")

        print(f"CNN - Accuracy: {acc:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | AUC: {auc:.4f}")
        print(f"Results saved to: {result_path}")

    print(f"\nInference complete. Results directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
