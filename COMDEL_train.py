import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_score, recall_score,
    f1_score, roc_curve, auc, precision_recall_curve, matthews_corrcoef, accuracy_score, roc_auc_score
)
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import torch.serialization
from sklearn.preprocessing import LabelEncoder
import time
import warnings
import sys
import psutil

warnings.filterwarnings('ignore')

torch.serialization.add_safe_globals([LabelEncoder])

plt.rcParams["axes.unicode_minus"] = False

# Training and evaluation input paths.
ROOT_DIR = "./"
train_csv = os.path.join(ROOT_DIR, "data/rawdata_train.csv")
val1_csv = os.path.join(ROOT_DIR, "data/rawdata_val.csv")
val2_csv = os.path.join(ROOT_DIR, "data/bagel4_1_val.csv")
val3_csv = os.path.join(ROOT_DIR, "data/bagel4_2_val.csv")
val4_csv = os.path.join(ROOT_DIR, "data/bagel4_3_val.csv")
test_csv = os.path.join(ROOT_DIR, "data/rawdata_test.csv")

# Training output paths.
val1_output = os.path.join(ROOT_DIR, "result/cnn_lite_train/val1_new_val_predictions.csv")
val2_output = os.path.join(ROOT_DIR, "result/cnn_lite_train/val2_bagel4_1_val_predictions.csv")
val3_output = os.path.join(ROOT_DIR, "result/cnn_lite_train/val3_bagel4_2_val_predictions.csv")
val4_output = os.path.join(ROOT_DIR, "result/cnn_lite_train/val4_bagel4_3_val_predictions.csv")
test_output = os.path.join(ROOT_DIR, "result/cnn_lite_train/test_val_test_predictions.csv")
metrics_summary_output = os.path.join(ROOT_DIR, "result/cnn_lite_train/metrics_summary.csv")
cumulative_curve_dir = os.path.join(ROOT_DIR, "result/cnn_lite_train/cumulative_curves")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


# Amino acid tokenizer.
class AminoAcidTokenizer:
    def __init__(self):
        # Use the 20 standard amino acids plus padding and unknown tokens.
        self.amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        self.special_tokens = {"[PAD]": 0, "[UNK]": 1}
        self.token_to_id = self.special_tokens.copy()
        for i, aa in enumerate(self.amino_acids):
            self.token_to_id[aa] = i + len(self.special_tokens)
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.pad_token_id = 0
        self.vocab_size = len(self.token_to_id)

    def __call__(self, sequences, return_tensors=None, padding="do_not_pad", truncation=True, max_length=256):
        tokenized = []
        for seq in sequences:
            seq_ids = []
            for char in seq:
                char = char.upper()
                seq_ids.append(self.token_to_id.get(char, self.token_to_id["[UNK]"]))
            if truncation and len(seq_ids) > max_length:
                seq_ids = seq_ids[:max_length]
            tokenized.append(seq_ids)
        return {"input_ids": tokenized}


# Model size and parameter counts.
def calculate_model_metrics(model):
    param_count = sum(p.numel() for p in model.parameters())
    param_count_k = param_count / 1000
    
    temp_path = "./model/cnn_lite/temp_model.pt"
    torch.save(model.state_dict(), temp_path)
    model_size_mb = os.path.getsize(temp_path) / (1024 * 1024)
    os.remove(temp_path)

    return {
        "Param Count (k)": round(param_count_k, 3),
        "Model Size (MB)": round(model_size_mb, 4)
    }


# Cumulative performance curves.
def plot_cumulative_curves(model_probs, dataset_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    def prepare_cumulative_data(probs):
        if len(probs) == 0:
            return np.array([]), np.array([])
        probs_sorted = np.sort(probs)
        cum_percent = np.arange(1, len(probs_sorted) + 1) / len(probs_sorted) * 100
        return probs_sorted, cum_percent

    model_probs_sorted, model_cum_percent = prepare_cumulative_data(model_probs)

    plt.figure(figsize=(10, 6))
    if len(model_probs_sorted) > 0:
        plt.plot(model_probs_sorted, model_cum_percent,
                 label='Model - All Samples',
                 color='#F18F01', linewidth=2.5, linestyle='-')

    plt.title(f'Cumulative Distribution of Prediction Scores - {dataset_name}',
              fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Positive Probability', fontsize=12)
    plt.ylabel('Cumulative Sample Percentage (%)', fontsize=12)
    plt.xlim(0, 1)
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11, loc='lower right')
    plt.tight_layout()

    save_filename = f'cumulative_curve_{dataset_name.replace(" ", "_").replace("/", "_")}.png'
    save_path = os.path.join(save_dir, save_filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Cumulative curve saved to: {save_path}")


# Dataset wrapper for peptide sequences.
class PeptideDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
        if len(self.sequences) != len(self.labels):
            raise ValueError("The numbers of sequences and labels must match")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx], len(self.sequences[idx])


# CSV loading and validation.
def load_csv_data(csv_path):
    if not os.path.exists(csv_path): raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required_cols = ['id', 'seq', 'label']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"Missing required CSV columns: {missing}")

    ids, sequences, labels = [], [], []
    for _, row in df.iterrows():
        id_val, seq = row['id'], str(row['seq']).strip().upper()
        if len(seq) == 0:
            print(f"Skipping empty sequence (id: {id_val})")
            continue
        try:
            label = int(row['label'])
            if label not in (0, 1): raise ValueError(f"Label must be 0 or 1")
        except Exception as e:
            print(f"Skipping invalid label ({e}): {row['label']} (id: {id_val})")
            continue
        ids.append(id_val)
        sequences.append(seq)
        labels.append(label)

    total, positive = len(labels), sum(labels)
    print(f"Loaded {os.path.basename(csv_path)}: {total} samples ({positive} positive, {positive / total * 100:.2f}%)")
    return ids, sequences, labels


# Lightweight CNN architecture.
class LightweightModel(nn.Module):
    """Lightweight CNN for direct training on amino acid tokens."""

    def __init__(self, vocab_size=22, embed_dim=192, hidden_dim=256, max_seq_len=256):
        super().__init__()
        self.max_seq_len = max_seq_len

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(0.1772219731237647)

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1772219731237647),
            nn.MaxPool1d(2),

            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1772219731237647),
            nn.MaxPool1d(2),

            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1772219731237647),
            nn.AdaptiveMaxPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1772219731237647),
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
        x = self.embedding(input_ids)  # [batch, seq_len, embed_dim]
        x = self.embed_dropout(x)
        x = x.transpose(1, 2)  # [batch, embed_dim, seq_len]
        x = self.feature_extractor(x)  # [batch, hidden_dim, 1]
        x = x.squeeze(-1)  # [batch, hidden_dim]
        return self.classifier(x)


# CNN training and evaluation.
class ModelTrainer:
    def __init__(self, device=device):
        self.device = device
        self.model = None
        self.tokenizer = None

    def init_model(self):
        print(f"\nInitializing the lightweight model")
        self.tokenizer = AminoAcidTokenizer()
        vocab_size = self.tokenizer.vocab_size
        self.model = LightweightModel(vocab_size=vocab_size).to(self.device)

        # Report parameter count and model size.
        model_metrics = calculate_model_metrics(self.model)
        print(f"Model parameters: {model_metrics['Param Count (k)']:.1f}k (about {model_metrics['Param Count (k)'] / 1000:.1f}M)")
        print(f"Model size: {model_metrics['Model Size (MB)']:.4f} MB")
        print("Model initialized")

    def _tokenize(self, sequences):
        encoded = self.tokenizer(
            sequences, return_tensors=None, padding="do_not_pad", truncation=True, max_length=256
        )
        return encoded["input_ids"]

    def _compute_metrics(self, y_true, y_pred, y_prob, inference_time, sample_count):
        y_prob_pos = y_prob[:, 1] if y_prob.shape[1] == 2 else y_prob
        
        metrics = {
            'Sample Count': sample_count,
            'Inference Time (s)': round(inference_time, 4),
            'Inference Time per Sample (ms)': round(inference_time * 1000 / sample_count, 4)
        }

        try:
            metrics['Accuracy'] = round(accuracy_score(y_true, y_pred), 4)
        except:
            metrics['Accuracy'] = 0
        
        try:
            metrics['Precision'] = round(precision_score(y_true, y_pred), 4)
        except:
            metrics['Precision'] = 0
        
        try:
            metrics['Recall'] = round(recall_score(y_true, y_pred), 4)
        except:
            metrics['Recall'] = 0
        
        try:
            metrics['F1-Score'] = round(f1_score(y_true, y_pred), 4)
        except:
            metrics['F1-Score'] = 0

        try:
            metrics['MCC'] = round(matthews_corrcoef(y_true, y_pred), 4)
        except:
            metrics['MCC'] = 0

        try:
            metrics['AUC'] = round(roc_auc_score(y_true, y_prob_pos), 4)
        except:
            metrics['AUC'] = 0

        return metrics

    def _compute_loss_and_metrics(self, model, dataloader):
        loss_fn = nn.CrossEntropyLoss()
        total_loss = 0.0
        sample_count = 0
        all_labels = []
        all_preds = []
        all_probs = []

        model.eval()
        with torch.no_grad():
            for input_ids, labels in dataloader:
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                outputs = model(input_ids)
                loss = loss_fn(outputs, labels)
                total_loss += loss.item() * input_ids.size(0)
                sample_count += input_ids.size(0)

                probs = F.softmax(outputs, dim=-1)
                preds = torch.argmax(probs, dim=-1)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        avg_loss = total_loss / sample_count
        metrics = self._compute_metrics(all_labels, all_preds, np.array(all_probs), 0, sample_count)

        return avg_loss, metrics

    def train(self, train_seqs, train_labels, val_seqs, val_labels,
              epochs=50, lr=0.0008666341887480871, patience=8, weight_decay=5.16073525705775e-06):
        print(f"\nStarting training...")

        os.makedirs(os.path.dirname(val1_output), exist_ok=True)
        os.makedirs(cumulative_curve_dir, exist_ok=True)

        train_input_ids = self._tokenize(train_seqs)
        val_input_ids = self._tokenize(val_seqs)

        class TokenizedDataset(Dataset):
            def __init__(self, input_ids, labels):
                self.input_ids = input_ids
                self.labels = labels

            def __len__(self):
                return len(self.input_ids)

            def __getitem__(self, idx):
                return self.input_ids[idx], self.labels[idx]

        def collate_fn(batch):
            input_ids_list, labels_list = zip(*batch)
            batch_max_len = max(len(ids) for ids in input_ids_list)
            pad_token_id = self.tokenizer.pad_token_id
            padded_input_ids = [ids + [pad_token_id] * (batch_max_len - len(ids)) for ids in input_ids_list]
            return torch.tensor(padded_input_ids), torch.tensor(labels_list)

        train_dataset = TokenizedDataset(train_input_ids, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
        
        val_dataset = TokenizedDataset(val_input_ids, val_labels)
        val_loader = DataLoader(val_dataset, batch_size=64, collate_fn=collate_fn)

        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        loss_fn = nn.CrossEntropyLoss()
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model = None
        best_val_f1_metrics = 0.0
        for epoch in range(epochs):
            start_time = time.time()

            # Train for one epoch.
            self.model.train()
            train_loss = 0.0
            train_all_labels = []
            train_all_preds = []
            train_all_probs = []
            
            for input_ids, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(input_ids)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * input_ids.size(0)

                # Collect training predictions for epoch-level metrics.
                probs = F.softmax(outputs, dim=-1)
                preds = torch.argmax(probs, dim=-1)
                train_all_labels.extend(labels.cpu().numpy())
                train_all_preds.extend(preds.detach().cpu().numpy())
                train_all_probs.extend(probs.detach().cpu().numpy())

            train_loss /= len(train_loader.dataset)
            train_metrics = self._compute_metrics(train_all_labels, train_all_preds, np.array(train_all_probs), 0, len(train_loader.dataset))

            # Evaluate on the validation set.
            val_loss, val_metrics = self._compute_loss_and_metrics(self.model, val_loader)
            epoch_time = time.time() - start_time

            print(f"\nEpoch {epoch + 1}/{epochs} | Elapsed: {epoch_time:.2f}s")
            print(f"  Train: Loss {train_loss:.4f}, Accuracy {train_metrics['Accuracy']:.4f}, Precision {train_metrics['Precision']:.4f}, Recall {train_metrics['Recall']:.4f}, F1-Score {train_metrics['F1-Score']:.4f}, AUC {train_metrics['AUC']:.4f}")
            print(f"  Validation: Loss {val_loss:.4f}, Accuracy {val_metrics['Accuracy']:.4f}, Precision {val_metrics['Precision']:.4f}, Recall {val_metrics['Recall']:.4f}, F1-Score {val_metrics['F1-Score']:.4f}, AUC {val_metrics['AUC']:.4f}")

            # Step the learning rate scheduler using validation loss.
            scheduler.step(val_loss)

            # Stop early if validation loss stops improving.
            if val_loss < best_val_loss or val_metrics['F1-Score'] > best_val_f1_metrics:
                best_val_loss = val_loss
                best_val_f1_metrics = val_metrics['F1-Score']
                epochs_no_improve = 0
                best_model = self.model.state_dict()
                # Save the checkpoint with the lowest validation loss so far.
                torch.save(best_model, "./model/cnn_lite/best_model.pt")
                print(f"  Validation loss improved; saved best_model.pt")
            else:
                epochs_no_improve += 1
                print(f"  Validation loss has not improved for {epochs_no_improve} consecutive epochs")

                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered")
                    break

        if best_model:
            self.model.load_state_dict(best_model)
        # Save the final weights after training.
        torch.save(self.model.state_dict(), "./model/final_model.pt")
        print("Final model saved to final_model.pt")

        print("\nTraining complete")

    def evaluate(self, seqs, labels, ids, dataset_name):
        print(f"\nEvaluating the model on {dataset_name}...")

        input_ids = self._tokenize(seqs)

        class EvalDataset(Dataset):
            def __init__(self, input_ids, labels):
                self.input_ids = input_ids
                self.labels = labels

            def __len__(self):
                return len(self.input_ids)

            def __getitem__(self, idx):
                return self.input_ids[idx], self.labels[idx]

        def collate_fn(batch):
            input_ids_list, labels_list = zip(*batch)
            batch_max_len = max(len(ids) for ids in input_ids_list)
            pad_token_id = self.tokenizer.pad_token_id
            padded_input_ids = [ids + [pad_token_id] * (batch_max_len - len(ids)) for ids in input_ids_list]
            return torch.tensor(padded_input_ids), torch.tensor(labels_list)

        eval_dataset = EvalDataset(input_ids, labels)
        eval_loader = DataLoader(eval_dataset, batch_size=64, collate_fn=collate_fn)

        self.model.eval()
        all_labels = []
        all_preds = []
        all_probs = []
        start_time = time.time()

        with torch.no_grad():
            for input_ids, labels in tqdm(eval_loader, desc=f"Inference: {dataset_name}"):
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(input_ids)
                probs = F.softmax(outputs, dim=-1)
                preds = torch.argmax(probs, dim=-1)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        inference_time = time.time() - start_time
        sample_count = len(seqs)

        # Compute metrics across the full evaluation set.
        metrics = self._compute_metrics(all_labels, all_preds, np.array(all_probs), inference_time, sample_count)

        print(f"\n{dataset_name} evaluation results:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

        # Plot the cumulative performance curves.
        plot_cumulative_curves([p[1] for p in all_probs], dataset_name, cumulative_curve_dir)

        return {
            "labels": all_labels,
            "predictions": all_preds,
            "probabilities": np.array(all_probs),
            "metrics": metrics,
            "ids": ids
        }


def save_prediction_results(result, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame({
        "id": result["ids"],
        "label": result["labels"],
        "prediction": result["predictions"],
        "prob_0": result["probabilities"][:, 0],
        "prob_1": result["probabilities"][:, 1]
    })
    df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")


def main():
    print("=" * 50)
    print("Starting model training and evaluation")
    print("=" * 50)

    # Load the training, validation, and test data.
    print("\nLoading training data...")
    train_ids, train_seqs, train_labels = load_csv_data(train_csv)
    print("\nLoading validation set 1...")
    val1_ids, val1_seqs, val1_labels = load_csv_data(val1_csv)
    print("\nLoading validation set 2...")
    val2_ids, val2_seqs, val2_labels = load_csv_data(val2_csv)
    print("\nLoading validation set 3...")
    val3_ids, val3_seqs, val3_labels = load_csv_data(val3_csv)
    print("\nLoading validation set 4...")
    val4_ids, val4_seqs, val4_labels = load_csv_data(val4_csv)
    print("\nLoading test data...")
    test_ids, test_seqs, test_labels = load_csv_data(test_csv)

    # Initialize the trainer.
    trainer = ModelTrainer(device=device)
    trainer.init_model()

    # Train the model.
    trainer.train(
        train_seqs=train_seqs,
        train_labels=train_labels,
        val_seqs=val1_seqs,
        val_labels=val1_labels,
        epochs=50,
        lr=0.0008666341887480871,
        patience=10
    )

    # Evaluate the model and save the results.
    print("\n" + "=" * 50)
    print("Evaluating all datasets")
    print("=" * 50)

    # Validation set 1.
    val1_result = trainer.evaluate(val1_seqs, val1_labels, val1_ids, "val1_new_val")
    save_prediction_results(val1_result, val1_output)

    # Validation set 2.
    val2_result = trainer.evaluate(val2_seqs, val2_labels, val2_ids, "val2_bagel4_1_val")
    save_prediction_results(val2_result, val2_output)

    # Validation set 3.
    val3_result = trainer.evaluate(val3_seqs, val3_labels, val3_ids, "val3_bagel4_2_val")
    save_prediction_results(val3_result, val3_output)

    # Validation set 4.
    val4_result = trainer.evaluate(val4_seqs, val4_labels, val4_ids, "val4_bagel4_3_val")
    save_prediction_results(val4_result, val4_output)

    # Test set.
    test_result = trainer.evaluate(test_seqs, test_labels, test_ids, "test_val_test")
    save_prediction_results(test_result, test_output)

    print("\n" + "=" * 50)
    print("Evaluation complete")
    print("=" * 50)


if __name__ == "__main__":
    main()
