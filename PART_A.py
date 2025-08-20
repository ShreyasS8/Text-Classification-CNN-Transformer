import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from collections import Counter
import numpy as np
import random
from tqdm import tqdm
from sklearn.metrics import f1_score  # For micro-average F1 score

# -----------------------------
# Set Fixed Seed for Reproducibility
# -----------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -----------------------------
# 1. Preprocessing Functions
# -----------------------------
def tokenize(text):
    # A simple whitespace tokenizer (you can use more advanced tokenizers)
    return text.lower().split()

def build_vocab(texts, min_freq=1):
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    # Reserve index 0 for padding and index 1 for unknown words
    vocab = {"<pad>": 0, "<unk>": 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab

def encode_text(text, vocab):
    tokens = tokenize(text)
    return [vocab.get(token, vocab["<unk>"]) for token in tokens]

def pad_sequence(seq, max_len, pad_value=0):
    if len(seq) < max_len:
        return seq + [pad_value] * (max_len - len(seq))
    else:
        return seq[:max_len]

# -----------------------------
# 2. Dataset Classes
# -----------------------------
class TextDataset(Dataset):
    def __init__(self, csv_file, mode="train", vocab=None, max_len=50):
        # Read CSV with headers
        self.data = pd.read_csv(csv_file, header=0)
        self.mode = mode
        self.max_len = max_len

        # Define a mapping for string labels to integers
        label_to_int = {
            'business': 0,
            'tech': 1,
            'politics': 2,
            'sport': 3,
            'entertainment': 4
        }

        if mode == "train":
            self.sentences = self.data['Text'].astype(str).tolist()
            # Map string labels to integers
            self.labels = self.data['Category'].map(label_to_int).tolist()
        else:  # mode == "test"
            # Assume test CSV has columns 'ArticleId', 'Text', and 'Label - (business, tech, politics, sport, entertainment)'
            self.ids = self.data['ArticleId'].tolist()
            self.sentences = self.data['Text'].astype(str).tolist()
            # Map test labels from strings to integers
            self.labels = self.data['Label - (business, tech, politics, sport, entertainment)'].map(label_to_int).tolist()

        # Build vocab if not provided
        if vocab is None:
            self.vocab = build_vocab(self.sentences)
        else:
            self.vocab = vocab

        # Encode and pad sentences
        self.encoded_texts = [pad_sequence(encode_text(text, self.vocab), self.max_len)
                                for text in self.sentences]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        text_tensor = torch.tensor(self.encoded_texts[idx], dtype=torch.long)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.mode == "test":
            return text_tensor, label_tensor, self.ids[idx]
        else:
            return text_tensor, label_tensor

# -----------------------------
# 3. Modular C-LSTM with Optional Branches
# -----------------------------
class ModularTextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, lstm_hidden_dim,
                 num_classes, dropout=0.5, pad_idx=0, filter_sizes=[3,4,5],
                 use_cnn=True, use_lstm=True, use_embedding=False, use_attention=False):
        super(ModularTextClassifier, self).__init__()
        self.use_cnn = use_cnn
        self.use_lstm = use_lstm
        self.use_embedding = use_embedding
        self.use_attention = use_attention

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        if self.use_cnn:
            self.convs = nn.ModuleList([
                nn.Conv1d(in_channels=embedding_dim,
                          out_channels=num_filters,
                          kernel_size=fs)
                for fs in filter_sizes
            ])
            self.filter_sizes = filter_sizes

        if self.use_embedding:
            self.dense_branch = nn.Linear(embedding_dim, embedding_dim)

        if self.use_lstm:
            self.lstm = nn.LSTM(embedding_dim, lstm_hidden_dim, batch_first=True)
            if self.use_attention:
                self.attention = nn.Linear(lstm_hidden_dim, 1)
                self.lstm_dense = nn.Linear(lstm_hidden_dim, lstm_hidden_dim)

        fusion_dim = 0
        if self.use_cnn:
            fusion_dim += num_filters * len(filter_sizes)
        if self.use_embedding:
            fusion_dim += embedding_dim
        if self.use_lstm:
            fusion_dim += lstm_hidden_dim

        self.fusion_dropout = nn.Dropout(dropout)
        self.fusion_fc = nn.Linear(fusion_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        features = []

        if self.use_cnn:
            embedded_perm = embedded.permute(0, 2, 1)
            conv_outs = []
            for conv, fs in zip(self.convs, self.filter_sizes):
                conv_out = F.relu(conv(embedded_perm))
                pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2)).squeeze(2)
                conv_outs.append(pooled)
            cnn_feature = torch.cat(conv_outs, dim=1)
            features.append(cnn_feature)

        if self.use_embedding:
            avg_emb = embedded.mean(dim=1)
            dense_feature = F.relu(self.dense_branch(avg_emb))
            features.append(dense_feature)

        if self.use_lstm:
            lstm_out, (h_n, _) = self.lstm(embedded)
            if self.use_attention:
                attn_scores = torch.tanh(self.attention(lstm_out))
                attn_weights = F.softmax(attn_scores, dim=1)
                attn_applied = torch.sum(attn_weights * lstm_out, dim=1)
                lstm_feature = F.relu(self.lstm_dense(attn_applied))
            else:
                lstm_feature = h_n[-1]
            features.append(lstm_feature)

        fusion = torch.cat(features, dim=1)
        fusion = self.fusion_dropout(fusion)
        logits = self.fusion_fc(fusion)
        return logits

# -----------------------------
# 4. Training and Evaluation Functions
# -----------------------------
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    for batch in dataloader:
        texts, labels = batch[0], batch[1]
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * texts.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    micro_f1 = f1_score(all_labels, all_preds, average="micro")
    return avg_loss, micro_f1

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                texts, labels, _ = batch
            else:
                texts, labels = batch
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * texts.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    micro_f1 = f1_score(all_labels, all_preds, average="micro")
    return avg_loss, micro_f1

# -----------------------------
# 5. Main Training Script
# -----------------------------
if __name__ == '__main__':
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # File paths (adjust if necessary)
    train_csv = r"./Datasets/TrainData.csv"
    test_csv = r"./Datasets/TestLabels.csv"

    # Hyperparameters
    max_len = 500
    embedding_dim = 200
    num_filters = 150
    lstm_hidden_dim = 150
    num_classes = 5
    dropout = 0.5
    batch_size = 32
    num_epochs = 100
    learning_rate = 1e-3

    # -------------------------
    # 5.1 Choose Model Variant
    # -------------------------
    use_cnn = True
    use_lstm = True
    use_embedding = True  # Meta Embedding
    use_attention = True

    # -------------------------
    # 5.2 Create Dataset and Vocab
    # -------------------------
    train_dataset = TextDataset(train_csv, mode="train", max_len=max_len)
    vocab = train_dataset.vocab
    vocab_size = len(vocab)
    print("Vocabulary size:", vocab_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TextDataset(test_csv, mode="test", vocab=vocab, max_len=max_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # -------------------------
    # 5.3 Initialize Model, Loss, Optimizer, and LR Scheduler
    # -------------------------
    model = ModularTextClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_filters=num_filters,
        lstm_hidden_dim=lstm_hidden_dim,
        num_classes=num_classes,
        dropout=dropout,
        pad_idx=vocab["<pad>"],
        filter_sizes=[3, 4, 5],
        use_cnn=use_cnn,
        use_lstm=use_lstm,
        use_embedding=use_embedding,
        use_attention=use_attention
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # -------------------------
    # 5.4 Training Loop and Recording Metrics
    # -------------------------
    for epoch in range(num_epochs):
        train_loss, train_f1 = train_model(model, train_loader, optimizer, criterion, device)
        print("Epoch:",epoch+1, "Train Loss: ", train_loss,"Train F1: ", train_f1)

        test_loss, test_f1 = evaluate_model(model, test_loader, criterion, device)
        print("Epoch:", epoch+1,"Test Loss: ", test_loss, "Test F1: ", test_f1)

        scheduler.step()

    # -------------------------
    # 5.5 Final Evaluation (already done in the loop)
    # -------------------------
    print("\nFinal Evaluation on Test Set:")
    print("Test Loss: ", test_loss, "Test F1 Score: ", test_f1)