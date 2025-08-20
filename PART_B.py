import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import math
import re
import random
from sklearn.metrics import f1_score
import copy

##############################################
# Data Cleaning, Advanced Tokenization, and Augmentation
##############################################

def clean_text(text):
    """
    Basic cleaning: lower-case and strip extra whitespace.
    You can extend this function with more cleaning steps.
    """
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def advanced_tokenize(text):
    """
    Advanced tokenization that separates words and punctuation.
    This returns a list of tokens (subword-level granularity).
    """
    text = clean_text(text)
    # This regex will keep punctuation as separate tokens.
    tokens = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
    return tokens

def build_vocab(sentences, min_freq=1):
    """
    Build a vocabulary dictionary from the list of sentences.
    Each unique token with frequency >= min_freq gets an index.
    Reserve index 0 for padding.
    """
    word_freq = {}
    for sentence in sentences:
        tokens = advanced_tokenize(sentence)
        for token in tokens:
            word_freq[token] = word_freq.get(token, 0) + 1
    # Start indexing from 1 since 0 is reserved for padding.
    vocab = {word: idx + 1 for idx, (word, freq) in enumerate(word_freq.items()) if freq >= min_freq}
    vocab["<UNK>"] = len(vocab) + 1
    return vocab

def encode_text(text, vocab):
    """
    Tokenizes the text using advanced tokenization and encodes each token.
    Unknown tokens are mapped to <UNK>.
    """
    tokens = advanced_tokenize(text)
    return [vocab.get(token, vocab["<UNK>"]) for token in tokens]

def pad_sequence(seq, max_len, padding_value=0):
    """
    Pads or truncates a sequence to a fixed length.
    """
    if len(seq) < max_len:
        return seq + [padding_value] * (max_len - len(seq))
    else:
        return seq[:max_len]

def random_deletion(sentence, p=0.1):
    """
    Data augmentation via random deletion:
    With probability p, remove each token.
    """
    tokens = sentence.split()
    if len(tokens) == 1:
        return sentence
    new_tokens = [token for token in tokens if random.random() > p]
    if len(new_tokens) == 0:
        new_tokens.append(random.choice(tokens))
    return " ".join(new_tokens)

##############################################
# Custom Dataset with Augmentation for Training
##############################################

class TextDataset(Dataset):
    def __init__(self, csv_file, mode="train", vocab=None, max_len=50, augment=False, aug_prob=0.1):
        # Read CSV with headers
        self.data = pd.read_csv(csv_file, header=0)
        self.mode = mode
        self.max_len = max_len
        self.augment = augment
        self.aug_prob = aug_prob

        # Mapping for string labels to integers
        label_to_int = {
            'business': 0,
            'tech': 1,
            'politics': 2,
            'sport': 3,
            'entertainment': 4
        }

        if mode == "train":
            self.sentences = self.data['Text'].astype(str).tolist()
            self.labels = self.data['Category'].map(label_to_int).tolist()
        else:  # mode == "test"
            self.ids = self.data['ArticleId'].tolist()
            self.sentences = self.data['Text'].astype(str).tolist()
            self.labels = self.data['Label - (business, tech, politics, sport, entertainment)'].map(label_to_int).tolist()

        # Build vocab if not provided
        if vocab is None:
            self.vocab = build_vocab(self.sentences)
        else:
            self.vocab = vocab

        # Pre-encode sentences; augmentation will be applied at __getitem__
        self.encoded_texts = [pad_sequence(encode_text(text, self.vocab), self.max_len)
                              for text in self.sentences]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        # For augmentation, re-encode the sentence if needed
        text = self.sentences[idx]
        if self.mode == "train" and self.augment and random.random() < self.aug_prob:
            # Apply random deletion augmentation
            text = random_deletion(text, p=self.aug_prob)
        encoded = pad_sequence(encode_text(text, self.vocab), self.max_len)
        text_tensor = torch.tensor(encoded, dtype=torch.long)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.mode == "test":
            return text_tensor, label_tensor, self.ids[idx]
        else:
            return text_tensor, label_tensor

##############################################
# Label Smoothing Loss
##############################################

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        # pred: (batch, num_classes)
        num_classes = pred.size(1)
        log_probs = F.log_softmax(pred, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))

##############################################
# Positional Encoding (Learnable version)
##############################################

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]

##############################################
# Custom Transformer Encoder Block
##############################################

class CustomTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, ff_hidden=512):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, d_model)
        )

    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)
        attn_output, _ = self.self_attn(x, x, x)
        x = self.layernorm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.layernorm2(x + self.dropout(ff_output))
        return x

##############################################
# Transformer-based Text Classifier with Encoder-only Architecture
##############################################

class TransformerTextClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_encoder_layers, num_classes,
                 max_seq_length=50, dropout=0.1, use_positional_encoding=True):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        if use_positional_encoding:
            self.pos_encoder = LearnablePositionalEncoding(max_seq_length, d_model)
        else:
            self.pos_encoder = None

        # Encoder: Stack of transformer encoder blocks
        self.encoder_blocks = nn.ModuleList(
            [CustomTransformerBlock(d_model, num_heads, dropout=dropout) for _ in range(num_encoder_layers)]
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, num_classes)

    def forward(self, src):
        # src: (batch_size, seq_len)
        embedded = self.embedding(src) * math.sqrt(self.d_model)
        if self.pos_encoder is not None:
            embedded = self.pos_encoder(embedded)
        # Prepare for encoder: shape (seq_len, batch_size, d_model)
        encoder_input = embedded.transpose(0, 1)
        for block in self.encoder_blocks:
            encoder_input = block(encoder_input)
        # Global context via mean pooling over the sequence length
        memory = encoder_input  # shape: (seq_len, batch_size, d_model)
        global_context = memory.mean(dim=0)  # shape: (batch_size, d_model)
        global_context = self.dropout(global_context)
        logits = self.fc_out(global_context)
        return logits

##############################################
# Training and Evaluation Functions
##############################################

def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for texts, labels in dataloader:
        texts = texts.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        preds = outputs.argmax(dim=1)
        total_loss += loss.item()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(dataloader), correct / total

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for texts, labels in dataloader:
            texts = texts.to(device)
            labels = labels.to(device)
            outputs = model(texts)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(dataloader), correct / total

def evaluate_on_test(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for texts, labels, _ in dataloader:
            texts = texts.to(device)
            labels = labels.to(device)
            outputs = model(texts)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    micro_f1 = f1_score(all_labels, all_preds, average='micro')
    return micro_f1

##############################################
# Main Experiment Function (without validation)
##############################################

def run_experiment(train_csv_file, num_encoder_layers, use_positional_encoding,
                   max_len=50, batch_size=32, num_epochs=25, lr=1e-3, augment=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load full training dataset (using entire dataset for training)
    train_dataset = TextDataset(train_csv_file, mode="train", max_len=max_len, 
                                augment=augment, aug_prob=0.1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Build model using vocabulary size from training
    vocab_size = max(train_dataset.vocab.values()) + 1
    num_classes = 5  # business, tech, politics, sport, entertainment
    model = TransformerTextClassifier(vocab_size=vocab_size,
                                      d_model=128,             # Model dimensionality
                                      num_heads=4,
                                      num_encoder_layers=num_encoder_layers,
                                      num_classes=num_classes,
                                      max_seq_length=max_len,
                                      dropout=0.5,
                                      use_positional_encoding=use_positional_encoding)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=9, gamma=0.9)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    
    # print(f"Training with {num_encoder_layers} encoder blocks, "
    #       f"{'with' if use_positional_encoding else 'without'} positional encoding, "
    #       f"data augmentation = {augment}.")

    for epoch in range(num_epochs):
        train_loss, train_acc = train_model(model, train_loader, optimizer, criterion, device)
        print("Epoch:" , epoch+1, "Train Loss: ", train_loss , "Train Acc: " , train_acc)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        # print("LR: "current_lr:.6f)
    
    return model, train_dataset.vocab

##############################################
# Main: Experiment Setup, Training, and Testing
##############################################

if __name__ == '__main__':
    # Global Paths
    TRAIN_CSV = './Datasets/TrainData.csv'
    TEST_CSV = './Datasets/TestLabels.csv'
    
    # Experiment parameters
    max_len = 500
    batch_size = 32
    num_epochs = 70
    lr = 0.001
    augment = True

    # Define the configuration to test
    num_encoder_layers = 2  # You can modify this as needed
    use_pos = True
    
    # print("\n" + "="*60)
    # print(f"Running experiment: Encoders_{num_encoder_layers}_PosEnc_{use_pos}")
    # print("="*60 + "\n")
    
    # Run training experiment with current configuration
    model, vocab = run_experiment(
        train_csv_file=TRAIN_CSV,
        num_encoder_layers=num_encoder_layers,
        use_positional_encoding=use_pos,
        max_len=max_len,
        batch_size=batch_size,
        num_epochs=num_epochs,
        lr=lr,
        augment=augment
    )
    
    # Evaluate on test dataset
    test_dataset = TextDataset(TEST_CSV, mode="test", vocab=vocab, max_len=max_len, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    micro_f1 = evaluate_on_test(model, test_loader, device)
    print("\nMicro-average F1 score on test set: ",  micro_f1)
