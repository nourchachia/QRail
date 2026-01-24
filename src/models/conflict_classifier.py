import torch
import torch.nn as nn
import torch.nn.functional as F

class ConflictClassifier(nn.Module):
    """
    Multi-Layer Perceptron for predicting 8 types of operational conflicts.
    Input: 512-dim vector (concatenated GNN, LSTM, Semantic embeddings)
    Output: 8 probabilities (one per conflict type)
    Architecture: 3-layer MLP with dropout and batch normalization
    """
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=8, dropout=0.3):
        super(ConflictClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x  # logits; apply sigmoid at inference

if __name__ == "__main__":
    # Test script: random input
    model = ConflictClassifier()
    x = torch.randn(4, 512)  # batch of 4
    out = model(x)
    print("Output shape:", out.shape)  # Should be [4, 8]
    print("Sample output:", out)
