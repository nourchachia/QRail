"""
LSTM Cascade Encoder - Model 2
Captures how delays ripple and propagate through time.
Input: [batch, 10 timesteps, 4 features] -> Output: [batch, 64-dim embedding]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMEncoder(nn.Module):
    def __init__(
        self, 
        input_size=4, 
        hidden_size=64, 
        num_layers=2, 
        output_size=64,
        bidirectional=False,
        dropout=0.2,
        use_attention=True
    ):
        """
        The 'Cascade Encoder' that captures delay momentum.
        
        Args:
            input_size (int): 4 features [delay, progress, speed, hub].
            hidden_size (int): Internal LSTM memory size (64).
            num_layers (int): Stacked LSTM layers (2 for deeper patterns).
            output_size (int): Final embedding vector size (64).
            bidirectional (bool): If True, processes sequence forwards & backwards.
            dropout (float): Dropout rate between LSTM layers.
            use_attention (bool): Apply attention over sequence outputs.
        """
        super(LSTMEncoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.num_directions = 2 if bidirectional else 1
        
        # Input normalization (helps with convergence)
        self.input_norm = nn.LayerNorm(input_size)
        
        # Core LSTM Layer
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,  # Input: (batch, seq, features)
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention mechanism (learns which timesteps matter most)
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_size * self.num_directions, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1)
            )
        
        # Final projection layer
        lstm_output_size = hidden_size * self.num_directions
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, output_size),
            nn.LayerNorm(output_size),
            nn.ReLU()
        )
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for better gradient flow."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # LSTM forget gate bias trick (helps with long sequences)
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)
    
    def forward(self, x, return_sequence=False):
        """
        Args:
            x: Tensor of shape [batch_size, sequence_length=10, input_size=4]
            return_sequence: If True, returns all timestep outputs (for visualization)
        
        Returns:
            embedding: [batch_size, output_size=64]
        """
        batch_size, seq_len, _ = x.shape
        
        # Normalize input features
        x = self.input_norm(x)
        
        # Pass through LSTM
        # lstm_out: [batch, seq, hidden * num_directions]
        # h_n: [num_layers * num_directions, batch, hidden]
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        if self.use_attention:
            # Attention-weighted pooling over sequence
            # attention_scores: [batch, seq, 1]
            attention_scores = self.attention(lstm_out)
            attention_weights = F.softmax(attention_scores, dim=1)
            
            # Weighted sum: [batch, hidden * num_directions]
            context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        else:
            # Use final hidden state from last layer
            if self.bidirectional:
                # Concatenate forward and backward final states
                h_n = h_n.view(self.num_layers, 2, batch_size, self.hidden_size)
                context_vector = torch.cat([h_n[-1, 0], h_n[-1, 1]], dim=1)
            else:
                context_vector = h_n[-1]
        
        # Generate final embedding
        embedding = self.fc(context_vector)
        
        if return_sequence:
            return embedding, lstm_out, attention_weights if self.use_attention else None
        
        return embedding
    
    def get_attention_weights(self, x):
        """
        Extract attention weights for visualization.
        Returns: [batch, seq_len] tensor showing which timesteps model focuses on.
        """
        with torch.no_grad():
            _, _, attention_weights = self.forward(x, return_sequence=True)
            if attention_weights is not None:
                return attention_weights.squeeze(-1)
            return None


# Quick Test
if __name__ == "__main__":
    print("=" * 60)
    print("LSTM Encoder Test - Neural Rail Conductor")
    print("=" * 60)
    
    # Configuration
    BATCH_SIZE = 8
    SEQ_LENGTH = 10
    FEATURES = 4  # [delay, progress, speed, is_hub]
    
    # Create model variants
    models = {
        "Standard": LSTMEncoder(bidirectional=False, use_attention=False),
        "Bidirectional": LSTMEncoder(bidirectional=True, use_attention=False),
        "With Attention": LSTMEncoder(bidirectional=False, use_attention=True),
    }
    
    # Generate synthetic telemetry (simulating expanding delay)
    dummy_data = []
    for i in range(SEQ_LENGTH):
        delay = i * 1.5          # Delay growing from 0 to 13.5 minutes
        progress = 0.1 * i       # Train progressing
        speed = 120 - i * 5      # Speed decreasing
        is_hub = 1 if i > 7 else 0  # Approaching a hub
        dummy_data.append([delay, progress, speed, is_hub])
    
    input_tensor = torch.tensor([dummy_data] * BATCH_SIZE, dtype=torch.float32)
    
    print(f"\n📊 Input Shape: {input_tensor.shape}")
    print(f"   Batch: {BATCH_SIZE}, Sequence: {SEQ_LENGTH}, Features: {FEATURES}\n")
    
    # Test each variant
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            embedding = model(input_tensor)
            params = sum(p.numel() for p in model.parameters())
            
            print(f"✅ {name:20} | Output: {embedding.shape} | Parameters: {params:,}")
            
            # Show attention if available
            if "Attention" in name:
                attn = model.get_attention_weights(input_tensor[0:1])
                if attn is not None:
                    print(f"   Attention Focus: {attn[0].tolist()}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! Model ready for training.")
    print("=" * 60)