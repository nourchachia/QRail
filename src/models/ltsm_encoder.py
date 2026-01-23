import torch
import torch.nn as nn

class LSTMEncoder(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=2, output_size=64):
        """
        Args:
            input_size (int): Number of features per time step. 
                              Default 4: [delay, progress, speed_limit, is_hub]
            hidden_size (int): Internal memory size of the LSTM.
            num_layers (int): Number of stacked LSTM layers (2 adds depth for complex patterns).
            output_size (int): Size of the final embedding vector.
        """
        super(LSTMEncoder, self).__init__()
        
        # The Core LSTM Layer
        # batch_first=True means input shape is (batch, seq, features)
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=0.2  # Prevents overfitting during training
        )
        
        # Final projection layer to get exactly 64 dimensions
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Activation function (optional, but good for embeddings)
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, sequence_length, input_size)
               Example: (32 batches, 10 time_steps, 4 features)
        """
        # LSTM returns: output, (hidden_state, cell_state)
        # We only need the final hidden state (h_n) to capture the full sequence summary
        _, (h_n, _) = self.lstm(x)
        
        # h_n shape is (num_layers, batch, hidden_size)
        # We take the state from the last layer (layer index -1)
        last_layer_hidden = h_n[-1] 
        
        # Project to final embedding space
        embedding = self.fc(last_layer_hidden)
        
        return self.tanh(embedding)