
import json
from tokenizers import ByteLevelBPETokenizer
from transformers import RobertaTokenizer, RobertaModel
import torch
import torch.nn as nn
import numpy as np



### This is the PE by manual design ! 
    
class PositionalEncoding2D(nn.Module):
    def __init__(self, embed_size, max_lines, max_line_length):
        super(PositionalEncoding2D, self).__init__()
        # Generate separate positional encodings for lines and tokens
        self.line_pos_encoding = self.get_positional_encoding(max_lines, embed_size)
        self.token_pos_encoding = self.get_positional_encoding(max_line_length, embed_size)
        self.norm = nn.LayerNorm(embed_size)

    def forward(self, token_embeddings, line_numbers, token_positions):
        if not isinstance(token_embeddings, torch.Tensor):
          token_embeddings = torch.tensor(token_embeddings, dtype=torch.float, device="cpu")
        
        # Use line_numbers and token_positions indices to fetch positional encodings
        line_numbers = torch.as_tensor(line_numbers, dtype=torch.long, device=token_embeddings.device)
        token_positions = torch.as_tensor(token_positions, dtype=torch.long, device=token_embeddings.device)

        # Assuming line_numbers and token_positions are zero-indexed and within the valid range
        # Clamp line_numbers and token_positions to ensure they are within bounds
        line_numbers = line_numbers.clamp(0, self.line_pos_encoding.size(0) - 1)
        token_positions = token_positions.clamp(0, self.token_pos_encoding.size(0) - 1)

        line_encoding = self.line_pos_encoding[line_numbers]
        token_encoding = self.token_pos_encoding[token_positions]
        # line_encoding = line_encoding.unsqueeze(0)  # Add batch dimension: [1, seq_length, embed_size]
        # line_encoding = line_encoding.expand(token_embeddings.size(0), -1, -1)
        # token_encoding = token_encoding.unsqueeze(0)
        # token_encoding = token_encoding.expand(token_embeddings.size(0), -1, -1)
        print(token_embeddings.shape)
        print(line_encoding.shape)
        print(token_encoding.shape)
        # Add the positional encodings to the token embeddings
        pos_encoded = token_embeddings + line_encoding + token_encoding
        return self.norm(pos_encoded)

    def get_positional_encoding(self, max_len, embed_size):
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, embed_size, 2) * -(np.log(10000.0) / embed_size))
        pe = np.zeros((max_len, embed_size))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return torch.from_numpy(pe).float()
    


    

def get_positional_encoding(max_seq_len, embed_size):
    """Generate sinusoidal positional encodings."""
    positional_encoding = np.array([
        [pos / np.power(10000, 2 * (i // 2) / embed_size) for i in range(embed_size)]
        for pos in range(max_seq_len)])

    positional_encoding[:, 0::2] = np.sin(positional_encoding[:, 0::2])  # dim 2i
    positional_encoding[:, 1::2] = np.cos(positional_encoding[:, 1::2])  # dim 2i+1

    return torch.from_numpy(positional_encoding).type(torch.FloatTensor)

# Example usage

# pos_encoding_layer = PositionalEncoding(embed_size, max_seq_len)

# # Example token embeddings (batch size, seq len, embed size)
# token_embeddings = chunk_text(snippet)

# # Apply positional encoding and normalization
# encoded = pos_encoding_layer(token_embeddings)

# relative_pos_encodings = rel_pos_encoding_layer(token_embeddings)
# print(encoded)
# print(relative_pos_encodings)