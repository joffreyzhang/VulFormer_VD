import torch
import torch.nn as nn


class DegreeEmbedding(nn.Module):
    """
    Two-Dimensional Degree Embeddings (ED) for integrating in-degree and out-degree
    information from Integrated Code Dependency Graphs (ICDGs) into the Transformer model.

    ED = ED_in + ED_out

    where ED_in and ED_out ∈ R^(n × d_model) are learnable embedding vectors
    specified by the in-degree and out-degree respectively.
    """

    def __init__(self, embed_size, max_degree=100):
        """
        Initialize the DegreeEmbedding module.

        Args:
            embed_size (int): The embedding dimension (d_model)
            max_degree (int): Maximum degree value to support (default: 100)
        """
        super(DegreeEmbedding, self).__init__()
        self.embed_size = embed_size
        self.max_degree = max_degree

        # Learnable embeddings for in-degree and out-degree
        self.in_degree_embeddings = nn.Embedding(max_degree + 1, embed_size)
        self.out_degree_embeddings = nn.Embedding(max_degree + 1, embed_size)

        # Layer normalization for stability
        self.norm = nn.LayerNorm(embed_size)

    def forward(self, token_embeddings, in_degrees, out_degrees):
        """
        Apply degree embeddings to token embeddings.

        Args:
            token_embeddings (torch.Tensor): Token embeddings of shape (batch_size, seq_len, embed_size)
            in_degrees (torch.Tensor): In-degree values for each token of shape (batch_size, seq_len)
            out_degrees (torch.Tensor): Out-degree values for each token of shape (batch_size, seq_len)

        Returns:
            torch.Tensor: Token embeddings with degree information added, shape (batch_size, seq_len, embed_size)
        """
        # Clamp degrees to max_degree to avoid index out of bounds
        in_degrees = torch.clamp(in_degrees, 0, self.max_degree)
        out_degrees = torch.clamp(out_degrees, 0, self.max_degree)

        # Get degree embeddings
        in_degree_embeds = self.in_degree_embeddings(in_degrees)
        out_degree_embeds = self.out_degree_embeddings(out_degrees)

        # Combine: ED = ED_in + ED_out
        degree_embeds = in_degree_embeds + out_degree_embeds

        # Add degree embeddings to token embeddings
        degree_encoded = token_embeddings + degree_embeds

        return self.norm(degree_encoded)
