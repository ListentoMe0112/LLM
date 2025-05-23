import torch
import torch.nn as nn
from einops import einsum, pack, unpack
class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None): 
        """
            Construct a linear transformation module. This function should accept the following parameters:
            in_features: int final dimension of the input
            out_features: int final dimension of the output
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.param = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.trunc_normal_(self.param, mean=0.0, std= 2 / (self.in_features + self.out_features), a=-3, b=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor :
        """
            Apply the linear transformation to the input. 
        """
        return einsum(x, self.param, "... d_in, d_out d_in -> ... d_out")

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None): 
        """
        Construct an embedding module. This function should accept the following parameters
        Args:
            num_embeddings (_type_): int Size of the vocabulary
            embedding_dim (_type_):  int sieze of d_model
            device (_type_, optional): _description_. Defaults to None.
            dtype (_type_, optional): _description_. Defaults to None.
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding_matrix = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        nn.init.trunc_normal_(self.embedding_matrix, mean = 0.0, std = 2.0, a = -3, b = 3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor :
        """ 
            Lookup the embedding vectors
        """
        return self.embedding_matrix[token_ids]
        

        