import torch
import math
from einops import einsum, reduce,rearrange, repeat

# def softmax(x : torch.Tensor, dim: int):
#     x_max = torch.max(x, dim = dim, keepdim=True)
#     x = x - x_max[0]
#     x_exp = torch.exp(x)
#     numerator = torch.sum(x_exp, dim = dim, keepdim=True)
#     return x_exp / numerator 
 

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute scaled dot-product attention.

    Args:
        Q: Query tensor of shape (batch_size, num_heads, seq_len_q, d_k)
        K: Key tensor of shape   (batch_size, num_heads, seq_len_k, d_k)
        V: Value tensor of shape (batch_size, num_heads, seq_len_k, d_v)
        mask: (optional) attention mask of shape (batch_size, 1, seq_len_q, seq_len_k)
              with 0 for valid positions and -inf (or a large negative number) for masked positions.

    Returns:
        output: Tensor of shape (batch_size, num_heads, seq_len_q, d_v)
        attention_weights: Tensor of shape (batch_size, num_heads, seq_len_q, seq_len_k)
    """
    pass
    # inv_d_k = torch.sqrt(torch.tensor(1 / K.shape[-1], dtype = torch.float32))
    # pre_softmax_score = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    # pre_softmax_score *= inv_d_k

    # if mask != None:
    #     mask = mask.to(torch.float32)
    #     mask = ((1 - mask) * 1e9)
    #     pre_softmax_score = pre_softmax_score - mask

    # score = softmax(pre_softmax_score, dim = -1)
    # attention = einsum(score, V, "... queries keys, ... keys d_v -> ... queries d_v")
    # return attention, score 

def test_attention(impl):
    Q = torch.tensor([[[[1., 0.], [0., 1.]]]])  # shape (1, 1, 2, 2)
    K = torch.tensor([[[[1., 0.], [0., 1.]]]])  # shape (1, 1, 2, 2)
    V = torch.tensor([[[[1., 2.], [3., 4.]]]])  # shape (1, 1, 2, 2)

    output, attn = impl(Q, K, V)

    print("Attention weights:\n", attn)
    print("Output:\n", output)

    target_attn = torch.tensor([[[[0.6698, 0.3302],
                                  [0.3302, 0.6698]]]])
    target_output = torch.tensor([[[[1.6605, 2.6605],
                                [2.3395, 3.3395]]]])


    print("Attention weights:\n", target_attn)
    print("Output:\n", target_output)
    
    assert torch.allclose(target_attn, attn, atol=1e-4)
    assert torch.allclose(target_output, output, atol=1e-4)

if __name__ == "__main__":
    test_attention(scaled_dot_product_attention)