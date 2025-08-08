import torch
from einops import rearrange, einsum

class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal = False):
        # q = torch.randn(batch_size, n_queries, D, device=device, requires_grad=True)
        batch_size = q.shape[0]
        n_q = q.shape[-2]
        n_k = k.shape[-2]
        b_q = 16
        b_k = 16
        t_q = n_q // b_q
        t_k = n_k // b_k

        d = q.shape[-1]
        _const = 1 / torch.sqrt(torch.tensor(d, dtype = torch.float32))

        o = torch.zeros_like(q)
        l = torch.zeros(q.shape[0], q.shape[1])

        for i in range(1, t_q+1):
            q_i = q[:, b_q * (i-1) : b_q * i] # [b_q, d]
            o_i = torch.zeros(batch_size, b_q, d) # [b_q, d]
            l_i = torch.zeros(batch_size, b_q) #[batch_size, b_q]
            m_i = torch.zeros(batch_size, b_q) #[b_q]
            m_i = torch.fill(m_i, -1e10)
            for j in range(1, t_k+1):
                m_pre = m_i
                l_pre = l_i

                k_j = k[:, b_k * (j-1) : b_k * j] #[b_k, d]
                v_j = v[:, b_k * (j-1) : b_k * j] #[b_k, d]
                s_i_j = einsum(q_i, rearrange(k_j, "... b_k d -> ... d b_k"), "... b_q d, ... d b_k -> ... b_q b_k") * _const #[b_q, b_k]
                m_i = torch.max(m_i, torch.max(s_i_j, dim = -1).values) #[b_q]
                p_i_j = torch.exp(s_i_j - rearrange(m_i, "... -> ... 1")) #[b_q, b_k]
                l_i = torch.exp(m_pre - m_i) * l_pre + torch.sum(p_i_j, dim = -1)
                o_i = einsum(torch.diag_embed(torch.exp(m_pre - m_i)) , o_i, "batch_size b_q b_q, batch_size b_q d -> batch_size b_q d") + einsum(p_i_j , v_j, "batch_size b_q b_k, batch_size b_k d -> batch_size b_q d")

            o_i = einsum(torch.diag_embed(l_i).inverse(), o_i, "batch_size b_q b_q, batch_size b_q d -> batch_size b_q d")
            l_i = m_i + torch.log(l_i)

            o[:, b_q * (i-1) : b_q * i] = o_i
            l[:, b_q * (i-1) : b_q * i] = l_i

        ctx.save_for_backward(l)
        return o

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError

# Your launch grid should be set as (Tq , batch_size), meaning each Triton program instance
# will load only elements from a single batch index, and only read/write to a single query tile
# of Q, O, and L.
# The kernel should only have a single loop, which will iterate key tiles 1 ≤ j ≤ Tk.
@triton.jit
def flash_fwd_kernel(
        Q_ptr, K_ptr, V_ptr,
        O_ptr, L_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,
        stride_lb, stride_lq,
        N_QUERIES, N_KEYS,
        scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
    ):

    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0,0),
        block_shape=(K_TILE_SIZE, D),
        order=(1,0),
    )
    
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0,0),
        block_shape=(K_TILE_SIZE, D),
        order=(1,0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(0,0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1,0),
    )
    
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    o_i = tl.zeros(N_QUERIES, D)
    l_i = tl.zeros(N_QUERIES,)
    m_i = tl.full((N_QUERIES,), -1e10, tl.float32)

    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        m_pre = m_i
        # Extract the key and value slices for current j
        k_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero") # [b_k, d]
        v_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero") # [b_k, d]

        # Compute s_i_j using dot product
        s_i_j = tl.dot(q_i, tl.transpose(k_j)) * _const  # [b_q, b_k]
        
        # Update m_i: max of m_i and max(s_i_j, dim=-1)
        m_i = tl.max(m_i, tl.max(s_i_j, axis=-1))  # [b_q]
        
        # Compute p_i_j (exponential scaling for softmax)
        p_i_j = tl.exp(s_i_j - m_i[:, None])  # Broadcast max(m_i) for stable softmax [b_q, b_k]
        
        # Update l_i
        l_i = tl.exp(m_pre - m_i) * l_i + tl.sum(p_i_j, axis=-1)  # Update l_i based on m_pre and m_i

        # Broadcasting exp_m to scale o_i
        o_i = o_i * tl.exp(m_pre - m_i)[:, None] + tl.dot(p_i_j, v_j)  # [batch_size, b_q, d] * [b_q, 1] + [b_q, b_k] * [b_k, d]
        k_j.advanced((0, K_TILE_SIZE))
        v_j.advanced((0, K_TILE_SIZE))
    
    # Final steps after the loop
    o_i = tl.dot(tl.inverse(tl.diag(l_i)), o_i)  # Inverse diag multiplication
    l_i = m_i + tl.log(l_i)  # Final log update
    tl.store(O_block_ptr, o_i, boundary_check=(0,1))
    tl.store(L_block_ptr, l_i, boundary_check=(0,))


