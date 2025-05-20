import torch
from torch import nn
import torch.nn.functional as F

from torch.nn.init import constant_
from einops import rearrange

from .rope import RotaryEmbedding
from .utils import RMSNorm, init_method_normal, print_mask


class SwiGLUFFN(nn.Module):
    def __init__(self, args, in_features, hidden_features = None, out_features = None, bias = True, drop = 0.):
        super().__init__()
        self.args = args
        self.drop = drop
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w2 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)
        self.dropout = nn.Dropout(drop)

        self.init_method = init_method_normal((1 / in_features) ** 0.5)
        self.init_method_w3 = init_method_normal((1 / hidden_features) ** 0.5)
        self.reset_parameters()

    def reset_parameters(self):
        self.init_method(self.w1.weight)
        self.init_method(self.w2.weight)
        self.init_method_w3(self.w3.weight)

        if self.w1.bias is not None:
            constant_(self.w1.bias, 0.)

        if self.w2.bias is not None:
            constant_(self.w2.bias, 0.)

        if self.w3.bias is not None:
            constant_(self.w3.bias, 0.)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)

        hidden = F.relu(x1).square() * x2
        hidden = self.dropout(hidden)
        w3 = self.w3(hidden)
        return w3

class MultiHeadAttention(nn.Module):
    def __init__(self, args, dim, rotary_emb, nheads,
            qkv_bias = False,
            qk_norm = False,
            attn_drop = 0.,
            causal = False,
        ):

        super().__init__()
        from lib.accelerator import AcumenAccelerator
        self.accelerator = AcumenAccelerator()

        self.args = args
        self.causal = causal
        self.attn_drop = attn_drop
        self.nheads = nheads
        self.head_dim = dim // nheads
        self.qkv_bias = qkv_bias
        self.dim = dim

        self.scale = 1 / self.head_dim

        self.to_q = nn.Linear(dim, dim, bias = qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias = qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias = qkv_bias)

        self.q_norm = RMSNorm(eps = 1e-5) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(eps = 1e-5) if qk_norm else nn.Identity()

        self.rotary_emb = rotary_emb

        self.proj_out = nn.Linear(dim, dim)

        # muP fan-in style initialization
        self.init_method = init_method_normal((1 / dim) ** 0.5)
        self.reset_parameters()

    def reset_parameters(self):
        self.init_method(self.proj_out.weight)
        constant_(self.proj_out.bias, 0.)

        self.init_method(self.to_q.weight)
        self.init_method(self.to_k.weight)
        self.init_method(self.to_v.weight)

        if self.qkv_bias:
            constant_(self.to_q.bias, 0.)
            constant_(self.to_k.bias, 0.)
            constant_(self.to_v.bias, 0.)

    def apply_attention(self, q, k, v, mask = None, apply_pos_emb = False):
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.nheads), (q, k, v))

        q = self.q_norm(q)
        k = self.k_norm(k)

        if apply_pos_emb and self.rotary_emb is not None:
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        if mask is not None:
            mask = mask.to(q.dtype)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask = mask,
            scale = self.scale,
            dropout_p = self.attn_drop if self.training else 0.0
        )
        out = rearrange(out, 'b h n d -> b n (h d)', h = self.nheads)
        out = self.proj_out(out)
        return out

class SelfAttention(MultiHeadAttention):

    @torch.compiler.disable
    def compute_mask(self, batch_size, seq_len, device, mask = None):
        if mask is None:
            mask = torch.zeros((batch_size, seq_len, seq_len), device = device).float()

        # mask is (b, n, n)
        mask = mask.unsqueeze(1) # (b, 1, n, n)
        mask = mask.expand(-1, self.nheads, -1, -1) # (b, h, n, n)

        return mask

    def forward(self, x, mask = None, causal_mask = None):
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        final_mask = self.compute_mask(x.shape[0], x.shape[1], x.device, mask = mask)
        
        if type(causal_mask) == bool and causal_mask:
            # also allow for causal mask to be passed in as a boolean
            causal_mask = torch.ones_like(final_mask)
            causal_mask = torch.tril(causal_mask)
            final_mask = final_mask.masked_fill(causal_mask == 0, -10000)
        
        elif causal_mask is not None:
            final_mask = final_mask.masked_fill(causal_mask < 0, -10000)

        out = self.apply_attention(q, k, v, mask = final_mask, apply_pos_emb = True)

        return out

class CrossAttention(MultiHeadAttention):

    @torch.compiler.disable
    def compute_mask(self, batch_size, tgt_len, context_len, device, mask = None, context_mask = None):
        if mask is None:
            mask = torch.zeros((batch_size, tgt_len, tgt_len), device = device).float()

        if context_mask is None:
            context_mask = torch.zeros((batch_size, context_len, context_len), device = device).float()

        # get the diagonals
        mask = mask.diagonal(dim1 = -1, dim2 = -2)
        context_mask = context_mask.diagonal(dim1 = -1, dim2 = -2)

        # combine masks
        final_mask = mask.unsqueeze(1).unsqueeze(-1) + context_mask.unsqueeze(1).unsqueeze(-2)
        final_mask = final_mask.expand(-1, self.nheads, -1, -1) # (b, h, n, m)

        return final_mask

    def forward(self, x, context, mask = None, context_mask = None, causal_context_mask = None):
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        mask = self.compute_mask(x.shape[0], x.shape[1], context.shape[1], x.device, mask = mask, context_mask = context_mask)
        if causal_context_mask is not None:
            mask = mask.masked_fill(causal_context_mask < 0, -10000)

        out = self.apply_attention(q, k, v, mask = mask, apply_pos_emb = True)
        return out

class TransformerEncoder(nn.Module):
    def __init__(self,
                args,
                dmodel,
                depth,
                nheads,
                dropout = 0.,
                has_context = False,
                context_position = None,
                compile_selfattention = False,
            ):
        super().__init__()
        self.args = args

        self.has_context = has_context

        self.norm = RMSNorm(eps = 1e-5)
        self.layers = nn.ModuleList([])

        self.nheads = nheads
        dim_head = dmodel // nheads
        
        self.rotary_emb = RotaryEmbedding(dim_head // 2)

        if context_position is None:
            context_position = list(range(depth))

        compile_fn = lambda x: x
        if compile_selfattention:
            compile_fn = torch.compile

        for block_idx in range(depth):
            self.layers.append(nn.ModuleList([
                compile_fn(SelfAttention(
                    args = self.args,
                    dim = dmodel,
                    rotary_emb = self.rotary_emb,
                    nheads = nheads,
                    qkv_bias = False,
                    qk_norm = True,
                )),
                CrossAttention(
                    args = self.args,
                    dim = dmodel,
                    rotary_emb = self.rotary_emb,
                    nheads = nheads,
                    qkv_bias = False,
                    qk_norm = True,
                ) if has_context and block_idx in context_position else None,
                compile_fn(SwiGLUFFN(
                    args,
                    in_features = dmodel,
                    hidden_features = 4 * dmodel,
                    out_features = dmodel,
                    drop = dropout,
                    bias = True
                ))
            ]))

        for n, m in self.named_modules():
            m.auto_name = n

    def forward(self, 
        x, 
        context = None, 
        mask = None, 
        causal_mask = True,
        context_mask = None, 
        causal_context_mask = None,
    ):
        
        if self.has_context:
            assert context is not None, 'context must be provided if model has context'

        for _, (attn, cross_attn, ff) in enumerate(self.layers):
            x = x + attn(
                self.norm(x), 
                mask = mask, 
                causal_mask = causal_mask,
            )
            
            if self.has_context and cross_attn is not None:
                x = x + cross_attn(
                    self.norm(x), context,
                    mask = mask,
                    context_mask = context_mask,
                    causal_context_mask = causal_context_mask
                )
            x = x + ff(self.norm(x))
        return x