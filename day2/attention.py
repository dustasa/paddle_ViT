# ViT Online Class
# Author: Dr. Zhu
# Project: PaddleViT (https://github.com/BR-IDL/PaddleViT)
# 2021.11
import paddle
import paddle.nn as nn

paddle.set_device('cpu')


class Attention(nn.Layer):
    def __init__(self, embed_dim, num_heads, qkv_bias, qk_scale, dropout, attention_dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = int(embed_dim / num_heads)
        self.num_heads = num_heads
        self.all_head_dim = self.head_dim * num_heads
        self.qkv = nn.Linear(embed_dim,
                             self.all_head_dim * 3,
                             bias_attr=False if qkv_bias is False else None
                             )
        self.scale = self.head_dim ** -0.5 if qk_scale is None else qk_scale
        self.soft_max = nn.Softmax(-1)
        self.proj = nn.Linear(self.all_head_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(attention_dropout)

    def transpose_multi_head(self, x):
        # x: [B, num_patches, all_head_dim]
        new_shape = x.shape[:-1] + [self.num_heads, self.head_dim]
        x = x.reshape(new_shape)
        # x: [B, num_patches, num_heads, head_dim]
        x = paddle.transpose(x, [0, 2, 1, 3])
        # x: [B, num_heads, num_patches, head_dim]

        return x

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).chunk(3, -1)
        # [B, N, all_head_dim] * 3
        q, k, v = map(self.transpose_multi_head, qkv)
        # q,k,v = [B, num_heads, num_patches, head_dim]
        attn = paddle.matmul(q, k, transpose_y=True)  # q * k^T
        attn = self.scale * attn
        attn = self.soft_max(attn)
        attn_weight = attn

        attn = self.attn_dropout(attn)
        # attn = [B, num_heads, num_patches, num_patches]

        out = paddle.matmul(attn, v)  # softmax(scale*(q*k^T)) * v
        # out = out.transpose[0, 2, 1, 3]
        out = paddle.transpose(out, [0, 2, 1, 3])
        # attn = [B, num_patches, num_heads, head_dim]
        out = out.reshape([B, N, -1])
        out = self.proj(out)
        out = self.dropout(out)

        return out, attn_weight


def main():
    t = paddle.randn([4, 16, 96])
    # [4, 16, 96]
    print('input shape = ', t.shape)

    model = Attention(embed_dim=96, num_heads=8,
                      qkv_bias=False, qk_scale=None, dropout=0., attention_dropout=0.)
    print(model)

    out, attn_weights = model(t)
    print(out.shape)
    print(attn_weights.shape)


if __name__ == "__main__":
    main()
