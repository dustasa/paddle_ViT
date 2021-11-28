import paddle
import paddle.nn as nn


class PatchEmbedding(nn.Layer):

    def __init__(self, patch_size=4, embed_dim=96):
        super().__init__()
        self.patch_embed = nn.Conv2D(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2)
        x = x.transpose([0, 2, 1])
        x = self.norm(x)
        return x


class PatchMerging(nn.Layer):

    def __init__(self, input_resolution, dim):
        super().__init__()
        self.resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        h, w = self.resolution
        b, _, c = x.shape

        x = x.reshape([b, h, w, c])
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 0::2, 1::2, :]
        x2 = x[:, 1::2, 0::2, :]
        x3 = x[:, 1::2, 1::2, :]

        x = paddle.concat([x0, x1, x2, x3], axis=-1)  # [B, h/2, w/2, 4c]
        x = x.reshape([b, -1, 4 * c])
        x = self.norm(x)
        x = self.reduction(x)

        return x


class Mlp(nn.Layer):
    def __init__(self, embed_dim, mlp_ratio, dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, int(embed_dim * mlp_ratio))
        self.fc2 = nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

def windows_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.reshape([B, H//window_size, window_size, W//window_size, window_size, C])
    x = x.transpose([0,1,3,2,4,5])
    x = x.reshape([[-1, window_size, window_size, c]])
    # [B* num_patches. ws. ws .c]

def window_reverse(widows, window_size, H, W):
    B = int(widows.shape[0] // (H // window_size * W /window_size))
    x = widows.
    return x


class WindowAttention(nn.Layer):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.dim_head = dim // num_heads
        self.scale = self.dim_head ** -0.5
        self.softmax = nn.Softmax(-1)
        self.qkv = nn.Linear(
            dim,
            dim*3
        )
        self.proj = nn.Linear(
            dim,
            dim
        )

    def transpose_multi_head(self, x):
        new_shape = x.shape[:-1] + [self.num_heads, self.head_dim]
        x = x.reshape(new_shape)
        x = paddle.transpose(x, [0, 2, 1, 3])
        return x

    def forward(self, x):
        B, N, C = self.shape
        qkv = self.qkv(x).chunk(3, -1)
        # [B, N, all_head_dim] * 3
        q, k, v = map(self.transpose_multi_head, qkv)

        q = q * self.scale
        atten = paddle.matmul(q, k ,transpose_y=True)
        atten = self.softmax(atten)

        out = paddle.matmul(atten, v)
        out = out.transpose([0, 2 , 1, 3])
        out = out.reshape([B, N, C])
        out = self.proj(out)
        return out


class SwinBlock(nn.Layer):
    def __init__(self,dim, input_resoluation, num_heads, window_size):
        super().__init__()



        self.attn_norm = nn.LayerNorm(dim)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = Mlp(dim)
    def forward(self, x):
        return x


def main():
    model = SwinBlock(dim=96, input_resoluation=[56,56], num_heads=4, window_size=7)
    print(model)
    paddle.summary(model, (4, 3, 224, 224))  # must be tuple


if __name__ == "__main__":
    main()
