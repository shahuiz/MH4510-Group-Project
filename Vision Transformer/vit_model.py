from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn

class DropPath(nn.Module):
# Drop paths per sample  (when applied in main path of residual blocks). It is used to reduce overfitting.
    def __init__(self, dropprob=None):
        super(DropPath, self).__init__()
        self.dropprob = dropprob

    def droppath(self,x, dropprob: float = 0., training: bool = False):
        if dropprob == 0. or not training:
            return x
        keep_prob = 1 - dropprob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with different dim tensors
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

    def forward(self, x):
        return self.droppath(x, self.dropprob, self.training)


# 2D Image to Patch Embedding ( Linear Projection of Flattened Patches part)
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channel=3, embedding_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_channel, embedding_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embedding_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape        # (H,W) is input's image resolution, B is batch size, C is num of channels
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

# Multiple-headed attention model
class Multi_headed_Attention(nn.Module):
    def __init__(self,
                 dim,   # input the dim of token
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Multi_headed_Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        #   [B,N1,C] -> [batch_size, num_patches + 1, total_embedding_dim]
        B, N1, C = x.shape

        # qkv(): size-> [batch_size, num_patches + 1, 3 * total_embedding_dim]
        # reshape(): size-> [batch_size, num_patches + 1, 3, num_heads, embedding_dim_per_head]
        # permute(): size -> [3, batch_size, num_heads, num_patches + 1, embedding_dim_per_head]
        qkv = self.qkv(x).reshape(B, N1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # qkv -> [batch_size, num_heads, num_patches + 1, embedding_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]


        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N1, C) # here x size-> [batch_size, num_patches + 1, total_embedding_dim]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# MLP(Multilayer Perceptron) as used in Vision Transformer and related networks
class MLP(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# Transformer Encoder block
class Encoder_Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Encoder_Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Multi_headed_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act=act, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

#ViT
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channel=3, num_classes=4,
                 embedding_dim=768, ViTdepth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, drop_ratio=0.05,
                 attn_drop_ratio=0.05, drop_path_ratio=0.1, embed_layer=PatchEmbedding, norm_layer=nn.LayerNorm,
                 ):
        """
            img_size : sie of input image
            patch_size : size of patch
            in_channel : number of input channels, rgb is 3
            num_classes : number of classes for classification head, here we have 4 classes
            embedding_dim: embedding dimension
            ViTdepth: depth of transformer, meaning the num of times that transformer encoder block is stacked
            num_heads : number of attention heads in multiple-headed attention
            mlp_ratio: ratio of mlp hidden dim to embedding dim, in ViT it is usually 4.
            qkv_bias (bool): if its value is True, then bias are enabled for qkv
            qk_scale : if it is set, model changes default qk scale value of head_dim^(-0.5)
            representation_size : if it is set, it is enabled and representation layer (pre-logits) is set to this value
            drop_ratio : Dropout rate in Linear Projection of Flattened Patches part
            attn_drop_ratio : Dropout rate in Multi-Headed Attention
            drop_path_ratio : Droppath rate in the Transformer Encoder Block
            embed_layer : patch embedding layer
            norm_layer: : normalization layer
        """
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embedding_dim = embedding_dim  # num_features for consistency with other models
        self.num_tokens =  1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act = nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_channel=in_channel, embedding_dim=embedding_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.dist_token =  None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embedding_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        droppathratio = [x.item() for x in torch.linspace(0, drop_path_ratio, ViTdepth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Encoder_Block(dim=embedding_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=droppathratio[i],
                  norm_layer=norm_layer, act=act)
            for i in range(ViTdepth)
        ])
        self.norm = norm_layer(embedding_dim)

        # Representation layer(pre-logit)
        if representation_size:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embedding_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier heads
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None


        # Weight initiation
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_weights)

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x


def _init_weights(m):
    # ViT weight initialization ; m: module
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def vit_base(num_classes: int = 4, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/16)
    weights (ImageNet-21k weight @ 224x224) got from official Google JAX:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embedding_dim=768,
                              ViTdepth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model



def vit_large(num_classes: int = 4, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/16)
    weights(ImageNet-21k weight @ 224x224) got from official Google JAX:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embedding_dim=1024,
                              ViTdepth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes,
                              )

    return model
