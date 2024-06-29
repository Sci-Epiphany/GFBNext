import torch
from torch import nn, Tensor
from torch.nn import functional as F
import einops
from semseg.models.layers import DropPath
from semseg.models.modules.cfm import ExternalAttentionRectifyModule as EARM
from semseg.models.modules.cfm import CrossFusionModule as CFM
import functools

from semseg.models.modules.mspa import MSPABlock


class Attention(nn.Module):
    def __init__(self, dim, head, sr_ratio):
        super().__init__()
        self.head = head
        self.sr_ratio = sr_ratio
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio, sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x = x.permute(0, 2, 1).reshape(B, C, H, W)
            x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)

        k, v = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class MLP(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x: Tensor, H, W) -> Tensor:
        return self.fc2(F.gelu(self.dwconv(self.fc1(x), H, W)))


class PatchEmbed(nn.Module):
    def __init__(self, c1=3, c2=32, patch_size=7, stride=4):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, patch_size, stride, patch_size // 2)  # padding=(ps[0]//2, ps[1]//2)
        self.norm = nn.LayerNorm(c2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class PatchEmbedParallel(nn.Module):
    def __init__(self, c1=3, c2=32, patch_size=7, stride=4, padding=0, num_modals=4):
        super().__init__()
        self.proj = ModuleParallel(nn.Conv2d(c1, c2, patch_size, stride, padding))    # padding=(ps[0]//2, ps[1]//2)
        self.norm = LayerNormParallel(c2, num_modals)

    def forward(self, x: list) -> list:
        x = self.proj(x)
        _, _, H, W = x[0].shape
        x = self.norm(x)
        return x, H, W


class ModuleParallel(nn.Module):
    def __init__(self, module):
        super(ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel):
        return [self.module(x) for x in x_parallel]

class ConvLayerNorm(nn.Module):
    """Channel first layer norm
    """
    def __init__(self, normalized_shape, eps=1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class LayerNormParallel(nn.Module):
    def __init__(self, num_features, num_modals=4):
        super(LayerNormParallel, self).__init__()
        # self.num_modals = num_modals
        for i in range(num_modals):
            setattr(self, 'ln_' + str(i), ConvLayerNorm(num_features, eps=1e-6))

    def forward(self, x_parallel):
        return [getattr(self, 'ln_' + str(i))(x) for i, x in enumerate(x_parallel)]



class Block(nn.Module):
    def __init__(self, dim, head, sr_ratio=1, dpr=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, head, sr_ratio)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * 4))

    def forward(self, x: Tensor, H, W) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

class PredictorLG(nn.Module):
    """ Image to Patch Embedding from DydamicVit
    """

    def __init__(self, embed_dim=384, num_modals=4):
        super().__init__()
        self.num_modals = num_modals
        self.score_nets = nn.ModuleList([nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1),
            nn.Softmax(dim=-1)
        ) for _ in range(num_modals)])

    def forward(self, x):
        x = [self.score_nets[i](x[i]) for i in range(self.num_modals)]
        return x


class PredictorConv(nn.Module):
    def __init__(self, embed_dim=384, num_modals=4):
        super().__init__()
        self.num_modals = num_modals
        self.score_nets = nn.ModuleList([nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, groups=(embed_dim)),
            nn.Conv2d(embed_dim, 1, 1),
            nn.Sigmoid()
        )for _ in range(num_modals)])

    def forward(self, x):
        B, C, H, W = x[0].shape
        x_ = [torch.zeros((B, 1, H, W)) for _ in range(self.num_modals)]
        for i in range(self.num_modals):
            x_[i] = self.score_nets[i](x[i])
        return x_


mit_settings = {
    'B0': [[32, 64, 160, 256], [2, 2, 2, 2]],  # [embed_dims, depths]
    'B1': [[64, 128, 320, 512], [2, 2, 2, 2]],
    'B2': [[64, 128, 320, 512], [3, 4, 6, 3]],
    'B3': [[64, 128, 320, 512], [3, 4, 18, 3]],
    'B4': [[64, 128, 320, 512], [3, 8, 27, 3]],
    'B5': [[64, 128, 320, 512], [3, 6, 40, 3]]
}


class GFBNBGM(nn.Module):
    def __init__(self, model_name: str = 'B0', modals: list = ['rgb', 'depth', 'event', 'lidar']):
        super().__init__()
        assert model_name in mit_settings.keys(), f"Model name should be in {list(mit_settings.keys())}"
        embed_dims, depths = mit_settings[model_name]
        extra_depths = depths  # for fusion branch
        self.modals = modals[1:] if len(modals) > 1 else []  # remove rgb
        self.num_modals = len(self.modals)
        drop_path_rate = 0.1
        self.channels = embed_dims
        norm_cfg = dict(type='BN', requires_grad=True)

        # patch_embed
        self.patch_embed1 = PatchEmbed(3, embed_dims[0], 7, 4)
        self.patch_embed2 = PatchEmbed(embed_dims[0], embed_dims[1], 3, 2)
        self.patch_embed3 = PatchEmbed(embed_dims[1], embed_dims[2], 3, 2)
        self.patch_embed4 = PatchEmbed(embed_dims[2], embed_dims[3], 3, 2)

        if self.num_modals > 0:
            self.extra_downsample_layers = nn.ModuleList([
                PatchEmbedParallel(3, embed_dims[0], 7, 4, 7//2, self.num_modals),
                *[PatchEmbedParallel(embed_dims[i], embed_dims[i+1], 3, 2, 3//2, self.num_modals) for i in range(3)]
            ])

        if self.num_modals > 0:
            self.extra_patch_embed1 = PatchEmbed(3, embed_dims[0], 7, 4)
            self.extra_patch_embed2 = PatchEmbed(embed_dims[0], embed_dims[1], 3, 2)
            self.extra_patch_embed3 = PatchEmbed(embed_dims[1], embed_dims[2], 3, 2)
            self.extra_patch_embed4 = PatchEmbed(embed_dims[2], embed_dims[3], 3, 2)

        if self.num_modals > 1:
            self.extra_score_predictor = nn.ModuleList([PredictorConv(embed_dims[i], self.num_modals) for i in range(len(depths))])
            # self.extra_score_predictor = nn.ModuleList(
            #     [PredictorLG(embed_dims[i], self.num_modals) for i in range(len(depths))])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        cur = 0
        self.block1 = nn.ModuleList([Block(embed_dims[0], 1, 8, dpr[cur + i]) for i in range(depths[0])])
        self.norm1 = nn.LayerNorm(embed_dims[0])
        if self.num_modals > 0:
            self.extra_block1 = nn.ModuleList([MSPABlock(embed_dims[0], mlp_ratio=8, drop_path=dpr[cur+i], norm_cfg=norm_cfg) for i in range(extra_depths[0])])
            self.extra_norm1 = ConvLayerNorm(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(embed_dims[1], 2, 4, dpr[cur + i]) for i in range(depths[1])])
        self.norm2 = nn.LayerNorm(embed_dims[1])
        if self.num_modals > 0:
            self.extra_block2 = nn.ModuleList([MSPABlock(embed_dims[1], mlp_ratio=8, drop_path=dpr[cur+i], norm_cfg=norm_cfg) for i in range(extra_depths[1])])
            self.extra_norm2 = ConvLayerNorm(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(embed_dims[2], 5, 2, dpr[cur + i]) for i in range(depths[2])])
        self.norm3 = nn.LayerNorm(embed_dims[2])
        if self.num_modals > 0:
            self.extra_block3 = nn.ModuleList([MSPABlock(embed_dims[2], mlp_ratio=8, drop_path=dpr[cur+i], norm_cfg=norm_cfg) for i in range(extra_depths[2])])
            self.extra_norm3 = ConvLayerNorm(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(embed_dims[3], 8, 1, dpr[cur + i]) for i in range(depths[3])])
        self.norm4 = nn.LayerNorm(embed_dims[3])
        if self.num_modals > 0:
            self.extra_block4 = nn.ModuleList([MSPABlock(embed_dims[3], mlp_ratio=8, drop_path=dpr[cur+i], norm_cfg=norm_cfg) for i in range(extra_depths[3])])
            self.extra_norm4 = ConvLayerNorm(embed_dims[3])

        if self.num_modals > 0:
            num_heads = [1, 2, 5, 8]
            self.EARMs = nn.ModuleList([
                EARM(dim=embed_dims[0], reduction=1),
                EARM(dim=embed_dims[1], reduction=1),
                EARM(dim=embed_dims[2], reduction=1),
                EARM(dim=embed_dims[3], reduction=1)])
            self.CFMs = nn.ModuleList([
                CFM(dim=embed_dims[0], reduction=1, num_heads=num_heads[0], kernel=7, norm_layer=nn.BatchNorm2d),
                CFM(dim=embed_dims[1], reduction=1, num_heads=num_heads[1], kernel=7, norm_layer=nn.BatchNorm2d),
                CFM(dim=embed_dims[2], reduction=1, num_heads=num_heads[2], kernel=7, norm_layer=nn.BatchNorm2d),
                CFM(dim=embed_dims[3], reduction=1, num_heads=num_heads[3], kernel=5, norm_layer=nn.BatchNorm2d)])

    # ---Hard selection
    def tokenselect(self, x_ext, module):
        # x_scores = module(x_ext)
        # # select tokens according to the max score of multiple modals, regarding H, W
        # x_stack = torch.stack(x_ext, dim=-1)  # B, N, C, N_modals
        # B, N, C, N_modals = x_stack.shape
        # x_scores = torch.stack(x_scores, dim=-1)  # B, N, 1, N_modals
        # x_index = torch.argmax(x_scores, dim=-1, keepdim=True)  # B, N, 1, N_modals
        # x_index = einops.repeat(x_index, 'b n 1 m  -> b n c m', c=C)  # B, C, H, W, N_modals
        # # --- token selection
        # x_select = x_stack.gather(-1, x_index)
        # return x_select.squeeze(-1)  # B, C, H, W
        x_scores = module(x_ext)
        for i in range(len(x_ext)):
            x_ext[i] = x_scores[i] * x_ext[i] + x_ext[i]
        x_f = functools.reduce(torch.max, x_ext)
        return x_f

    def forward(self, x: list) -> list:
        x_cam = x[0]
        if self.num_modals > 0:
            x_ext = x[1:]
        B = x_cam.shape[0]
        outs = []
        # outs_aux = []
        # stage 1
        x_cam, H, W = self.patch_embed1(x_cam)
        for blk in self.block1:
            x_cam = blk(x_cam, H, W)
        x_cam = self.norm1(x_cam).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        if self.num_modals > 0:
            x_ext, _, _ = self.extra_downsample_layers[0](x_ext)
            x_f = self.tokenselect(x_ext, self.extra_score_predictor[0]) if self.num_modals > 1 else x_ext[0]
            for blk in self.extra_block1:
                x_f = blk(x_f)
            x_f = self.extra_norm1(x_f)
            # --- CFM
            x_cam, x_f = self.EARMs[0](x_cam, x_f)
            x_fused = self.CFMs[0](x_cam, x_f)
            # x_fused = x_cam + x_f
            outs.append(x_fused)
            # outs_aux.append(x_aux)
            x_ext = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2) + x_f for x_ in x_ext] if self.num_modals > 1 else [
                x_f]
        else:
            outs.append(x_cam)

        # stage 2
        x_cam, H, W = self.patch_embed2(x_cam)
        for blk in self.block2:
            x_cam = blk(x_cam, H, W)
        x_cam = self.norm2(x_cam).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        if self.num_modals > 0:
            # x_ext = [self.extra_patch_embed2(x_ext_)[0] for x_ext_ in x_ext]
            x_ext, _, _ = self.extra_downsample_layers[1](x_ext)
            x_f = self.tokenselect(x_ext, self.extra_score_predictor[1]) if self.num_modals > 1 else x_ext[0]
            for blk in self.extra_block2:
                x_f = blk(x_f)
            x_f = self.extra_norm2(x_f)
            # --- FFM
            x_cam, x_f = self.EARMs[1](x_cam, x_f)
            x_fused = self.CFMs[1](x_cam, x_f)
            # x_fused = x_cam + x_f
            outs.append(x_fused)
            # outs_aux.append(x_aux)
            x_ext = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2) + x_f for x_ in x_ext] if self.num_modals > 1 else [
                x_f]
        else:
            outs.append(x_cam)

        # stage 3
        x_cam, H, W = self.patch_embed3(x_cam)
        for blk in self.block3:
            x_cam = blk(x_cam, H, W)
        x_cam = self.norm3(x_cam).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        if self.num_modals > 0:
            x_ext, _, _ = self.extra_downsample_layers[2](x_ext)
            x_f = self.tokenselect(x_ext, self.extra_score_predictor[2]) if self.num_modals > 1 else x_ext[0]
            for blk in self.extra_block3:
                x_f = blk(x_f)
            x_f = self.extra_norm3(x_f)
            # --- FFM
            x_cam, x_f = self.EARMs[2](x_cam, x_f)
            x_fused = self.CFMs[2](x_cam, x_f)
            # x_fused = x_cam + x_f
            outs.append(x_fused)
            # outs_aux.append(x_aux)
            x_ext = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2) + x_f for x_ in x_ext] if self.num_modals > 1 else [
                x_f]
        else:
            outs.append(x_cam)

        # stage 4
        x_cam, H, W = self.patch_embed4(x_cam)
        for blk in self.block4:
            x_cam = blk(x_cam, H, W)
        x_cam = self.norm4(x_cam).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        if self.num_modals > 0:
            x_ext, _, _ = self.extra_downsample_layers[3](x_ext)
            x_f = self.tokenselect(x_ext, self.extra_score_predictor[3]) if self.num_modals > 1 else x_ext[0]
            # for blk in self.extra_block4:
            #     x_f = blk(x_f, H, W)
            # x_f = self.extra_norm4(x_f).reshape(B, H, W, -1).permute(0, 3, 1, 2)
            for blk in self.extra_block4:
                x_f = blk(x_f)
            x_f = self.extra_norm4(x_f)
            # --- FFM
            x_cam, x_f = self.EARMs[3](x_cam, x_f)
            x_fused= self.CFMs[3](x_cam, x_f)
            # x_fused = x_cam + x_f
            outs.append(x_fused)
            # outs_aux.append(x_aux)
            # x_ext = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2) + x_f for x_ in x_ext] if self.num_modals > 1 else [x_f]
        else:
            outs.append(x_cam)

        return outs


if __name__ == '__main__':
    modals = ['img']
    # modals = ['img', 'depth', 'event', 'lidar']
    # x = [torch.zeros(1, 3, 1024, 1024), torch.ones(1, 3, 1024, 1024), torch.ones(1, 3, 1024, 1024)*2, torch.ones(1, 3, 1024, 1024) *3]
    # modals = ['img', 'depth']
    # x = [torch.zeros(1, 3, 1024, 1024), torch.ones(1, 3, 1024, 1024)]
    x = [torch.zeros(1, 3, 1024, 1024)]
    model = GFBNBGM('B2', modals)
    outs = model(x)
    print(model)
    for y in outs:
        print(y.shape)

    # print(flop_count_table(FlopCountAnalysis(model, x)))


