import torch
from torch import Tensor
from torch.nn import functional as F
from semseg.models.base import BaseModel
# from semseg.models.heads import SegFormerHead
from semseg.models.heads import BgFormerHead
from semseg.models.modules import BGM
from fvcore.nn import flop_count_table, FlopCountAnalysis


class GFBNEXT(BaseModel):
    def __init__(self, backbone: str = 'GFBNEXT-B0', num_classes: int = 25,
                 modals: list = ['img', 'depth', 'event', 'lidar']) -> None:
        super().__init__(backbone, num_classes, modals)
        self.decode_head = BgFormerHead(self.backbone.channels, 256 if 'B0' in backbone or 'B1' in backbone else 512,
                                         num_classes)
        self.apply(self._init_weights)
        # self.bgm = BGM(dims=[64, 128, 320, 512])    # "B2"
        # self.bat = BAT(dims=[64, 128, 320, 512])

    def forward(self, x: list) -> list:
        y = self.backbone(x)
        # edg = self.bgm(y[0])
        y, y_aux, edg = self.decode_head(y)
        y = F.interpolate(y, size=x[0].shape[2:], mode='bilinear', align_corners=False)  # to original image shape
        y_aux = F.interpolate(y_aux, size=x[0].shape[2:], mode='bilinear', align_corners=False)  # to original image shape
        edg = F.interpolate(edg, size=x[0].shape[2:], mode='bilinear', align_corners=False)
        return y, y_aux, edg

    def init_pretrained(self, pretrained: str = None) -> None:
        if pretrained:
            if self.backbone.num_modals > 0:
                load_dualpath_model(self.backbone, pretrained)
            else:
                checkpoint = torch.load(pretrained, map_location='cpu')
                if 'state_dict' in checkpoint.keys():
                    checkpoint = checkpoint['state_dict']
                if 'model' in checkpoint.keys():
                    checkpoint = checkpoint['model']
                msg = self.backbone.load_state_dict(checkpoint, strict=False)
                print(msg)


def load_dualpath_model(model, model_file):
    extra_pretrained = None
    if isinstance(extra_pretrained, str):
        raw_state_dict_ext = torch.load(extra_pretrained, map_location=torch.device('cpu'))
        if 'state_dict' in raw_state_dict_ext.keys():
            raw_state_dict_ext = raw_state_dict_ext['state_dict']
    if isinstance(model_file, str):
        raw_state_dict = torch.load(model_file, map_location=torch.device('cpu'))
        if 'model' in raw_state_dict.keys():
            raw_state_dict = raw_state_dict['model']
    else:
        raw_state_dict = model_file

    state_dict = {}
    for k, v in raw_state_dict.items():
        if k.find('patch_embed') >= 0:
            state_dict[k] = v
        elif k.find('block') >= 0:
            state_dict[k] = v
        elif k.find('norm') >= 0:
            state_dict[k] = v

    if isinstance(extra_pretrained, str):
        for k, v in raw_state_dict_ext.items():
            if k.find('patch_embed1.proj') >= 0:
                state_dict[k.replace('patch_embed1.proj', 'extra_downsample_layers.0.proj.module')] = v
            if k.find('patch_embed2.proj') >= 0:
                state_dict[k.replace('patch_embed2.proj', 'extra_downsample_layers.1.proj.module')] = v
            if k.find('patch_embed3.proj') >= 0:
                state_dict[k.replace('patch_embed3.proj', 'extra_downsample_layers.2.proj.module')] = v
            if k.find('patch_embed4.proj') >= 0:
                state_dict[k.replace('patch_embed4.proj', 'extra_downsample_layers.3.proj.module')] = v

            if k.find('patch_embed1.norm') >= 0:
                for i in range(model.num_modals):
                    state_dict[k.replace('patch_embed1.norm', 'extra_downsample_layers.0.norm.ln_{}'.format(i))] = v
            if k.find('patch_embed2.norm') >= 0:
                for i in range(model.num_modals):
                    state_dict[k.replace('patch_embed2.norm', 'extra_downsample_layers.1.norm.ln_{}'.format(i))] = v
            if k.find('patch_embed3.norm') >= 0:
                for i in range(model.num_modals):
                    state_dict[k.replace('patch_embed3.norm', 'extra_downsample_layers.2.norm.ln_{}'.format(i))] = v
            if k.find('patch_embed4.norm') >= 0:
                for i in range(model.num_modals):
                    state_dict[k.replace('patch_embed4.norm', 'extra_downsample_layers.3.norm.ln_{}'.format(i))] = v
            elif k.find('block') >= 0:
                state_dict[k.replace('block', 'extra_block')] = v
            elif k.find('norm') >= 0:
                state_dict[k.replace('norm', 'extra_norm')] = v

    msg = model.load_state_dict(state_dict, strict=False)
    del state_dict


if __name__ == '__main__':
    modals = ['img']
    # modals = ['img', 'depth', 'event', 'lidar']
    model = GFBNEXT('GFBNBGM-B2', 25, modals)
    model.init_pretrained('checkpoints/pretrained/segformer/mit_b2.pth')
    x = [torch.zeros(1, 3, 512, 512)]
    y, edge = model(x)
    print(y.shape)
