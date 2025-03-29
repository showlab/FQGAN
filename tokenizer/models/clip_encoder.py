import torch
import torch.nn as nn
from transformers import CLIPVisionConfig, CLIPImageProcessor, CLIPVisionModel


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('vq_mean', torch.Tensor([0.5, 0.5, 0.5])[None, :, None, None])
        self.register_buffer('vq_std', torch.Tensor([0.5, 0.5, 0.5])[None, :, None, None])

        self.register_buffer('openai_mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073])[None, :, None, None])
        self.register_buffer('openai_std', torch.Tensor([0.26862954, 0.26130258, 0.27577711])[None, :, None, None])

    def forward(self, inp):
        inp = inp * self.vq_std + self.vq_mean

        assert inp.max() < 10       # ensure the input image is in the scale of 0~1
        inp = torch.nn.functional.interpolate(inp, (224, 224))

        return (inp - self.openai_mean) / self.openai_std


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = -1
        self.select_feature = "cls"
        self.load_model()
        self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

        self.scaling_layer = ScalingLayer()

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        elif self.select_feature == 'cls':
            image_features = image_features[:, :1]
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def extract_patch_feats(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @torch.no_grad()
    def extract_last_feats(self, images):
        image_features = self.clip_model_full.get_image_features(images)

        return image_features

    @torch.no_grad()
    def extract_image_cls_feature(self, images):
        images = self.scaling_layer(images)  # -1 ~ 1?
        visual_feat_vls = self.extract_patch_feats(images)

        return visual_feat_vls

    def forward(self, images, semantic_feat):
        images = self.scaling_layer(images)     # -1 ~ 1?
        visual_feat_vls = self.extract_patch_feats(images)
        semantic_feat_cls = semantic_feat[:, :1]

        visual_feat_vls = visual_feat_vls.squeeze()
        semantic_feat_cls = semantic_feat_cls.squeeze()

        visual_feat_normalized = visual_feat_vls / visual_feat_vls.norm(dim=-1, keepdim=True)
        semantic_feat_normalized = semantic_feat_cls / semantic_feat_cls.norm(dim=-1, keepdim=True)
        loss_mat = (semantic_feat_normalized @ visual_feat_normalized.detach().T).exp()
        loss_diag = loss_mat.diag()
        loss_denom = loss_mat.sum(1)
        loss_InfoNCE = -(loss_diag / loss_denom).log().mean()

        return loss_InfoNCE

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

