import torch
import torch.nn as nn
from transformers import Dinov2Model


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('vq_mean', torch.Tensor([0.5, 0.5, 0.5])[None, :, None, None])
        self.register_buffer('vq_std', torch.Tensor([0.5, 0.5, 0.5])[None, :, None, None])

        self.register_buffer('dinov2_mean', torch.Tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer('dinov2_std', torch.Tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def forward(self, inp):
        inp = inp * self.vq_std + self.vq_mean

        assert inp.max() < 10       # ensure the input image is in the scale of 0~1
        inp = torch.nn.functional.interpolate(inp, (224, 224))

        return (inp - self.dinov2_mean) / self.dinov2_std


class DinoVisionTower(nn.Module):
    def __init__(self, vision_tower):
        super(DinoVisionTower, self).__init__()

        """try to extract an image resolution and interpolation res from the model name string

        valid model names:
            facebook/dinov2-small
            facebook/dinov2-base
            facebook/dinov2-large
            facebook/dinov2-giant
            facebook/dinov2-giant-imagenet1k-1-layer

        res pattern: <model_name>-res<res>-interp<interp>

        eg: facebook/dinov2-small-res518-interp224
        """

        self.vision_tower_name = vision_tower
        self.load_model()
        self.select_feature = "cls"
        self.scaling_layer = ScalingLayer()

    def load_model(self, device_map=None):
        self.vision_tower = Dinov2Model.from_pretrained(self.vision_tower_name)
        """ValueError: Dinov2Model does not support `device_map='auto'`. To implement support, the model class needs to implement the `_no_split_modules` attribute."""
        self.vision_tower._no_split_modules = ["Dinov2SwiGLUFFN"]

        # Assign the output channels of the projection convolution as the hidden size
        self._hidden_size = self.vision_tower.embeddings.patch_embeddings.projection.out_channels
        # Assign the first value of the stride of the projection convolution as the patch size
        self._patch_size = self.vision_tower.embeddings.patch_embeddings.projection.stride[0]

        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def feature_select(self, outputs):
        sequence_output = outputs["last_hidden_state"]  # batch_size, sequence_length, hidden_size

        if self.select_feature == 'cls_patch':
            image_features = sequence_output
        elif self.select_feature == 'patch':
            image_features = sequence_output[:, 1:]
        elif self.select_feature == 'cls':
            image_features = sequence_output[:, 0]
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def extract_patch_feats(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0))
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype),
                                                   output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

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
    def device(self):
        return self.vision_tower.device

    @property
    def dtype(self):
        return self.vision_tower.dtype


if __name__ == '__main__':
    model = DinoVisionTower("facebook/dinov2-base")
    data_1 = torch.randn(size=(3, 256, 256))
    data_2 = torch.randn(size=(3, 512))
    feat = model(data_1, data_2)
