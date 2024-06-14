import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

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
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    def get_shape(self, x):
        if isinstance(x, torch.Tensor):
            return x.shape
        elif isinstance(x, tuple):
            x = [sub_x for sub_x in x]
            x = torch.cat(x, dim=0)
            return x.shape

    @torch.no_grad()
    def forward(self, images, output_attentions=False):
        if type(images) is list:
            image_features = []
            feature_attentions = [] if output_attentions else None
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                                                      output_hidden_states=True,
                                                      output_attentions=output_attentions)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
                print(f"image shape: ".format(image.shape))
                print(f"hidden state shape: ".format(self.get_shape(image_forward_out.hidden_states)))
                print(f"attention shape: ".format(self.get_shape(image_forward_out.attentions)))
                print(f"image feature shape: ".format(image_feature.shape))
                # if output_attentions:
                #     feature_attention = image_forward_out.attentions[self.select_layer, :, 0, 1:]
                #     feature_attentions.append(feature_attention)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype),
                                                   output_hidden_states=True,
                                                   output_attentions=output_attentions)
            print(f"images shape: ".format(images.shape))
            print(f"hidden states shape: ".format(self.get_shape(image_forward_outs.hidden_states)))
            print(f"attentions shape: ".format(self.get_shape(image_forward_outs.attentions)))
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
            print(f"image features shape: ".format(image_features.shape))
            # if output_attentions:
            #     feature_attentions = image_forward_outs.attentions[self.select_layer, :, 0, 1:]

        return image_features

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
