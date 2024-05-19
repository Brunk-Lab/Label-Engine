from typing import Dict, List

from engine.core.utils.serialization import serialize
from engine.core.model.modules.unet_modules import *
from engine.core.model.inter_base_model import interModel
from engine.core.model.ops import DistMaps, BatchImageNormalize


class Encoder(nn.Module):
  def __init__(
    self,
    in_channels=3,
    out_channels=256
  ):
    super().__init__()
    assert out_channels % 16 == 0
    self.in_block   =   InputBlock(in_channels, out_channels//16)
    self.down_32    =   DownBlock(out_channels//16, 1)
    self.down_64    =   DownBlock(out_channels//8, 2)
    self.down_128   =   DownBlock(out_channels//4, 3)
    self.down_256   =   DownBlock(out_channels//2, 3)

  def forward(self, input):
    out16  =  self.in_block(input)
    out32  =  self.down_32(out16)
    out64  =  self.down_64(out32)
    out128 = self.down_128(out64)
    out256 = self.down_256(out128)
    return [out16, out32, out64, out128, out256]


class Decoder(nn.Module):
  def __init__(
      self,
      in_channels=256,
      out_channels=2,
  ):
    super().__init__()

    self.up_256     =   UpBlock(in_channels, in_channels, 3)
    self.up_128     =   UpBlock(in_channels, in_channels//2, 3)
    self.up_64      =   UpBlock(in_channels//2, in_channels//4, 2)
    self.up_32      =   UpBlock(in_channels//4, in_channels//8, 1)
    self.out_block  =   OutputBlock(in_channels//8, out_channels)

  def forward(self, feature_pyramid):
    out16, out32, out64, out128, out256 = feature_pyramid
    out    =  self.up_256(out256, out128)
    out    =  self.up_128(out, out64)
    out    =  self.up_64(out, out32)
    out    =  self.up_32(out, out16)
    out    =  self.out_block(out)
    return out


class interUNet(interModel):
  """ unet implementation """

  @serialize
  def __init__(
    self,
    encoder_params={},
    decoder_params={},
    norm_radius=3,
    use_disks=False,
    cpu_dist_maps=False,
    norm_mean_std=([.485, .456, .406], [.229, .224, .225])
  ):
    super().__init__()
    self.normalization = BatchImageNormalize(norm_mean_std[0], norm_mean_std[1])

    self.dist_maps = DistMaps(
      norm_radius=norm_radius, 
      spatial_scale=1.0,
      cpu_mode=cpu_dist_maps, 
      use_disks=use_disks,
    )

    self.image_encoder = Encoder(**encoder_params)
    self.prompt_encoder = Encoder(**encoder_params)
    self.mask_decoder = Decoder(**decoder_params)

  def get_image_feats(self, image: torch.Tensor) -> List[torch.Tensor]:
    image = self.normalization(image)
    image_feats = self.image_encoder(image)
    return image_feats
  
  def get_prompt_feats(self, prompts: Dict) -> List[torch.Tensor]:
    prev_mask = prompts['prev_mask']
    points = prompts['points']
    point_mask = self.dist_maps(prev_mask.shape, points)

    prompt_mask = torch.cat((prev_mask, point_mask), dim=1)
    prompt_feats = self.prompt_encoder(prompt_mask)

    return prompt_feats

  def forward(self, image_feats, prompt_feats):
    feats_fused = [x + y for x, y in zip(image_feats, prompt_feats)]
    mask_prob = self.mask_decoder(feats_fused)
    return {'instances': mask_prob}