
from engine.core.utils.serialization import serialize
from engine.core.model.modules.unet_modules import *
from engine.core.model.auto_base_model import autoModel


class autoUNet(autoModel):
  """ unet implementation """

  @serialize
  def __init__(self, num_in_channels, num_out_channels):
    super().__init__(num_in_channels, num_out_channels)
    
    self.in_block   =   InputBlock(num_in_channels, 16)
    self.down_32    =   DownBlock(16, 1)
    self.down_64    =   DownBlock(32, 2)
    self.down_128   =   DownBlock(64, 3)
    self.down_256   =   DownBlock(128, 3)
    self.up_256     =   UpBlock(256, 256, 3)
    self.up_128     =   UpBlock(256, 128, 3)
    self.up_64      =   UpBlock(128, 64, 2)
    self.up_32      =   UpBlock(64, 32, 1)
    self.out_block  =   OutputBlock(32, num_out_channels)

  def forward(self, input):
    out16  =  self.in_block(input)
    out32  =  self.down_32(out16)
    out64  =  self.down_64(out32)
    out128 = self.down_128(out64)
    out256 = self.down_256(out128)
    out    =  self.up_256(out256, out128)
    out    =  self.up_128(out, out64)
    out    =  self.up_64(out, out32)
    out    =  self.up_32(out, out16)
    out    =  self.out_block(out)
    return out

  @property
  def max_stride(self):
    return 16