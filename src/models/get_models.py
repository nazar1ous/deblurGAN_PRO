from src.models.SRGAN import Generator as SRGAN_generator
from src.models.SRGAN import Discriminator as SRGAN_discriminator
from src.models.MIMO_UNet_pure.MIMOUNet import MIMOUNetExtracted
from src.models.n_layers_discriminator import NLayerDiscriminator


def get_generator(generator_name):
    if generator_name == "SRGAN_generator":
        return SRGAN_generator(base_channels=64,
                                n_ps_blocks=0,
                                n_res_blocks=16)
    elif generator_name == "MIMO-Unet_pure":
        return MIMOUNetExtracted(num_res=1)


def get_discriminator(discriminator_name):
    if discriminator_name == "SRGAN_discriminator":
        return SRGAN_discriminator(n_blocks=1, base_channels=8)
    elif discriminator_name == "n_layers":
        return NLayerDiscriminator(input_nc=3,
                                      n_layers=3,
                                   use_sigmoid=False)