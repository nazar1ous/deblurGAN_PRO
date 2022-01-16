from src.models.SRGAN import Generator as SRGAN_generator
from src.models.SRGAN import Discriminator as SRGAN_discriminator
from src.models.MIMO_FPN.MIMO_FPN import MIMOFPN


def get_generator(generator_name):
    if generator_name == "SRGAN_generator":
        return SRGAN_generator(base_channels=64,
                                n_ps_blocks=0,
                                n_res_blocks=16)
    elif generator_name == "MIMO-FPN":
        return MIMOFPN(num_res=1)


def get_discriminator(discriminator_name):
    if discriminator_name == "SRGAN_discriminator":
        return SRGAN_discriminator(n_blocks=5, base_channels=64)