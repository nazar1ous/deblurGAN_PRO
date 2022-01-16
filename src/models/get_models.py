from src.models.SRGAN import Generator as SRGAN_generator
from src.models.SRGAN import Discriminator as SRGAN_discriminator


def get_generator(generator_name):
    if generator_name == "SRGAN_generator":
        return SRGAN_generator(base_channels=64,
                                n_ps_blocks=0,
                                n_res_blocks=16)


def get_discriminator(discriminator_name):
    if discriminator_name == "SRGAN_discriminator":
        return SRGAN_discriminator(n_blocks=1, base_channels=8)