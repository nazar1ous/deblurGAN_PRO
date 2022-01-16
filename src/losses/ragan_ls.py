from torch import mean
from torch.nn import Module

from src.utils.image_pool import ImagePool


class RelativisticDiscLossLS:
    def name(self):
        return 'RelativisticDiscLossLS'

    def __init__(self, use_l1=False):

        self.fake_pool = ImagePool(50)  # create image buffer to store previously generated images
        self.real_pool = ImagePool(50)

    def get_g_loss(self, net, fakeB, realB):
        # First, G(A) should fake the discriminator
        self.pred_fake = net.forward(fakeB)

        # Real
        self.pred_real = net.forward(realB)

        # Combined loss
        errG = (mean((self.pred_real - mean(self.fake_pool.query()) + 1) ** 2) +
                mean((self.pred_fake - mean(self.real_pool.query()) - 1) ** 2)) / 2
        return errG

    def get_loss(self, net, fakeB, realB):
        # Fake
        # stop backprop to the generator by detaching fake_B
        # Generated Image Disc Output should be close to zero
        self.pred_fake = net.forward(fakeB.detach())
        self.fake_pool.add(self.pred_fake)

        # Real
        self.pred_real = net.forward(realB)
        self.real_pool.add(self.pred_real)

        # Combined loss
        self.loss_D = (mean((self.pred_real - mean(self.fake_pool.query()) - 1) ** 2) +
                       mean(
                           (self.pred_fake - mean(self.real_pool.query()) + 1) ** 2)) / 2
        return self.loss_D

    # def __call__(self, net, fakeB, realB):
    #     return self.get_loss(net, fakeB, realB)
