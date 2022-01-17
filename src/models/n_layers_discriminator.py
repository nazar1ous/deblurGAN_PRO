from functools import partial

# from torchsummary import summary
from numpy import ceil
from torch.nn import Module, BatchNorm2d, InstanceNorm2d, Conv2d, LeakyReLU, Sigmoid, Sequential


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(Module):
    def __init__(self, input_nc=4, ndf=64, n_layers=3, norm_layer=BatchNorm2d, use_sigmoid=False, use_parallel=True):
        super(NLayerDiscriminator, self).__init__()
        self.use_parallel = use_parallel
        if type(norm_layer) == partial:
            use_bias = norm_layer.func == InstanceNorm2d
        else:
            use_bias = norm_layer == InstanceNorm2d

        kw = 4
        padw = int(ceil((kw - 1) / 2))
        sequence = [
            Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                       kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                   kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            LeakyReLU(0.2, True)
        ]

        sequence += [Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [Sigmoid()]

        self.model = Sequential(*sequence)

    def forward(self, input):
        """
        #for testing
        print(type(input))
        print(input.shape)
        a = input[0][:3, :, :]
        a[2, :, :] += input[0][3, :, :]
        a = a.permute(1, 2, 0).cpu().numpy()

        print(a.shape)
        a = (a + 1) / 2
        print(a.max())
        cv2.imwrite('test.jpg', a * 256)

        # inputs[0].detach().mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()

        exit()
        """
        # print('\n\n\n')
        # print(input.device)
        # print('\n\n\n')
        # print('\n\n\nhere')
        # print(input.shape)
        # print('\n\n\n')
        return self.model(input)

#
# if __name__ == "__main__":
#     print("start")
#     model = NLayerDiscriminator(input_nc=3, ndf=64, n_layers=5)
#     model.cuda()
#
#     # summary(model, input_size=(3, 256, 256), batch_size=1)
#     # TODO: check other method of printing model summary like print(model)
