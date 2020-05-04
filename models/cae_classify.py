import torch
import torch.nn as nn
import torch.nn.functional as F


class CAE_Classify(nn.Module):
    """
    This AE module will be fed 3x128x128 patches from the original image
    Shapes are (batch_size, channels, height, width)

    Latent representation: 32x32x32 bits per patch => 240KB per image (for 720p)
    """

    def __init__(self):
        super(CAE_Classify, self).__init__()

        self.encoded = None

        # ENCODER

        # 64x64x64
        self.e_conv_1 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),      # tuple的顺序是左右上下
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5), stride=(2, 2)),
            # nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
            # nn.MaxPool2d(2, 2),
            nn.LeakyReLU()
        )

        # 128x32x32
        self.e_conv_2 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(2, 2)),
            # nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            # nn.MaxPool2d(2, 2),
            nn.LeakyReLU()
        )

        # 128x32x32
        self.e_block_1 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            # nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            # nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            # nn.MaxPool2d(2, 2),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 128x32x32
        self.e_block_2 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            # nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1)),
            # nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1)),
            # nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1)),
            # nn.MaxPool2d(2, 2),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 128x32x32
        self.e_block_3 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 32x32x32
        self.e_conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.Tanh()
        )

        for p in self.parameters():
            p.requires_grad = False

        # Classify
        self.c_linear_1 = nn.Linear(32*32*32*4, 1024)
        self.c_linear_2 = nn.Linear(1024, 7)


    def forward(self, x):
        ec1 = self.e_conv_1(x)
        ec2 = self.e_conv_2(ec1)
        eblock1 = self.e_block_1(ec2) + ec2
        eblock2 = self.e_block_2(eblock1) + eblock1
        # ec3 = eblock2
        eblock3 = self.e_block_3(eblock2) + eblock2
        ec3 = self.e_conv_3(eblock3)  # in [-1, 1] from tanh activation

        # stochastic binarization
        with torch.no_grad():
            rand = torch.rand(ec3.shape).cuda()
            prob = (1 + ec3) / 2
            eps = torch.zeros(ec3.shape).cuda()
            eps[rand <= prob] = (1 - ec3)[rand <= prob]
            eps[rand > prob] = (-ec3 - 1)[rand > prob]

        # encoded tensor
        self.encoded = 0.5 * (ec3 + eps + 1)  # (-1|1) -> (0|1)

        return self.classify(self.encoded)

    def classify(self, encoded):
        out = encoded.view(encoded.size(0), -1)
        out = self.c_linear_1(out)
        out = self.c_linear_2(out)
        out = F.log_softmax(out, 1)                 # can replace with softmax()

        return out


