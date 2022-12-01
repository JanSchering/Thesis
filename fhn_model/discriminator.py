# %%

import torch as t
import torch.nn as nn
from torchsummary import summary


class Discriminator(nn.Module):
    def __init__(self, grid_size: int):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            # shape (N,4,grid_size,grid_size)
            nn.Conv2d(
                in_channels=4, out_channels=8, kernel_size=4, stride=2, bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),
            # new_dim = (grid_size-4)/2+1
            # shape (N,8,new_dim,new_dim)
            nn.Conv2d(
                in_channels=8, out_channels=16, kernel_size=4, stride=2, bias=False
            ),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            # new_dim = (new_dim-4)/2+1
            # shape (N,16,new_dim,new_dim)
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=4, stride=2, bias=False
            ),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            # new_dim = (new_dim-4)/2+1
            # shape (N,32,new_dim,new_dim)
            nn.Conv2d(
                in_channels=32, out_channels=16, kernel_size=4, stride=2, bias=False
            ),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            # new_dim = (new_dim-4)/2+1
            # shape (N,16,new_dim,new_dim)
            nn.Conv2d(
                in_channels=16, out_channels=4, kernel_size=4, stride=2, bias=False
            ),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2, inplace=True),
            # new_dim = (new_dim-4)/2+1
            # shape (N,4,new_dim,new_dim)
            nn.Conv2d(
                in_channels=4, out_channels=1, kernel_size=4, stride=2, bias=False
            ),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2, inplace=True),
            # new_dim = (new_dim-4)/2+1
            # shape (N,1,new_dim,new_dim)
            nn.Flatten(),
            nn.Linear(
                in_features=self.reduce_dim(grid_size, 6) ** 2,
                out_features=10,
            ),
            nn.Linear(in_features=10, out_features=1),
            nn.Sigmoid(),
        )

    def reduce_dim(self, dim, num_reductions):
        res = dim
        for i in range(num_reductions):
            res = self.reduce(res)
        return res

    def reduce(self, dim):
        return (dim - 4) // 2 + 1

    def forward(self, input):
        return self.net(input)


# %%
discriminator_net = Discriminator(grid_size=512).cuda()
# %%
summary(discriminator_net, input_size=(4, 512, 512))
# %%
