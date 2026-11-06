import torch
import torch.nn as nn
import torch.nn.functional as F
from core import ALIF


class DispHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=2):
        super(DispHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class ParametricSigmoid(nn.Module):
    def __init__(self, alpha=1., beta=0.):
        super().__init__()

        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, x):
        return torch.sigmoid(self.alpha * x + self.beta)


class ALIFModel(nn.Module):
    def __init__(self, hidden_dim, input_dim, kernel_size=3, layer='32'):
        super(ALIFModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        self.layer = layer

        if self.layer == '32':
            self.snn = ALIF.ALIFNode32()

        if self.layer == '16':
            self.snn = ALIF.ALIFNode16()

        if self.layer == '08':
            self.snn = ALIF.ALIFNode08()

        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        
        self.ln_z = nn.GroupNorm(num_groups=1, num_channels=hidden_dim, affine=False)
        self.ln_r = nn.GroupNorm(num_groups=1, num_channels=hidden_dim, affine=False)
        self.ln_q = nn.GroupNorm(num_groups=1, num_channels=hidden_dim, affine=False)
        
        self.parametric_sigmoid = ParametricSigmoid()

    
    def forward(self, s, h, cz, cr, cq, *x_list):

        x = torch.cat(x_list, dim=1)
        hx = torch.cat([h, x], dim=1)
        
        z = self.parametric_sigmoid(self.ln_z(self.convz(hx) + cz))   # LayerNorm + Parametric Sigmoid
        r = self.parametric_sigmoid(self.ln_r(self.convr(hx) + cr))
        q = self.parametric_sigmoid(self.ln_q(self.convq(hx) + cq))
        
        h = self.snn(s, h, z, r, q)
        return h
  

class BasicMotionEncoder(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder, self).__init__()
        self.args = args

        cor_planes = args.corr_levels * (2*args.corr_radius + 1)  # default: 36

        self.convc1 = nn.Conv2d(cor_planes, 64, 1, padding=0)
        self.convc2 = nn.Conv2d(64, 64, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+64, 128-2, 3, padding=1)

    def forward(self, disp, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        dis = F.relu(self.convf1(disp))
        dis = F.relu(self.convf2(dis))

        cor_dis = torch.cat([cor, dis], dim=1)
        out = F.relu(self.conv(cor_dis))
        return torch.cat([out, disp], dim=1)


def pool2x(x):
    return F.avg_pool2d(x, 3, stride=2, padding=1)

def interp(x, dest):
    interp_args = {'mode': 'bilinear', 'align_corners': True}
    return F.interpolate(x, dest.shape[2:], **interp_args)


class BasicMultiUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dims=[]):
        super().__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        encoder_output_dim = 128

        self.snn08 = ALIFModel(hidden_dims[2], encoder_output_dim + hidden_dims[1] * (args.n_snn_layers > 1), layer='08')
        self.snn16 = ALIFModel(hidden_dims[1], hidden_dims[0] * (args.n_snn_layers == 3) + hidden_dims[2], layer='16')
        self.snn32 = ALIFModel(hidden_dims[0], hidden_dims[1], layer='32')

        self.disp_head = DispHead(hidden_dims[2], hidden_dim=256, output_dim=2)
        factor = 2**self.args.n_downsample

        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dims[2], 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, (factor**2)*9, 1, padding=0))

    
    def forward(self, net, inp, corr=None, disp=None, iter08=True, iter16=True, iter32=True, update=True):

        if iter32:
            net[2] = self.snn32(0., net[2], *(inp[2]), pool2x(net[1]))

        if iter16:
            if self.args.n_snn_layers > 2:
                net[1] = self.snn16(net[2], net[1], *(inp[1]), pool2x(net[0]), interp(net[2], net[1]))
            else:
                net[1] = self.snn16(net[1], *(inp[1]), pool2x(net[0]))

        if iter08:
            motion_features = self.encoder(disp, corr)
            if self.args.n_snn_layers > 1:
                net[0], v = self.snn08(net[1], net[0], *(inp[0]), motion_features, interp(net[1], net[0]))
            else:
                net[0] = self.snn08(net[0], *(inp[0]), motion_features)

        if not update:
            return net

        delta_disp = self.disp_head(v)

        mask = 0.25 * self.mask(v)

        return net, v, mask, delta_disp
