from torch import nn
from torch.nn.functional import normalize
import torch

class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)
class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )
    def forward(self, x):
        return self.decoder(x)

class GCFAggMVC(nn.Module):
    def __init__(self, view, input_size, low_feature_dim, high_feature_dim, device):
        super(GCFAggMVC, self).__init__()
        self.encoders = []
        self.decoders = []
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], low_feature_dim).to(device))
            self.decoders.append(Decoder(input_size[v], low_feature_dim).to(device))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.Specific_view = nn.Sequential(
            nn.Linear(low_feature_dim, high_feature_dim),
        )
        self.Common_view = nn.Sequential(
            nn.Linear(low_feature_dim*view, high_feature_dim),
        )
        self.view = view
        self.TransformerEncoderLayer = nn.TransformerEncoderLayer(d_model=low_feature_dim*view, nhead=1, dim_feedforward=256)
        self.TransformerEncoder = nn.TransformerEncoder(self.TransformerEncoderLayer, num_layers=1)
    def forward(self, xs):
        xrs = []
        zs = []
        hs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            h = normalize(self.Specific_view(z), dim=1)
            xr = self.decoders[v](z)
            hs.append(h)
            zs.append(z)
            xrs.append(xr)
        return  xrs, zs, hs
    def GCFAgg(self, xs):
        zs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            zs.append(z)
        commonz = torch.cat(zs, 1)
        commonz, S = self.TransformerEncoderLayer(commonz)
        commonz = normalize(self.Common_view(commonz), dim=1)
        return commonz, S
