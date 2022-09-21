import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Positional Encoding of the input coordinates.
    
    encodes x to (..., sin(2^k x), cos(2^k x), ...)
    k takes "num_freqs" number of values equally spaced between [0, max_freq]
    """
    def __init__(self, max_freq, num_freqs):
        """
        Args:
            max_freq (int): maximum frequency in the positional encoding.
            num_freqs (int): number of frequencies between [0, max_freq]
        """
        super().__init__()
        freqs = 2**torch.linspace(0, max_freq, num_freqs)
        self.register_buffer("freqs", freqs) #(num_freqs)

    def forward(self, x):
        """
        Inputs:
            x: (num_rays, num_samples, in_features)
        Outputs:
            out: (num_rays, num_samples, 2*num_freqs*in_features)
        """
        x_proj = x.unsqueeze(dim=-2)*self.freqs.unsqueeze(dim=-1) #(num_rays, num_samples, num_freqs, in_features)
        x_proj = x_proj.reshape(*x.shape[:-1], -1) #(num_rays, num_samples, num_freqs*in_features)
        out = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1) #(num_rays, num_samples, 2*num_freqs*in_features)
        return out


class SimpleNeRF(nn.Module):
    """
    A simple NeRF MLP without view dependence and skip connections.
    """
    def __init__(self, in_features, max_freq, num_freqs,
                 hidden_features, hidden_layers, out_features):
        """
        in_features: number of features in the input.
        max_freq: maximum frequency in the positional encoding.
        num_freqs: number of frequencies between [0, max_freq] in the positional encoding.
        hidden_features: number of features in the hidden layers of the MLP.
        hidden_layers: number of hidden layers.
        out_features: number of features in the output.
        """
        super().__init__()

        self.net = []
        self.net.append(PositionalEncoding(max_freq, num_freqs))
        self.net.append(nn.Linear(2*num_freqs*in_features, hidden_features))
        self.net.append(nn.ReLU())
    
        for i in range(hidden_layers-1):
            self.net.append(nn.Linear(hidden_features, hidden_features))
            self.net.append(nn.ReLU())

        self.net.append(nn.Linear(hidden_features, out_features))
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        """
        At each input xyz point return the rgb and sigma values.
        Input:
            x: (num_rays, num_samples, 3)
        Output:
            rgb: (num_rays, num_samples, 3)
            sigma: (num_rays, num_samples)
        """
        out = self.net(x)
        rgb = torch.sigmoid(out[..., :-1])
        sigma = F.softplus(out[..., -1])
        return rgb, sigma


def build_nerf(args):
    model = SimpleNeRF(in_features=3, max_freq=args.max_freq, num_freqs=args.num_freqs,
                        hidden_features=args.hidden_features, hidden_layers=args.hidden_layers,
                        out_features=4)
    return model


# import pdb

BOX_OFFSETS = torch.tensor([[[i, j, k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]],
                           device='cuda')


class PositionalEncoding(nn.Module):
    """
    Positional Encoding of the input coordinates.
    
    encodes x to (..., sin(2^k x), cos(2^k x), ...)
    k takes "num_freqs" number of values equally spaced between [0, max_freq]
    """

    def __init__(self, max_freq, num_freqs):
        """
        Args:
            max_freq (int): maximum frequency in the positional encoding.
            num_freqs (int): number of frequencies between [0, max_freq]
        """
        super().__init__()
        freqs = 2**torch.linspace(0, max_freq, num_freqs)
        self.register_buffer("freqs", freqs)  # (num_freqs)

    def forward(self, x):
        """
        Inputs:
            x: (num_rays, num_samples, in_features)
        Outputs:
            out: (num_rays, num_samples, 2*num_freqs*in_features)
        """
        x_proj = x.unsqueeze(dim=-2)*self.freqs.unsqueeze(dim=-
                                                          1)  # (num_rays, num_samples, num_freqs, in_features)
        # (num_rays, num_samples, num_freqs*in_features)
        x_proj = x_proj.reshape(*x.shape[:-1], -1)
        # (num_rays, num_samples, 2*num_freqs*in_features)
        out = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return out


class SimpleNeRF(nn.Module):
    """
    A simple NeRF MLP without view dependence and skip connections.
    """

    def __init__(self, in_features, max_freq, num_freqs,
                 hidden_features, hidden_layers, out_features):
        """
        in_features: number of features in the input.
        max_freq: maximum frequency in the positional encoding.
        num_freqs: number of frequencies between [0, max_freq] in the positional encoding.
        hidden_features: number of features in the hidden layers of the MLP.
        hidden_layers: number of hidden layers.
        out_features: number of features in the output.
        """
        super().__init__()

        self.net = []
        self.net.append(PositionalEncoding(max_freq, num_freqs))  # ()
        self.net.append(nn.Linear(2*num_freqs*in_features, hidden_features))
        self.net.append(nn.ReLU())

        for i in range(hidden_layers-1):
            self.net.append(nn.Linear(hidden_features, hidden_features))
            self.net.append(nn.ReLU())

        self.net.append(nn.Linear(hidden_features, out_features))
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        """
        At each input xyz point return the rgb and sigma values.
        Input:
            x: (num_rays, num_samples, 3)
        Output:
            rgb: (num_rays, num_samples, 3)
            sigma: (num_rays, num_samples)
        """
        out = self.net(x)
        rgb = torch.sigmoid(out[..., :-1])
        sigma = F.softplus(out[..., -1])
        return rgb, sigma


class HashEmbedder(nn.Module):
    def __init__(self, bounding_box, n_levels=16, n_features_per_level=2,
                 log2_hashmap_size=19, base_resolution=16, finest_resolution=512):
        super(HashEmbedder, self).__init__()
        # [x_min, x_max, y_min, y_max, z_min, z_max]
        self.bounding_box = bounding_box
        self.n_levels = n_levels  # 16
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = torch.tensor(base_resolution)
        self.finest_resolution = torch.tensor(finest_resolution)
        self.out_dim = self.n_levels * self.n_features_per_level

        self.b = torch.exp((torch.log(self.finest_resolution) -
                           torch.log(self.base_resolution))/(n_levels-1))

        self.embeddings = nn.ModuleList([nn.Embedding(2**self.log2_hashmap_size,
                                        self.n_features_per_level) for i in range(n_levels)])
        # custom uniform initialization
        for i in range(n_levels):
            nn.init.uniform_(self.embeddings[i].weight, a=-0.0001, b=0.0001)
            # self.embeddings[i].weight.data.zero_()

    def trilinear_interp(self, x, voxel_min_vertex, voxel_max_vertex, voxel_embedds):
        '''
        x: B x 3
        voxel_min_vertex: B x 3
        voxel_max_vertex: B x 3
        voxel_embedds: B x 8 x 2
        '''
        # source: https://en.wikipedia.org/wiki/Trilinear_interpolation
        weights = (x - voxel_min_vertex) / \
            (voxel_max_vertex-voxel_min_vertex)  # B x 3

        # step 1
        # 0->000, 1->001, 2->010, 3->011, 4->100, 5->101, 6->110, 7->111
        # breakpoint()
        c00 = voxel_embedds[:, :, 0]*(1-weights[:, :, 0][:, :, None]) + \
            voxel_embedds[:, :, 4]*weights[:, :, 0][:, :, None]
        c01 = voxel_embedds[:, :, 1]*(1-weights[:, :, 0][:, :, None]) + \
            voxel_embedds[:, :, 5]*weights[:, :, 0][:, :, None]
        c10 = voxel_embedds[:, :, 2]*(1-weights[:, :, 0][:, :, None]) + \
            voxel_embedds[:, :, 6]*weights[:, :, 0][:, :, None]
        c11 = voxel_embedds[:, :, 3]*(1-weights[:, :, 0][:, :, None]) + \
            voxel_embedds[:, :, 7]*weights[:, :, 0][:, :, None]

        c0 = c00*(1-weights[:, :, 1][:, :, None]) + \
            c10*weights[:, :, 1][:, :, None]
        c1 = c01*(1-weights[:, :, 1][:, :, None]) + \
            c11*weights[:, :, 1][:, :, None]

        c = c0*(1-weights[:, :, 2][:, :, None]) + \
            c1*weights[:, :, 2][:, :, None]

        return c

    def forward(self, x):
        # x is 3D point position: B x 3
        x_embedded_all = []
        for i in range(self.n_levels):
            resolution = torch.floor(self.base_resolution * self.b**i)
            voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices = get_voxel_vertices(
                x, self.bounding_box,
                resolution, self.log2_hashmap_size)

            voxel_embedds = self.embeddings[i](hashed_voxel_indices)

            x_embedded = self.trilinear_interp(
                x, voxel_min_vertex, voxel_max_vertex, voxel_embedds)
            x_embedded_all.append(x_embedded)

        return torch.cat(x_embedded_all, dim=-1)  # B x 32


def hash(coords, log2_hashmap_size):
    '''
    coords: this function can process upto 7 dim coordinates
    log2T:  logarithm of T w.r.t 2
    '''
    primes = [1, 2654435761, 805459861, 3674653429,
              2097192037, 1434869437, 2165219737]

    xor_result = torch.zeros_like(coords)[..., 0]
    for i in range(coords.shape[-1]):
        xor_result ^= coords[..., i]*primes[i]

    return torch.tensor((1 << log2_hashmap_size)-1).to(xor_result.device) & xor_result


def get_voxel_vertices(xyz, bounding_box, resolution, log2_hashmap_size):
    '''
    xyz: 3D coordinates of samples. B x 3
    bounding_box: min and max x,y,z coordinates of object bbox
    resolution: number of voxels per axis
    '''
    box_min, box_max = bounding_box

    if not torch.all(xyz <= box_max) or not torch.all(xyz >= box_min):
        # print("ALERT: some points are outside bounding box. Clipping them!")
        # pdb.set_trace()
        xyz = torch.clamp(xyz, min=box_min, max=box_max)

    grid_size = (box_max-box_min)/resolution
    # breakpoint()
    bottom_left_idx = torch.floor((xyz-box_min)/grid_size).int()
    voxel_min_vertex = bottom_left_idx*grid_size + box_min
    voxel_max_vertex = voxel_min_vertex + \
        torch.tensor([1.0, 1.0, 1.0]).to(xyz.device)*grid_size

    # hashed_voxel_indices = [] # B x 8 ... 000,001,010,011,100,101,110,111
    # for i in [0, 1]:
    #     for j in [0, 1]:
    #         for k in [0, 1]:
    #             vertex_idx = bottom_left_idx + torch.tensor([i,j,k])
    #             # vertex = bottom_left + torch.tensor([i,j,k])*grid_size
    #             hashed_voxel_indices.append(hash(vertex_idx, log2_hashmap_size))
    # breakpoint()

    voxel_indices = bottom_left_idx.unsqueeze(2) + BOX_OFFSETS
    hashed_voxel_indices = hash(voxel_indices, log2_hashmap_size)

    return voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices


class HashNeRF(nn.Module):
    """
    A Hash NeRF MLP without view dependence and skip connections.
    """

    def __init__(self, in_features, max_freq, num_freqs,
                 hidden_features, hidden_layers, out_features):
        """
        in_features: number of features in the input.
        max_freq: maximum frequency in the positional encoding.
        num_freqs: number of frequencies between [0, max_freq] in the positional encoding.
        hidden_features: number of features in the hidden layers of the MLP.
        hidden_layers: number of hidden layers.
        out_features: number of features in the output.
        """
        super().__init__()

        self.net = []
        # append hashembedder
        self.bounding = torch.tensor(
            [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]).to('cuda')
        self.net.append(HashEmbedder(self.bounding))  # 16384x32
        # breakpoint()
        self.net.append(nn.Linear(32, hidden_features))
        self.net.append(nn.ReLU())

        for i in range(hidden_layers-1):
            self.net.append(nn.Linear(hidden_features, hidden_features))
            self.net.append(nn.ReLU())

        self.net.append(nn.Linear(hidden_features, out_features))
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        """
        At each input xyz point return the rgb and sigma values.
        Input:
            x: (num_rays, num_samples, 3)
        Output:
            rgb: (num_rays, num_samples, 3)
            sigma: (num_rays, num_samples)
        """
        out = self.net(x)
        rgb = torch.sigmoid(out[..., :-1])
        sigma = F.softplus(out[..., -1])
        return rgb, sigma


def build_nerf(args):
    if args.model_type == 'SimpleNeRf':
        model = SimpleNeRF(in_features=3, max_freq=args.max_freq, num_freqs=args.num_freqs,
                           hidden_features=args.hidden_features, hidden_layers=args.hidden_layers,
                           out_features=4)

    elif args.model_type == 'hash':
        model = HashNeRF(in_features=3, max_freq=args.max_freq, num_freqs=args.num_freqs,
                         hidden_features=args.hidden_features, hidden_layers=args.hidden_layers,
                         out_features=4)

    return model
