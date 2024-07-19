import math

import tinycudann as tcnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import threestudio
from threestudio.utils.base import Updateable
from threestudio.utils.config import config_to_primitive
from threestudio.utils.misc import get_rank
from threestudio.utils.ops import get_activation
from threestudio.utils.typing import *

class ProgressiveBandFrequency(nn.Module, Updateable):
    def __init__(self, in_channels: int, config: dict):
        super().__init__()
        self.N_freqs = config["n_frequencies"]
        self.in_channels, self.n_input_dims = in_channels, in_channels
        self.funcs = [torch.sin, torch.cos]
        self.freq_bands = 2 ** torch.linspace(0, self.N_freqs - 1, self.N_freqs)
        self.n_output_dims = self.in_channels * (len(self.funcs) * self.N_freqs)
        self.n_masking_step = config.get("n_masking_step", 0)
        self.update_step(
            None, None
        )  # mask should be updated at the beginning each step

    def forward(self, x):
        out = []
        for freq, mask in zip(self.freq_bands, self.mask):
            for func in self.funcs:
                out += [func(freq * x) * mask]
        return torch.cat(out, -1)

    def update_step(self, epoch, global_step, on_load_weights=False):
        if self.n_masking_step <= 0 or global_step is None:
            self.mask = torch.ones(self.N_freqs, dtype=torch.float32)
        else:
            self.mask = (
                1.0
                - torch.cos(
                    math.pi
                    * (
                        global_step / self.n_masking_step * self.N_freqs
                        - torch.arange(0, self.N_freqs)
                    ).clamp(0, 1)
                )
            ) / 2.0
            threestudio.debug(
                f"Update mask: {global_step}/{self.n_masking_step} {self.mask}"
            )


class TCNNEncoding(nn.Module):
    def __init__(self, in_channels, config, dtype=torch.float32) -> None:
        super().__init__()
        self.n_input_dims = in_channels
        with torch.cuda.device(get_rank()):
            self.encoding = tcnn.Encoding(in_channels, config, dtype=dtype)
        self.n_output_dims = self.encoding.n_output_dims
    def forward(self, x):
        return self.encoding(x)


# 4D implicit decomposition of space and time (4D-fy)
class TCNNEncodingSpatialTime(nn.Module):
    def __init__(
        self, in_channels, config, dtype=torch.float32, init_time_zero=False
    ) -> None:
        super().__init__()
        self.n_input_dims = in_channels
        config["otype"] = "HashGrid"
        self.num_frames = 1  # config["num_frames"]
        self.static = config["static"]
        self.cfg = config_to_primitive(config)
        self.cfg_time = self.cfg
        self.n_key_frames = config.get("n_key_frames", 1)
        with torch.cuda.device(get_rank()):
            self.encoding = tcnn.Encoding(self.n_input_dims, self.cfg, dtype=dtype)
            self.encoding_time = tcnn.Encoding(
                self.n_input_dims + 1, self.cfg_time, dtype=dtype
            )
        self.n_output_dims = self.encoding.n_output_dims
        self.frame_time = None
        if self.static:
            self.set_temp_param_grad(requires_grad=False)
        self.use_key_frame = config.get("use_key_frame", False)
        self.is_video = True
        self.update_occ_grid = False

    def set_temp_param_grad(self, requires_grad=False):
        self.set_param_grad(self.encoding_time, requires_grad=requires_grad)

    def set_param_grad(self, param_list, requires_grad=False):
        if isinstance(param_list, nn.Parameter):
            param_list.requires_grad = requires_grad
        else:
            for param in param_list.parameters():
                param.requires_grad = requires_grad

    def forward(self, x):
        # TODO frame_time only supports batch_size == 1 cases
        if self.update_occ_grid and not isinstance(self.frame_time, float):
            frame_time = self.frame_time
        else:
            if (self.static or not self.training) and self.frame_time is None:
                frame_time = torch.zeros(
                    (self.num_frames, 1), device=x.device, dtype=x.dtype
                ).expand(x.shape[0], 1)
            else:
                if self.frame_time is None:
                    frame_time = 0.0
                else:
                    frame_time = self.frame_time
                frame_time = (
                    torch.ones((self.num_frames, 1), device=x.device, dtype=x.dtype)
                    * frame_time
                ).expand(x.shape[0], 1)
            frame_time = frame_time.view(-1, 1)
        enc_space = self.encoding(x)
        x_frame_time = torch.cat((x, frame_time), 1)
        enc_space_time = self.encoding_time(x_frame_time)
        enc = enc_space + enc_space_time
        return enc


class FourierFeatureTransform(nn.Module):
    def __init__(self, num_input_channels, mapping_size, initial_scale=1, dtype=torch.float32):
        super().__init__()
        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self._B = nn.Parameter(torch.randn((num_input_channels, mapping_size), dtype=dtype) * initial_scale, requires_grad=False)
    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            x = x @ self._B * 2 * torch.tensor(math.pi, dtype=x.dtype, device=x.device)
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class MultiScaleTriplane(nn.Module, Updateable):
    def __init__(self, n_input_dims=3, n_scales=4, channel=32, grid_size=512, true_global_step=0, dtype=torch.float32):
        super().__init__()
        self.n_input_dims = n_input_dims
        self.n_output_dims = channel * 4
        # self.n_output_dims = channel
        self.n_scales = n_scales
        self.iteration = 0
        self.vector_1 = nn.ParameterList([nn.Parameter(torch.randn(1, channel, grid_size, 1, dtype=dtype) * 1e-3) for _ in range(3)])
        self.plane_2 = nn.ParameterList([nn.Parameter(torch.randn(1, channel, grid_size // 2, grid_size // 2, dtype=dtype) * 3e-4) for _ in range(3)])
        self.plane_3 = nn.ParameterList([nn.Parameter(torch.randn(1, channel, grid_size // 4, grid_size // 4, dtype=dtype) * 1.5e-4) for _ in range(3)])
        self.plane_4 = nn.ParameterList([nn.Parameter(torch.randn(1, channel, grid_size // 8, grid_size // 8, dtype=dtype) * 7.5e-5) for _ in range(3)])
        self.net1 = nn.Sequential(
            # FourierFeatureTransform(channel, channel // 2, initial_scale=0.0075),
            FourierFeatureTransform(channel * 4, channel * 2, initial_scale=0.0075),
        )
        self.plane_4_frozen, self.plane_3_frozen, self.plane_2_frozen = False, False, False
    def sample_plane(self, coords2d, plane):
        assert len(coords2d.shape) == 3, coords2d.shape
        # coords2d = coords2d + torch.normal(mean=0, std=0.005, size=coords2d.shape).to(coords2d.device)
        sampled_features = F.grid_sample(plane, coords2d.view(coords2d.shape[0], 1, -1, coords2d.shape[-1]),
                                                           mode='bicubic', padding_mode='border', align_corners=True)
        #mode bicubic padding_mode
        N, C, H, W = sampled_features.shape
        sampled_features = sampled_features.view(N, C, H*W).permute(0, 2, 1)
        return sampled_features
    
    def sample_grid(self, coords3d, grid):
        assert len(coords3d.shape) == 3, coords3d.shape
        # coords3d = coords3d + torch.normal(mean=0, std=0.005, size=coords3d.shape).to(coords3d.device)
        sampled_features = F.grid_sample(grid, coords3d.view(coords3d.shape[0], 1, -1, 1, coords3d.shape[-1]), 
                                         mode='bicubic', padding_mode='border', align_corners=True)
        N, C, _, H, W = sampled_features.shape
        sampled_features = sampled_features.view(N, C, H*W).permute(0, 2, 1)
        return sampled_features
    
    def sample_vector(self, coords1d, vector):
        assert len(coords1d.shape) == 3, coords1d.shape
        coords1d = torch.stack([-torch.ones_like(coords1d), coords1d], dim=-1)
        # coords1d = coords1d + torch.normal(mean=0, std=0.005, size=coords1d.shape).to(coords1d.device)
        sampled_features = F.grid_sample(vector, coords1d.view(coords1d.shape[0], 1, -1, coords1d.shape[-1]), 
                                         mode='bicubic', padding_mode='border', align_corners=True)
        N, C, H, W = sampled_features.shape
        sampled_features = sampled_features.view(N, C, H*W).permute(0, 2, 1)

        return sampled_features

    def forward(self, coordinates, true_global_step=0):
        coordinates = coordinates.unsqueeze(0)
        # Update the iteration attribute
        # if self.iteration in (4001, 6001, 8001):
        #     self.update_freeze_status(self.iteration)
        
        feature1 = self.sample_plane(coordinates[..., 0:2], self.plane_3[0]) 
        feature1.add_(self.sample_plane(coordinates[..., 1:3], self.plane_3[1])) 
        feature1.add_(self.sample_plane(coordinates[..., :3:2], self.plane_3[2]))                 

        feature2 = self.sample_plane(coordinates[..., 0:2], self.plane_3[0]) 
        feature2.add_(self.sample_plane(coordinates[..., 1:3], self.plane_3[1]))
        feature2.add_(self.sample_plane(coordinates[..., :3:2], self.plane_3[2]))  

        feature3 = self.sample_plane(coordinates[..., 0:2], self.plane_2[0]) 
        feature3.add_(self.sample_plane(coordinates[..., 1:3], self.plane_2[1])) 
        feature3.add_(self.sample_plane(coordinates[..., :3:2], self.plane_2[2]))  

        feature4 = self.sample_vector(coordinates[..., 0:1], self.vector_1[0])
        feature4.add_(self.sample_vector(coordinates[..., 1:2], self.vector_1[1]))
        feature4.add_(self.sample_vector(coordinates[..., 2:3], self.vector_1[2]))

        feature = torch.cat([feature1, feature2, feature3, feature4], dim=-1)
        del feature1, feature2, feature3, feature4
        # if self.iteration > 4000:
        #     feature2 = self.sample_plane(coordinates[..., 0:2], self.plane_3[0]) 
        #     feature2.add_(self.sample_plane(coordinates[..., 1:3], self.plane_3[1]))
        #     feature2.add_(self.sample_plane(coordinates[..., :3:2], self.plane_3[2]))  

        # if self.iteration > 6000:
        #     feature3 = self.sample_plane(coordinates[..., 0:2], self.plane_2[0]) 
        #     feature3.add_(self.sample_plane(coordinates[..., 1:3], self.plane_2[1])) 
        #     feature3.add_(self.sample_plane(coordinates[..., :3:2], self.plane_2[2]))  

        # if self.iteration > 8000:
        #     feature4 = self.sample_vector(coordinates[..., 0:1], self.vector_1[0])
        #     feature4.add_(self.sample_vector(coordinates[..., 1:2], self.vector_1[1]))
        #     feature4.add_(self.sample_vector(coordinates[..., 2:3], self.vector_1[2]))

        # if self.iteration > 8000:
        #     feature = feature1 + feature2 * 0.7 + feature3 * 0.5 + feature4 * 0.3
        #     del feature1, feature2, feature3, feature4
        # elif self.iteration > 6000:
        #     feature = feature1 + feature2 * 0.7 + feature3 * 0.5
        #     del feature1, feature2, feature3
        # elif self.iteration > 4000:
        #     feature = feature1 + feature2 * 0.7
        #     del feature1, feature2
        # else:
        #     feature = feature1
        #     del feature1

        # if self.iteration > 8000:
        #     feature1 *= 0.3
        #     feature2 *= 0.5
        #     feature3 *= 0.7
        #     feature = feature1 + feature2 + feature3 + feature4
        #     del feature1, feature2, feature3, feature4
        # elif self.iteration > 6000:
        #     feature1 *= 0.5
        #     feature2 *= 0.7
        #     feature = feature1 + feature2 + feature3
        #     del feature1, feature2, feature3
        # if self.iteration > 4000:
        #     feature1 *= 0.7
        #     feature = feature1 + feature2
        #     del feature1, feature2
        # else:
        #     feature = feature1
        #     del feature1

        return self.net1(feature[0])

    
    def update_freeze_status(self, current_iteration):

        if self.iteration > 4000 and not self.plane_4_frozen:
            for p in self.plane_4.parameters():
                p.requires_grad_(False)
            self.plane_4_frozen = True

        if self.iteration > 6000 and not self.plane_3_frozen:
            for p in self.plane_3.parameters():
                p.requires_grad_(False)
            self.plane_3_frozen = True


        if self.iteration > 8000 and not self.plane_2_frozen:
            for p in self.plane_2.parameters():
                p.requires_grad_(False)
            self.plane_2_frozen = True

    def update_step(self, epoch, global_step, on_load_weights=False):
        self.iteration = global_step
class ProgressiveBandHashGrid(nn.Module, Updateable):
    def __init__(self, in_channels, config, dtype=torch.float32):
        super().__init__()
        self.n_input_dims = in_channels
        encoding_config = config.copy()
        encoding_config["otype"] = "Grid"
        encoding_config["type"] = "Hash"
        with torch.cuda.device(get_rank()):
            self.encoding = tcnn.Encoding(in_channels, encoding_config, dtype=dtype)
        self.n_output_dims = self.encoding.n_output_dims
        self.n_level = config["n_levels"]
        self.n_features_per_level = config["n_features_per_level"]
        self.start_level, self.start_step, self.update_steps = (
            config["start_level"],
            config["start_step"],
            config["update_steps"],
        )
        self.current_level = self.start_level
        self.mask = torch.zeros(
            self.n_level * self.n_features_per_level,
            dtype=torch.float32,
            device=get_rank(),
        )

    def forward(self, x):
        enc = self.encoding(x)
        enc = enc * self.mask
        return enc

    def update_step(self, epoch, global_step, on_load_weights=False):
        current_level = min(
            self.start_level
            + max(global_step - self.start_step, 0) // self.update_steps,
            self.n_level,
        )
        if current_level > self.current_level:
            threestudio.debug(f"Update current level to {current_level}")
        self.current_level = current_level
        self.mask[: self.current_level * self.n_features_per_level] = 1.0


class CompositeEncoding(nn.Module, Updateable):
    def __init__(self, encoding, include_xyz=False, xyz_scale=2.0, xyz_offset=-1.0):
        super(CompositeEncoding, self).__init__()
        self.encoding = encoding
        self.include_xyz, self.xyz_scale, self.xyz_offset = (
            include_xyz,
            xyz_scale,
            xyz_offset,
        )
        self.n_output_dims = (
            int(self.include_xyz) * self.encoding.n_input_dims
            + self.encoding.n_output_dims
        )

    def forward(self, x, *args):
        return (
            self.encoding(x, *args)
            if not self.include_xyz
            else torch.cat(
                [x * self.xyz_scale + self.xyz_offset, self.encoding(x, *args)], dim=-1
            )
        )


def get_encoding(n_input_dims: int, config) -> nn.Module:
    # input suppose to be range [0, 1]
    encoding: nn.Module
    if config.otype == "ProgressiveBandFrequency":
        encoding = ProgressiveBandFrequency(n_input_dims, config_to_primitive(config))
    elif config.otype == "ProgressiveBandHashGrid":
        encoding = ProgressiveBandHashGrid(n_input_dims, config_to_primitive(config))
    elif config.otype == "HashGridSpatialTime":
        encoding = TCNNEncodingSpatialTime(n_input_dims, config)  # 4D-fy encoding
    elif config.otype == "multiscale_triplane":
        encoding = MultiScaleTriplane(n_input_dims)
        # encoding = TCNNEncodingSpatialTime(n_input_dims, config)
    else:
        encoding = TCNNEncoding(n_input_dims, config_to_primitive(config))
    encoding = CompositeEncoding(
        encoding,
        include_xyz=config.get("include_xyz", False),
        xyz_scale=2.0,
        xyz_offset=-1.0,
    )  # FIXME: hard coded
    return encoding


class VanillaMLP(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, config: dict):
        super().__init__()
        self.n_neurons, self.n_hidden_layers = (
            config["n_neurons"],
            config["n_hidden_layers"],
        )
        layers = [
            self.make_linear(dim_in, self.n_neurons, is_first=True, is_last=False),
            self.make_activation(),
        ]
        for i in range(self.n_hidden_layers - 1):
            layers += [
                self.make_linear(
                    self.n_neurons, self.n_neurons, is_first=False, is_last=False
                ),
                self.make_activation(),
            ]
        layers += [
            self.make_linear(self.n_neurons, dim_out, is_first=False, is_last=True)
        ]
        self.layers = nn.Sequential(*layers)
        self.output_activation = get_activation(config.get("output_activation", None))

    def forward(self, x):
        # disable autocast
        # strange that the parameters will have empty gradients if autocast is enabled in AMP
        with torch.cuda.amp.autocast(enabled=False):
            x = self.layers(x)
            x = self.output_activation(x)
        return x

    def make_linear(self, dim_in, dim_out, is_first, is_last):
        layer = nn.Linear(dim_in, dim_out, bias=False)
        return layer

    def make_activation(self):
        return nn.ReLU(inplace=True)


class SphereInitVanillaMLP(nn.Module):
    def __init__(self, dim_in, dim_out, config):
        super().__init__()
        self.n_neurons, self.n_hidden_layers = (
            config["n_neurons"],
            config["n_hidden_layers"],
        )
        self.sphere_init, self.weight_norm = True, True
        self.sphere_init_radius = config["sphere_init_radius"]
        self.sphere_init_inside_out = config["inside_out"]

        self.layers = [
            self.make_linear(dim_in, self.n_neurons, is_first=True, is_last=False),
            self.make_activation(),
        ]
        for i in range(self.n_hidden_layers - 1):
            self.layers += [
                self.make_linear(
                    self.n_neurons, self.n_neurons, is_first=False, is_last=False
                ),
                self.make_activation(),
            ]
        self.layers += [
            self.make_linear(self.n_neurons, dim_out, is_first=False, is_last=True)
        ]
        self.layers = nn.Sequential(*self.layers)
        self.output_activation = get_activation(config.get("output_activation", None))

    def forward(self, x):
        # disable autocast
        # strange that the parameters will have empty gradients if autocast is enabled in AMP
        with torch.cuda.amp.autocast(enabled=False):
            x = self.layers(x)
            x = self.output_activation(x)
        return x

    def make_linear(self, dim_in, dim_out, is_first, is_last):
        layer = nn.Linear(dim_in, dim_out, bias=True)

        if is_last:
            if not self.sphere_init_inside_out:
                torch.nn.init.constant_(layer.bias, -self.sphere_init_radius)
                torch.nn.init.normal_(
                    layer.weight,
                    mean=math.sqrt(math.pi) / math.sqrt(dim_in),
                    std=0.0001,
                )
            else:
                torch.nn.init.constant_(layer.bias, self.sphere_init_radius)
                torch.nn.init.normal_(
                    layer.weight,
                    mean=-math.sqrt(math.pi) / math.sqrt(dim_in),
                    std=0.0001,
                )
        elif is_first:
            torch.nn.init.constant_(layer.bias, 0.0)
            torch.nn.init.constant_(layer.weight[:, 3:], 0.0)
            torch.nn.init.normal_(
                layer.weight[:, :3], 0.0, math.sqrt(2) / math.sqrt(dim_out)
            )
        else:
            torch.nn.init.constant_(layer.bias, 0.0)
            torch.nn.init.normal_(layer.weight, 0.0, math.sqrt(2) / math.sqrt(dim_out))

        if self.weight_norm:
            layer = nn.utils.weight_norm(layer)
        return layer

    def make_activation(self):
        return nn.Softplus(beta=100)


class TCNNNetwork(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, config: dict) -> None:
        super().__init__()
        with torch.cuda.device(get_rank()):
            self.network = tcnn.Network(dim_in, dim_out, config)

    def forward(self, x):
        return self.network(x).float()  # transform to float32


def get_mlp(n_input_dims, n_output_dims, config) -> nn.Module:
    network: nn.Module
    if config.otype == "VanillaMLP":
        network = VanillaMLP(n_input_dims, n_output_dims, config_to_primitive(config))
    elif config.otype == "SphereInitVanillaMLP":
        network = SphereInitVanillaMLP(
            n_input_dims, n_output_dims, config_to_primitive(config)
        )
    else:
        assert (
            config.get("sphere_init", False) is False
        ), "sphere_init=True only supported by VanillaMLP"
        network = TCNNNetwork(n_input_dims, n_output_dims, config_to_primitive(config))
    return network


class NetworkWithInputEncoding(nn.Module, Updateable):
    def __init__(self, encoding, network):
        super().__init__()
        self.encoding, self.network = encoding, network

    def forward(self, x):
        return self.network(self.encoding(x))


class TCNNNetworkWithInputEncoding(nn.Module):
    def __init__(
        self,
        n_input_dims: int,
        n_output_dims: int,
        encoding_config: dict,
        network_config: dict,
    ) -> None:
        super().__init__()
        with torch.cuda.device(get_rank()):
            self.network_with_input_encoding = tcnn.NetworkWithInputEncoding(
                n_input_dims=n_input_dims,
                n_output_dims=n_output_dims,
                encoding_config=encoding_config,
                network_config=network_config,
            )

    def forward(self, x):
        return self.network_with_input_encoding(x).float()  # transform to float32


def create_network_with_input_encoding(
    n_input_dims: int, n_output_dims: int, encoding_config, network_config
) -> nn.Module:
    # input suppose to be range [0, 1]
    network_with_input_encoding: nn.Module
    if encoding_config.otype in [
        "VanillaFrequency",
        "ProgressiveBandHashGrid",
    ] or network_config.otype in ["VanillaMLP", "SphereInitVanillaMLP"]:
        encoding = get_encoding(n_input_dims, encoding_config)
        network = get_mlp(encoding.n_output_dims, n_output_dims, network_config)
        network_with_input_encoding = NetworkWithInputEncoding(encoding, network)
    else:
        network_with_input_encoding = TCNNNetworkWithInputEncoding(
            n_input_dims=n_input_dims,
            n_output_dims=n_output_dims,
            encoding_config=config_to_primitive(encoding_config),
            network_config=config_to_primitive(network_config),
        )
    return network_with_input_encoding


class ToDTypeWrapper(nn.Module):
    def __init__(self, module: nn.Module, dtype: torch.dtype):
        super().__init__()
        self.module = module
        self.dtype = dtype

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return self.module(x).to(self.dtype)
