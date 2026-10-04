import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import math


# ------------------------------------------------------
# Component modules for conditional 1D Unet
# ------------------------------------------------------

class Downsample1d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.conv(x)


class Conv1dBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish()
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, 
            in_channels: int, 
            out_channels: int, 
            cond_dim: int,
            kernel_size: int = 3,
            n_groups: int = 8,
            cond_predict_scale: bool = False) -> None:
        super().__init__()

        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels

        self.conv1 = Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups)
        self.film_cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange('batch t -> batch t 1')
        )
        self.conv2 = Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups)

        if in_channels != out_channels:
            self.residual_conection_conv = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.residual_conection_conv = nn.Identity()

    def forward(self, x, cond):
        out = self.conv1(x)

        film_embed = self.film_cond_encoder(cond)
        if self.cond_predict_scale:
            film_embed = film_embed.reshape(film_embed.shape[0], 2, self.out_channels, 1)
            scale, bias = film_embed[:,0,...], film_embed[:,1,...]
            out = scale * out + bias
        else:
            out = out + film_embed

        out = self.conv2(out)

        out = out + self.residual_conection_conv(x)
        return out

    
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.tensor) -> torch.tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class SelfAttention1D(nn.Module):
    def __init__(self, channels: int, n_heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=n_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:        
        residual = x

        x = self.norm(x)
        x_reshaped = x.transpose(1, 2)
        attn_out, _ = self.attn(x_reshaped, x_reshaped, x_reshaped)
        attn_out = attn_out.transpose(1, 2)

        return attn_out + residual



# ---------------------------------------------------------
# Conditional 1D Unet arch with optional self attention 
# ---------------------------------------------------------

class ConditionalUnet1D(nn.Module):
    def __init__(self,
        input_dim: int,
        cond_dim: int,
        diffusion_timestep_embed_dim: int = 256,
        down_dims: list[int] = [128, 256, 512],
        kernel_size: int = 3,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
        use_attention: bool = False,
        attention_n_heads: int = 4
        ) -> None:
        super().__init__()

        all_dims = [input_dim] + list(down_dims)
        in_out_pairings = list(zip(all_dims[:-1], all_dims[1:]))
        # NOTE: every block's FiLM conditioning receives the concat of the diffusion timestep 
        # embedding AND the observation embedding (cond_dim), not observation embedding alone
        cond_dim += diffusion_timestep_embed_dim

        
        # Embedding representation of which timestep the diffusion is on
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_timestep_embed_dim),
            nn.Linear(diffusion_timestep_embed_dim, diffusion_timestep_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_timestep_embed_dim * 4, diffusion_timestep_embed_dim),
        )


        # Downsampling encoder portion
        downsample_modules = nn.ModuleList([])
        for j, (dim_in, dim_out) in enumerate(in_out_pairings):
            is_last = j >= (len(in_out_pairings) - 1)
            
            downsample_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))
        self.down_modules = downsample_modules


        # Bottleneck portion (with optional self attention -- could be useful for longer prediction horizons)
        bottleneck_dim = all_dims[-1]
        bottleneck_modules = [
            ConditionalResidualBlock1D(
                bottleneck_dim, bottleneck_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale)
        ]
        if use_attention:
            bottleneck_modules.append(SelfAttention1D(bottleneck_dim, n_heads=attention_n_heads))
        bottleneck_modules.append(
            ConditionalResidualBlock1D(
                bottleneck_dim, bottleneck_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            )
        )
        self.bottleneck_modules = nn.ModuleList(bottleneck_modules)


        # Upsampling decoder portion
        upsample_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out_pairings[1:])):
            is_last = ind >= (len(in_out_pairings) - 1)

            upsample_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))
        self.upsample_modules = upsample_modules


        self.final_conv = nn.Sequential(
            Conv1dBlock(down_dims[0], down_dims[0], kernel_size=kernel_size),
            nn.Conv1d(down_dims[0], input_dim, 1)
        )

        
    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,  
        cond: torch.Tensor
        ) -> torch.Tensor:
        """
        Predicts the noise added to `x`, conditioned on `cond` (our observation embedding) and 
        `timestep` (how noisy `x` currently is).

        x:           (B, T, input_dim) -- noisy action sequence
        timestep:    (B,) -- diffusion step index per batch element
        cond:        (B, cond_dim) -- observation embedding
        
        Output:      (B, T, input_dim).
        """
        # Reshape from channel last to channel first --> needed for convolutions
        x = x.transpose(1, 2)

        diffusion_time_embed = self.diffusion_step_encoder(timestep)
        full_cond = torch.cat([diffusion_time_embed, cond], dim=-1)

        # Down path
        skip_connections = []
        for res_block_1, res_block_2, downsample in self.down_modules:
            x = res_block_1(x, full_cond)
            x = res_block_2(x, full_cond)
            skip_connections.append(x)
            x = downsample(x)

        # Bottleneck
        for module in self.bottleneck_modules:
            if isinstance(module, SelfAttention1D):
                x = module(x)
            else:
                x = module(x, full_cond)

        # Up path
        for res_block_1, res_block_2, upsample in self.upsample_modules:
            x = torch.cat((x, skip_connections.pop()), dim=1) 
            x = res_block_1(x, full_cond)
            x = res_block_2(x, full_cond)
            x = upsample(x)

        x = self.final_conv(x)

        # Reshape back from channel first to channel last
        x = x.transpose(1, 2)

        return x