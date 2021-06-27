#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.
# Updates by Pablo Palafox 2021

import torch.nn as nn
import torch
import torch.nn.functional as F
import kornia

from utils import embedder
from utils import geometry_utils


class PoseDecoder(nn.Module):
    def __init__(
        self,
        latent_size,
        dims,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        xyz_in_all=None,
        latent_dropout=False,
        positional_enc=False,
        n_positional_freqs=8,
        n_alpha_epochs=80,
    ):
        super(PoseDecoder, self).__init__()

        input_dim = 3
        output_dim = 3

        if positional_enc:
            self.n_positional_freqs = n_positional_freqs
            self.pos_embedder, pos_embedder_out_dim = embedder.get_embedder_nerf(
                n_positional_freqs, input_dims=input_dim
            )
            input_dim = pos_embedder_out_dim
            self.n_alpha_epochs = n_alpha_epochs
            self.alpha_const = n_positional_freqs / n_alpha_epochs if n_alpha_epochs > 0 else self.n_positional_freqs

        dims = [latent_size + input_dim] + dims + [output_dim]

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        for l in range(0, self.num_layers - 1):
            if l + 1 in latent_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]
                if self.xyz_in_all and l != self.num_layers - 2:
                    out_dim -= 3

            if weight_norm and l in self.norm_layers:
                setattr(self, "lin" + str(l), nn.utils.weight_norm(nn.Linear(dims[l], out_dim)))
            else:
                print(l, dims[l], out_dim)
                setattr(self, "lin" + str(l), nn.Linear(dims[l], out_dim))

            if (not weight_norm) and self.norm_layers is not None and l in self.norm_layers:
                setattr(self, "bn" + str(l), nn.LayerNorm(out_dim))

        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout

    # input: N x (L+3)
    def forward(self, input, epoch=None):
        xyz = input[:, -3:]

        if hasattr(self, "pos_embedder"):
            alpha = self.alpha_const * epoch if self.n_alpha_epochs > 0 else self.alpha_const
            input_pos_embed = self.pos_embedder(xyz, alpha)
            x = torch.cat([input[:, :-3], input_pos_embed], 1)
            input_embed = x.clone()
        else:
            if input.shape[1] > 3 and self.latent_dropout:
                latent_vecs = input[:, :-3]
                latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
                x = torch.cat([latent_vecs, xyz], 1)
            else:
                x = input

        for l in range(0, self.num_layers - 1):
            
            lin = getattr(self, "lin" + str(l))
            
            if l in self.latent_in:
                if hasattr(self, "pos_embedder"):
                    x = torch.cat([x, input_embed], 1)
                else:
                    x = torch.cat([x, input], 1)
            elif l != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 1)

            x = lin(x)
            
            if l < self.num_layers - 2:
                if self.norm_layers is not None and l in self.norm_layers and not self.weight_norm:
                    bn = getattr(self, "bn" + str(l))
                    x = bn(x)

                x = self.relu(x)

                if self.dropout is not None and l in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        # Apply predicted translation
        xyz_warped = xyz + x

        return xyz_warped, x 


class PoseDecoderSE3(nn.Module):
    def __init__(
        self,
        latent_size,
        dims,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        xyz_in_all=None,
        latent_dropout=False,
        positional_enc=False,
        n_positional_freqs=8,
        n_alpha_epochs=80,
    ):
        super(PoseDecoderSE3, self).__init__()

        input_dim = 3
        output_dim = 9

        if positional_enc:
            self.n_positional_freqs = n_positional_freqs
            self.pos_embedder, pos_embedder_out_dim = embedder.get_embedder_nerf(
                n_positional_freqs, input_dims=input_dim
            )
            input_dim = pos_embedder_out_dim
            self.n_alpha_epochs = n_alpha_epochs
            self.alpha_const = n_positional_freqs / n_alpha_epochs if n_alpha_epochs > 0 else self.n_positional_freqs
        
        dims = [latent_size + input_dim] + dims + [output_dim]

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        for l in range(0, self.num_layers - 1):
            if l + 1 in latent_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]
                if self.xyz_in_all and l != self.num_layers - 2:
                    out_dim -= 3

            if weight_norm and l in self.norm_layers:
                setattr(self, "lin" + str(l), nn.utils.weight_norm(nn.Linear(dims[l], out_dim)))
            else:
                setattr(self, "lin" + str(l), nn.Linear(dims[l], out_dim))

            if (not weight_norm) and self.norm_layers is not None and l in self.norm_layers:
                setattr(self, "bn" + str(l), nn.LayerNorm(out_dim))

        # Initialize last layer from a uniform distribution U(-1e-5, 1e-5) to initialize the deformation near the identity (nerfie)
        lin_last = getattr(self, "lin" + str(self.num_layers - 2))
        torch.nn.init.uniform_(lin_last.weight, a=-1e-5, b=1e-5)

        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout

    # input: N x (L+3)
    def forward(self, input, epoch=None):
        xyz = input[:, -3:]

        if hasattr(self, "pos_embedder"):
            alpha = self.alpha_const * epoch if self.n_alpha_epochs > 0 else self.alpha_const
            input_pos_embed = self.pos_embedder(xyz, alpha)
            x = torch.cat([input[:, :-3], input_pos_embed], 1)
            input_embed = x.clone()
        else:
            if input.shape[1] > 3 and self.latent_dropout:
                latent_vecs = input[:, :-3]
                latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
                x = torch.cat([latent_vecs, xyz], 1)
            else:
                x = input

        for l in range(0, self.num_layers - 1):
            
            lin = getattr(self, "lin" + str(l))
            
            if l in self.latent_in:
                if hasattr(self, "pos_embedder"):
                    x = torch.cat([x, input_embed], 1)
                else:
                    x = torch.cat([x, input], 1)
            elif l != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 1)

            x = lin(x)
            
            if l < self.num_layers - 2:
                if self.norm_layers is not None and l in self.norm_layers and not self.weight_norm:
                    bn = getattr(self, "bn" + str(l))
                    x = bn(x)

                x = self.relu(x)

                if self.dropout is not None and l in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        #######################################################
        # Apply SE(3) transformation to input point xyz
        #######################################################

        # Extract v (rotation), s (pivot point), t (translation)
        v, s, t = x[:, :3], x[:, 3:-3], x[:, -3:]

        # Convert log-quaternion to unit quaternion
        q = kornia.quaternion_log_to_exp(v)

        # Points centered around pivot points s
        xyz_pivot = xyz - s

        # Apply rotation
        xyz_rotated = geometry_utils.rotate_points_with_quaternions(p=xyz_pivot, q=q)

        # Transform back to world space by adding s and also add the additional translation
        xyz_warped = xyz_rotated + s + t

        return xyz_warped, x