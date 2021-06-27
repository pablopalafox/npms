#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.
# Updates by Pablo Palafox 2021

import torch.nn as nn
import torch
import torch.nn.functional as F

from utils import embedder


class ShapeDecoder(nn.Module):
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
        use_tanh=False,
        latent_dropout=False,
        positional_enc=False,
        n_positional_freqs=8,
    ):
        super(ShapeDecoder, self).__init__()

        input_dim = 3
        output_dim = 1
        
        if positional_enc:
            self.n_positional_freqs = n_positional_freqs
            self.pos_embedder, pos_embedder_out_dim = embedder.get_embedder_nerf(
                self.n_positional_freqs, input_dims=input_dim, i=0
            )
            dims = [latent_size + pos_embedder_out_dim] + dims + [output_dim]
        else:
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
                    out_dim -= input_dim

            if weight_norm and l in self.norm_layers:
                setattr(self, "lin" + str(l), nn.utils.weight_norm(nn.Linear(dims[l], out_dim)))
            else:
                setattr(self, "lin" + str(l), nn.Linear(dims[l], out_dim))

            if (not weight_norm) and self.norm_layers is not None and l in self.norm_layers:
                setattr(self, "bn" + str(l), nn.LayerNorm(out_dim))

        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

    # input: N x (L+3)
    def forward(self, input, epoch=None):
        xyz = input[:, -3:]

        if hasattr(self, "pos_embedder"):
            input_pos_embed = self.pos_embedder(xyz, self.n_positional_freqs)
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

        if hasattr(self, "th"):
            x = self.th(x)

        return x

    
class ShapeDecoderCurriculum(nn.Module):
    def __init__(
        self,
        latent_size,
        curriculum_dist,
        epochs,
        dims,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        xyz_in_all=None,
        use_tanh=False,
        latent_dropout=False,
        positional_enc=False,
    ):
        assert not positional_enc, "Positional encoding not implemented for Curriculum DeepSDF"

        super(ShapeDecoderCurriculum, self).__init__()

        input_dim = 3
        output_dim = 1

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
                    out_dim -= input_dim

            if weight_norm and l in self.norm_layers:
                setattr(self, "lin" + str(l), nn.utils.weight_norm(nn.Linear(dims[l], out_dim)))
            else:
                setattr(self, "lin" + str(l), nn.Linear(dims[l], out_dim))

            if (not weight_norm) and self.norm_layers is not None and l in self.norm_layers:
                setattr(self, "bn" + str(l), nn.LayerNorm(out_dim))

        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

        self.curriculum_epochs = [int(ce * epochs) for ce in curriculum_dist]
        print("curriculum_epochs", self.curriculum_epochs)
        
        self.fade_in_epochs = int(0.1 * epochs)
        
        self.lbase_epochs = [(ce + self.fade_in_epochs) for ce in self.curriculum_epochs]
        print("lbase_epochs", self.lbase_epochs)


    # input: N x (L+3)
    def forward(self, input, epoch):
        xyz = input[:, -3:]

        if input.shape[1] > 3 and self.latent_dropout:
            latent_vecs = input[:, :-3]
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            x = torch.cat([latent_vecs, xyz], 1)
        else:
            x = input
        
        if epoch <= self.curriculum_epochs[0]: # [0, 200]
            ldim = 0
            lbase = -100
        elif epoch <= self.curriculum_epochs[1]: # [200, 600]
            ldim = 1
            lbase = self.lbase_epochs[0]
        elif epoch <= self.curriculum_epochs[2]: # [600, 1000]
            ldim = 2
            lbase = self.lbase_epochs[1]
        else: # [1000, 2000]
            ldim = 3
            lbase = self.lbase_epochs[2]

        # First 4 layers
        for l in range(0, 3):
            lin = getattr(self, "lin" + str(l))
            
            xi = x
            
            if l != 0 and self.xyz_in_all:
                xi = torch.cat([xi, xyz], 1)
            
            x = lin(xi)
            
            if self.norm_layers is not None and l in self.norm_layers and not self.weight_norm: 
                bn = getattr(self, "bn" + str(l))  
                x = bn(x)
            
            x = self.relu(x)
            
            if self.dropout is not None and l in self.dropout:    
                x = F.dropout(x, p=self.dropout_prob, training=self.training)

        # Last 4 layers
        for l in range(3, ldim + 5):
            lin = getattr(self, "lin" + str(l))
            
            xi = x
            
            if l in self.latent_in:
                xi = torch.cat([xi, input], 1)
            elif l != 0 and self.xyz_in_all:
                xi = torch.cat([xi, xyz], 1)

            # add the last layer as residual block (with a linear fade-in) to have a smooth transition
            if l == ldim + 4 and ldim > 0 and lbase - epoch > 0: 
                x = xi * max([0, float(lbase - epoch) / self.fade_in_epochs]) + lin(xi) * min([1, float(epoch - lbase) / self.fade_in_epochs + 1])
            else:
                x = lin(xi)
            
            if self.norm_layers is not None and l in self.norm_layers and not self.weight_norm:
                bn = getattr(self, "bn" + str(l))
                x = bn(x)
            
            x = self.relu(x)
            
            if self.dropout is not None and l in self.dropout:
                x = F.dropout(x, p=self.dropout_prob, training=self.training)
        
        # Last layer
        l = 8
        lin = getattr(self, "lin8")
        xi = x
        
        if self.xyz_in_all:
            xi = torch.cat([xi, xyz], 1)
        
        x = lin(xi)

        if hasattr(self, "th"):
            x = self.th(x)

        return x