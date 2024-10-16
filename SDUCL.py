# -*- coding: UTF-8 -*-
"""
@Project : SDULC
@File    : SDUCL.py
@Author  : Nan Chen
@Date    : 2024-01-15
"""

import torch
from torch import nn
import torch.nn.functional as F
from model import Encoder
from preprocess import (
    set_random_seed, preprocess_data, generate_interaction_matrix, 
    extract_spot_features, normalize_adjacency_matrix, get_osfs_image_features, 
    add_contrastive_labels, get_combined_features, apply_permutation, get_DLPFC_image_features
)
from signal_diffusion import signal_diffuse_torch


class SDUCLModel():
    def __init__(self, 
                 adata, 
                 device=torch.device('cpu'), 
                 lr=0.001, 
                 weight_decay=0.01, 
                 num_epochs=1000, 
                 permutation_ratio=1.0, 
                 hidden_dim=512, 
                 output_dim=64, 
                 seed=41, 
                 alpha=5, 
                 beta=1.5, 
                 diffusion_ratio=0.05, 
                 data_type='DLPFC', 
                 section_id=None):
        """
        init
        """
        self.adata = adata.copy()
        self.section_id = section_id
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.permutation_ratio = permutation_ratio
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seed = seed
        self.alpha = alpha
        self.beta = beta
        self.diffusion_ratio = diffusion_ratio
        self.data_type = data_type

        set_random_seed(self.seed)
        self._preprocess_data()
        self.features = get_combined_features(self.adata).to(self.device)
        self.input_dim = len(self.features[0])
        self.permuted_features = apply_permutation(self.features, self.permutation_ratio)
        self.adjacency_matrix = self._get_normalized_adjacency()
        self.sub_diffusion_matrix = self._apply_signal_diffusion()

    def _preprocess_data(self):
        if 'highly_variable' not in self.adata.var:
            self.adata = preprocess_data(self.adata)

        if 'feat_spot' not in self.adata.obsm:
            extract_spot_features(self.adata)

        if self.data_type == 'osfs':
            if 'feat_img' not in self.adata.obsm:
                get_osfs_image_features(self.adata, device=self.device, section_id=self.section_id)
        elif self.data_type == "DLPFC":
            if 'feat_img' not in self.adata.obsm:
                get_DLPFC_image_features(self.adata, device=self.device, section_id=self.section_id)

        if 'adj_spot' not in self.adata.obsm:
            generate_interaction_matrix(self.adata, x=self.diffusion_ratio)

        if 'label_CSL' not in self.adata.obsm:
            add_contrastive_labels(self.adata)

    def _get_normalized_adjacency(self):
        adjacency = self.adata.obsm['adj_spot']
        normalized_adjacency = normalize_adjacency_matrix(adjacency)
        return torch.FloatTensor(normalized_adjacency).to(self.device)

    def _apply_signal_diffusion(self):
        num = int(len(self.adjacency_matrix) * self.diffusion_ratio)
        return signal_diffuse_torch(self.adjacency_matrix, num=num)

    def train_model(self):
        model = Encoder(self.input_dim, self.output_dim, self.sub_diffusion_matrix).to(self.device)

        loss_function = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        model.train()
        for epoch in range(self.num_epochs):
            model.train()

            _, embeddings, pred_pos, pred_neg = model(self.features, self.permuted_features, self.adjacency_matrix)

            loss_pos = loss_function(pred_pos, torch.FloatTensor(self.adata.obsm['label_CSL']).to(self.device))
            loss_neg = loss_function(pred_neg, torch.FloatTensor(self.adata.obsm['label_CSL']).to(self.device))
            loss_feat_res = F.mse_loss(self.features, embeddings)
            total_loss = self.alpha * loss_feat_res + self.beta * (loss_pos + loss_neg)

            optimizer.zero_grad()
            total_loss.backward(retain_graph=True)
            optimizer.step()

        print("Training complete!")

        with torch.no_grad():
            model.eval()
            embeddings = model(self.features, self.permuted_features, self.adjacency_matrix)[1].detach().cpu().numpy()
            self.adata.obsm['embeddings'] = embeddings
            return self.adata

