# -*- coding: UTF-8 -*-
"""
@Project : SDUCL
@File    : dataset.py
@Author  : Nan Chen
@Date    : 2024-01-15 10:58 
"""
import scanpy as sc
from torch.utils import data
from pathlib import Path
import pandas as pd
import numpy as np
import os
import cv2
from PIL import Image
from anndata import AnnData
import matplotlib.pyplot as plt
from tqdm import tqdm


def read_10X_Visium_DLPFC(path,
                          count_file='filtered_feature_bc_matrix.h5',
                          load_images=True,
                          section_id='',
                          img_size=224
                          ):
    adata = sc.read_visium(path, count_file=count_file, load_images=load_images, )
    adata.var_names_make_unique()

    # ground_truth
    df_meta = pd.read_csv(path + '/metadata.tsv', sep='\t')
    df_meta_layer = df_meta['layer_guess']
    adata.obs['ground_truth'] = df_meta_layer.values
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]

    # cut image
    full_image = cv2.imread(os.path.join(path, 'spatial', 'full_image.tif'))
    full_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)
    save_dir = 'cut_img_DLPFC'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = str(save_dir + '/' + 'cut_img_' + section_id + '.npy')
    print(save_path)
    # save to 'cut_img_DLPFC/cut_img_151507.npy'
    if not os.path.exists(save_path):
        patches = []
        for x, y in adata.obsm['spatial']:
            patch = full_image[y - img_size:y + img_size, x - img_size:x + img_size]
            patches.append(patch)
        patches = np.array(patches)
        np.save(save_path, patches)

    print("cutting image  finished!")

    return adata


