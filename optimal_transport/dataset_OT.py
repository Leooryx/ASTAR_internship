# how to handle the different datasets for training?
# just define the dataset pytorch for the embeddings
# i have Akoya, KFBio but not Leica

import os
import multiprocessing
from pathlib import Path
from datetime import datetime
from functools import partial
from collections import defaultdict
from typing import (
    Any,
    Dict, 
    List,
    Tuple,
    Iterable,
    Optional,
    Callable
)

import torch
import pyvips
import deeplake
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset

from .chunk_helpers import foreground_patch
from .image_migration_helpers import match_distribution









#fusion between Eric and OT repositories

#what batch size?

# lets make a class that direclty handles deeplake dataset


import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
dataset_path_akoya_1 = f"/home/leolr-int/nfs/data/data/patched/dim_256/Train/Subset3_Train_1_Akoya"
akoya_1 = deeplake.open_read_only(dataset_path_akoya_1)
axes[0].imshow(akoya_1[200]["patch"])
#plt.show()
dataset_path_KFbio_1 = f"/home/leolr-int/nfs/data/data/patched/dim_256/Train/Subset3_Train_1_KFBio"
KFBio_1 = deeplake.open_read_only(dataset_path_KFbio_1)
axes[1].imshow(KFBio_1[200]["patch"])
plt.suptitle(f"Subset3_Train_1_ Akoya and KFBio")
plt.savefig("/home/leolr-int/nfs/ASTAR_internship/Fourier_Domain_Adaptation/images/akoya_vs_KFBio.svg")
plt.show()

class embeddings_loader(Dataset):
    '''
    Class to handle embeddings stored as deeplake objects
    '''
    def __init__(train_or_test, WSI_id, scanner):
        self.split = train_or_test
        self.WSI_id = WSI_id
        self.scanner = scanner
        directory = f"/home/leolr-int/nfs/data/data/patched/dim_256/{train_or_test}"
        file = f'Subset3_{train_or_test}_{WSI_id}_{scanner}'
        self.embedding = deeplake.open_read_only(f'{directory}/{file}')
        
        #attention la je fais les images et pas du tout les embeddings!

#example of the deeplake dataset for embeddings
        embed_ds = deeplake.create(save_dir)
        embed_ds.add_column("embedding", dtype=deeplake.types.Embedding(embedding_dim))
        embed_ds.add_column("label", dtype=deeplake.types.Int32)

        # metadata cols
        embed_ds.add_column("area", dtype=deeplake.types.Int32)
        embed_ds.add_column("x", dtype=deeplake.types.Int32)
        embed_ds.add_column("y", dtype=deeplake.types.Int32)
        embed_ds.add_column("w", dtype=deeplake.types.Int32)
        embed_ds.add_column("h", dtype=deeplake.types.Int32)
        embed_ds.add_column("img_idx", dtype=deeplake.types.Int32)


def linear_probing(weights_path, dataset):
    
    state_dict = torch.load(weights_path)

    weight = state_dict['head.weight'] # shape: [5, 1536]
    bias = state_dict.get('head.bias', None)  # shape: [5]
   
    EMBEDDING_DIR = f"/home/leolr-int/nfs/transformed_data/new_embeddings/{dataset}"  # or KFBio, etc.
    embedding_path = os.path.join(EMBEDDING_DIR, "mixed_precision", "dim_256", "Train", "gigapath")

    ds = deeplake.open_read_only(embedding_path)
    ds_torch = ds.pytorch(transform=embedding_transform_fn)
    ds_loader = DataLoader(ds_torch, batch_size=64, shuffle=False)  # larger batch_size for efficiency
    weight = weight.cpu()
    bias = bias.cpu()

    all_preds = []
    all_labels = []

    for batch in tqdm(ds_loader, desc="Linear probing"):
        embedding, label, _ = batch  
        
        # Linear probing: logits = embedding @ W^T + b
        logits = embedding @ weight.T + bias

        preds = torch.argmax(F.softmax(logits, dim=1), dim=1)

        all_preds.append(preds)
        all_labels.append(label)

def main():
    log_device()
    num_workers = max(1, (os.cpu_count() // 4))

    arg_path = os.path.join(CONFIG_DIR, "linear_probe.yaml")
    args = get_args(arg_path)
    set_seed(args["seed"])

    ds_path = os.path.join(EMBEDDING_DIR, f"{args['precision']}_precision", f"dim_{args['patch_dim']}", "Train", args["encoder"])
    encoder_dir = os.path.join(BASE_MODEL_DIR, "pre_trained_weights")

    log_dir = os.path.join(
        RUN_DIR,
        "linear_probe",
        f"{args['precision']}_precision",
        f"dim_{args['patch_dim']}",
        args["encoder"],
        f"split_{args['split_num']}",
        f"experiment_{args['experiment_num']}"
    )
    writer = SummaryWriter(log_dir)
    save_args(args, log_dir)

    model_dir = os.path.join(
        BASE_MODEL_DIR,
        "linear_probe_weights",
        f"{args['precision']}_precision",
        f"dim_{args['patch_dim']}",
        args["encoder"],
        f"split_{args['split_num']}",
        f"experiment_{args['experiment_num']}"
    )
    os.makedirs(model_dir, exist_ok=True)

    id_path = os.path.join(METADATA_DIR, "id_table.json")
    split_path = os.path.join(SPLIT_DIR, f"dim_{args['patch_dim']}", f"split_{args['split_num']}", "split.json")

    id_table = load_json(id_path)
    split_table = load_json(split_path)

    train_indices = tuple(id_table[slide_id] for slide_id in split_table["train"])
    val_indices = tuple(id_table[slide_id] for slide_id in split_table["val"])

    ds = deeplake.open_read_only(ds_path)
    
    train_dataset = ds.query(
        f"SELECT * WHERE img_idx in {train_indices}"
    ).pytorch(transform=embedding_transform_fn)
    
    val_dataset = ds.query(
        f"SELECT * WHERE img_idx in {val_indices}"
    ).pytorch(transform=embedding_transform_fn)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args["batch_size"],
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args["batch_size"],
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
        num_workers=num_workers
    )



#how the data is accessed to generate embeddings:
# embedding extraction
def main():
    log_device()
    env_path = os.path.join(ROOT_DIR, ".env")
    load_dotenv(env_path)

    arg_path = os.path.join(CONFIG_DIR, "embed.yaml")
    args = get_args(arg_path)

    id_path = os.path.join(METADATA_DIR, "id_table.json")
    id_table = load_json(id_path)

    # path initialization for patched datasets
    train_dir = os.path.join(PATCH_DIR, f"dim_{args['patch_dim']}", "Train")
    test_dir = os.path.join(PATCH_DIR, f"dim_{args['patch_dim']}", "Test")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    train_ds_paths = [os.path.join(train_dir, d) for d in os.listdir(train_dir)]
    test_ds_paths = [os.path.join(test_dir, d) for d in os.listdir(test_dir)]

    dest_dir = os.path.join(EMBEDDING_DIR, f"{args['precision']}_precision", f"dim_{args['patch_dim']}")

split_table = {
        "Train": train_ds_paths,
        "Test": test_ds_paths
    }

    speed_table = {
        "slide_id": [],
        "num_patches": [],
        "start_time": [],
        "end_time": [],
        "elapsed_time": []
    }


    # clean memory
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    
    # encoder initialization
    encoder_dir = os.path.join(BASE_MODEL_DIR, "pre_trained_weights")
    model = Network(args["encoder"], encoder_dir)
    network_handler = NetworkHandler(model, precision=args["precision"])
    embedding_dim = model.fc.head.in_features

    for split in split_table.keys():
        # logging
        split_log = f"Extracting embeddings for {split}...".center(BORDER_WIDTH)
        border = "-" * BORDER_WIDTH
        print(f"\n{border}\n  {split_log}  \n{border}\n")

        split_paths = split_table[split]
        save_dir = os.path.join(dest_dir, split, args["encoder"])
        os.makedirs(save_dir, exist_ok=True)
        num_samples = len(split_paths)
        for i, ds_path in enumerate(split_paths):
            img_id = Path(ds_path).name

            # this is used for the new dataset i generated
            #img_id = img_id.split("_FDA_L=")[0] + '_KFBio'

            if scanner not in img_id:
                continue  # Skip files that don't contain the scanner

            img_idx = id_table[img_id]

            print(f"{img_id} | [{i+1}/{num_samples}]")
            print(ds_path)
            
            try:
                patch_ds = deeplake.open_read_only(ds_path).pytorch(transform=img_transform_fn)
                patch_loader = DataLoader(
                    patch_ds,
                    batch_size=args["batch_size"],
                    shuffle=False
                )

class ImageDataset(Dataset):
    def __init__(self, root,domain, lr_transforms=None, hr_transforms=None):
 
        dataset = get_dataset(dataset="camelyon17",root_dir=root, download=False)
        # Get the training set
        self.train_data = dataset.get_subset(
            domain
        )
        self.domain = domain

       

    def __getitem__(self, index):
        
        img,y,meta = self.train_data[index%len(self.train_data)]
        img = np.array(img,dtype = 'float32')[:,:,:]/255
        
        Transforms = [  
                        transforms.ToTensor(),
                        transforms.Normalize((.5,0.5,0.5), (0.5,0.5,0.5))
                        ]
    
        T = transforms.Compose(Transforms)
        img = T(img)

        return img,y,self.domain

    def __len__(self):
        return len(self.train_data)






def embedding_transform_fn(row: Dict[str, Any]) -> Tuple[torch.Tensor]:

    """"
    Performs patch-level processing.
    """

    embedding = torch.tensor(row["embedding"])
    label = torch.tensor(row["label"], dtype=torch.long)
    
    area = torch.tensor(row["area"], dtype=torch.long)
    x = torch.tensor(row["x"], dtype=torch.long)
    y = torch.tensor(row["y"], dtype=torch.long)
    w = torch.tensor(row["w"], dtype=torch.long)
    h = torch.tensor(row["h"], dtype=torch.long)
    img_idx = torch.tensor(row["img_idx"], dtype=torch.long)

    metadata = {
        "area": area,
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "img_idx": img_idx
    }

    return embedding, label, metadata





