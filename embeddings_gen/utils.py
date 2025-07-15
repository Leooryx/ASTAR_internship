
# ==================
#   PATH CONSTANTS
# ==================
import os
from pathlib import Path

ROOT_DIR = str(Path(__file__).resolve().parents[2])
SRC_DIR = os.path.join(ROOT_DIR, "src")
ASSET_DIR = os.path.join(ROOT_DIR, "assets")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
PROFILING_DIR = os.path.join(ROOT_DIR, "profiling")
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
RUN_DIR = os.path.join(ROOT_DIR, "tensorboard")
SPLIT_DIR = os.path.join(DATA_DIR, "splits")
CONFIG_DIR = os.path.join(ROOT_DIR, "configs")
PATCH_DIR = os.path.join(DATA_DIR, "patched")
METADATA_DIR = os.path.join(DATA_DIR, "metadata")
EMBEDDING_DIR = os.path.join(ROOT_DIR, "embeddings")
PATCH_TABLE_DIR = os.path.join(DATA_DIR, "patch_tables")
BASE_MODEL_DIR = os.path.join(ROOT_DIR, "model_weights")
FULL_PATCH_DIR = os.path.join(DATA_DIR, "patched_full") # only used for downstream analysis
HISTOGRAM_DIR = os.path.join(DATA_DIR, "histogram_matching_stats")
COMBINED_PATCH_DIR = os.path.join(DATA_DIR, "patched_combined")

# ===================
#   LABEL CONSTANTS  
# ===================

LABEL_MAP = {
    "Stroma": 0,
    "Normal": 1,
    "G3":     2,
    "G4":     3,
    "G5":     4
}

COLOR_MAP = {
    0: [242, 182, 216],   
    1: [163, 196, 243],  
    2: [255, 213, 128],   
    3: [190, 224, 200],   
    4: [217, 185, 255],   
}

SEVERITY_COLOR_MAP = {
    0: [255, 0,     255],
    1: [51,  0,     255],
    2: [51,  255,   0  ],
    3: [255, 229.5, 0  ],
    4: [255, 0,     0  ]
}

# ===================
#   MODEL CONSTANTS
# ===================

ENCODER_DIMS = {
    "uni":      1024,
    "gigapath": 1536,
    "virchow":  2560
}

PRETRAINED_ENCODERS = {
    "uni", 
    "gigapath", 
    "virchow"
}

# ===================
#   MISC. CONSTANTS  
# ===================

SCANNERS = {"akoya", "kfbio", "leica", "olympus", "philips", "zeiss"}
GRAPH_COLORS = ["#ff66c4", "#cb6ce6", "#875dca"]
BORDER_WIDTH = 70




# ==================
#   get_args
# ==================

import os
import json
import yaml
import pickle
from typing import Dict, Union, Any

def get_args(args_path: str) -> Dict[str, Union[float | str]]:

    """
    Gets relevant arguments from a yaml file.

    Parameters
    ----------
    args_path: str
        The path to the yaml file containing the arguments.
    
    Returns
    -------    
    args: Dict[str, Union[float, str]]
        The arguments in the form of a dictionary.
    """

    with open(args_path, "r") as f:
        args = yaml.safe_load(f)

    return args


# ==================
#   load_json
# ==================

def load_json(json_path: str):

    """
    Loads a json object from a path.
    """
    
    with open(json_path, "r") as f:
        json_object = json.load(f)

    return json_object




import os
from typing import Dict, Union

import numpy as np
import pandas as pd

def save_table(
    data_dict: Dict[str, Union[str, np.ndarray]],
    save_dir: str,
    filename: str
    ) -> None:

    """
    Converts a dictionary to a pandas DataFrame for downstream analysis.
    """

    os.makedirs(save_dir, exist_ok=True)

    df = pd.DataFrame(data_dict)
    df.to_parquet(os.path.join(save_dir, f"{filename}.parquet"), index=False)


# ==================
#   log_device
# ==================
import pynvml
import torch
import platform
from torch.utils.tensorboard import SummaryWriter

def log_device():

    if not torch.cuda.is_available():
        print("CUDA not available — running on CPU.")
        return

    if platform.system() == "Darwin": 
        print("NVML not supported on macOS — skipping device log.")
        return

    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        border = "=" * BORDER_WIDTH

        print(f"\n{border}")
        print(f"{'Devices':^{BORDER_WIDTH}}")
        print(f"{'-' * BORDER_WIDTH}")

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            name = pynvml.nvmlDeviceGetName(handle)

            print("\n" + f"{f'GPU {i}: {name}':^{BORDER_WIDTH}}")
            print(f"{f'Memory Used : {mem.used / 1024**2:.2f} MB':^{BORDER_WIDTH}}")
            print(f"{f'Memory Total: {mem.total / 1024**2:.2f} MB':^{BORDER_WIDTH}}\n")

        print(f"{'-' * BORDER_WIDTH}")
        print(f"{f'PyTorch CUDA Available: {torch.cuda.is_available()}':^{BORDER_WIDTH}}")
        print(f"{border}\n")

    finally:
        pynvml.nvmlShutdown()






# ==================
#   Network
# ==================

import os

import timm
import torch
import torch.nn as nn
from timm.layers import SwiGLUPacked
from timm.models.vision_transformer import VisionTransformer 


class Network(nn.Module):

    """
    Initializes the network with a foundation model as the encoder
    and a linear layer as the classifier.

    Parameters
    ----------
    encoder_name: str
        The foundation model to be used as the encoder.
        One of [uni, gigapath, virchow].

    encoder_dir: str
        The directory containing the encoder weights.

    num_classes: int
        The number of classes to be classified.
    
    freeze_encoder: bool
        Whether to freeze the encoder during finetuning.
    """

    def __init__(
        self,
        encoder_name: str,
        encoder_dir: str,
        num_classes: int = 2,
        freeze_encoder: bool = True
        ):
        super().__init__()

        self.encoder = get_encoder(encoder_name, encoder_dir)
        self.fc = get_classification_head(encoder_name, num_classes)
        self.freeze_encoder = freeze_encoder

        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        if self.freeze_encoder:
            with torch.no_grad():
                embedding = self.encoder(x)

        else:
            embedding = self.encoder(x)
            
        logits = self.fc(embedding)

        return logits


class ClassificationHead(nn.Module):

    """
    Initializes a linear classification head.

    Parameters
    ----------
    in_dim: int
        The input dimension of the classifier.

    out_dim: int
        The output dimension of the classifier or 
        the number of classes.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.head = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        logits = self.head(x)

        return logits
    

class VirchowConcat(nn.Module):
    
    """
    Performs the concatenation of the CLS token and mean across patch
    tokens as defined in Virchow.

    Source: https://arxiv.org/pdf/2309.07778

    Parameters
    ----------
    encoder: nn.Module
        The Virchow encoder. 
    """
    
    def __init__(
        self, 
        encoder: nn.Module
        ):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        all_tokens = self.encoder(x)

        cls_token = all_tokens[:, 0]
        patch_tokens = all_tokens[:, 1:].mean(1)

        embedding = torch.cat([cls_token, patch_tokens], dim=-1)

        return embedding


def download_weights(
    encoder_name: str,
    encoder_dir: str
    ):

    """
    Downloads the weights of a foundation model to a selected directory.

    Parameters
    ----------
    encoder_name: str
        The foundation model to be used as the encoder.
        One of [uni, gigapath, virchow].

    encoder_dir: str
        The directory containing the encoder weights.
    """

    if encoder_name not in ENCODER_DIMS:
        raise ValueError(f"encoder must be one of {ENCODER_DIMS}")

    if encoder_name == "uni":
        encoder = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)

    if encoder_name == "gigapath":
        encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True, dynamic_img_size=True)

    if encoder_name == "virchow":
        encoder = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)

    encoder_path = os.path.join(encoder_dir, f"{encoder_name}.pth")
    torch.save(encoder, encoder_path)


def get_encoder(
    encoder_name: str,
    encoder_dir: str
    ) -> VisionTransformer:

    """
    Returns an initialized foundation encoder.

    Parameters
    ----------
    encoder_name: str
        The foundation model to be used as the encoder.
        One of [uni, gigapath, virchow].

    encoder_dir: str
        The directory containing the encoder weights.

    Returns
    -------
    encoder: VisionTransformer
        The initialized foundation encoder.
    """

    if encoder_name not in ENCODER_DIMS:
        raise ValueError(f"encoder must be one of {list(ENCODER_DIMS.keys())}.")
    
    os.makedirs(encoder_dir, exist_ok=True)
    encoder_path = os.path.join(encoder_dir, f"{encoder_name}.pth")

    if not os.path.isfile(encoder_path):
        download_weights(encoder_name, encoder_dir=encoder_dir)

    encoder = torch.load(encoder_path, map_location=torch.device("cpu"), weights_only=False)

    if encoder_name == "virchow":
        encoder = VirchowConcat(encoder=encoder)

    return encoder

    
def get_classification_head(
    encoder_name: str,
    num_classes: int
    ) -> ClassificationHead:

    """
    Initializes the appropriate classification head
    according to a selected foundation encoder.

    Parameters
    ----------
    encoder_name: str
        The foundation model to be used as the encoder.
        One of [uni, gigapath, virchow].

    num_classes: int
        The number of output classes.
    """

    if encoder_name not in ENCODER_DIMS:
        raise ValueError(f"encoder must be one of {ENCODER_DIMS}")

    in_dims = ENCODER_DIMS[encoder_name]
    head = ClassificationHead(in_dim=in_dims, out_dim=num_classes)

    return head




# ==================
#   NetworkHandler
# ==================


import os
from typing import (
    Tuple, 
    Literal
)

import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from deeplake import Dataset as DeepLakeDataset
from sklearn.metrics import balanced_accuracy_score


class NetworkHandler:

    """
    This class encapsulates all computation logic for a given neural network,
    including training, validation, inference and embedding extraction.
    
    Supports mixed precision training.

    Parameters
    ----------
    model: Network
        The neural network.

    criterion: nn.Module
        The function for loss computation.

    optimizer: torch.optim.Optimizer
        The optimizer for gradient descent.

    precision: Literal['single', 'mixed']
        Whether to train in mixed or single precision.
        Must be one of ['single', 'mixed'].

    freeze_encoder: bool
        Whether the encoder is frozen.
        Will be used as a flag in switching between train and eval modes.

    embedding_mode: bool
        Whether to perform computations on pre-extracted embeddings.
    """

    def __init__(
        self,
        model: Network,
        criterion: nn.Module = None,
        optimizer: torch.optim.Optimizer = None, 
        precision: Literal["single", "mixed"] = "single",
        freeze_encoder: bool = True,
        embedding_mode: bool = False
        ):

        valid_precisions = ["single", "mixed"]

        if precision not in valid_precisions:
            raise ValueError(f"precision must be one of  {valid_precisions}.")

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.freeze_encoder = freeze_encoder
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_amp = precision == "mixed" and self.device == "cuda"
        self.grad_scaler = GradScaler(enabled=self.use_amp)
        self.model = self.model.to(self.device)
        self.embedding_mode = embedding_mode

        if self.device != "cuda" and precision == "mixed":
            raise ValueError(f"Mixed precision unavailable with current device: {self.device}. Switch to single precision.\n")


    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:

        """
        Trains the model for 1 epoch.

        Parameters
        ----------
        train_loader: DataLoader
            The data loader for training.

        Returns
        -------
        epoch_loss: float
            The loss for the epoch.

        epoch_balanced_accuracy: float
            The average balanced accuracy for the given epoch.  
        """
        
        metrics = {
            "running_loss": 0,
            "predictions": [],
            "targets": []
        }
        
        self.model.train()
        if self.freeze_encoder or self.embedding_mode: self.model.encoder.eval()

        pbar = tqdm(train_loader, desc="Training in progress")

        for patch, target, *_ in pbar:
            patch = patch.to(self.device)
            target =  target.to(self.device)

            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.use_amp):
                logits = self.model.fc(patch) if self.embedding_mode else self.model(patch) 
                loss = self.criterion(logits, target)

            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.optimizer.zero_grad()

            confidence = F.softmax(logits, dim=1)
            pred = torch.argmax(confidence, dim=1)

            metrics["running_loss"] += loss.detach().cpu().item()
            metrics["predictions"].extend(pred.cpu().numpy())
            metrics["targets"].extend(target.cpu().numpy())

            pbar.set_postfix({"step_loss": loss.detach().cpu().item()})

        epoch_loss = metrics["running_loss"] / len(train_loader)
        epoch_balanced_accuracy = balanced_accuracy_score(metrics["targets"], metrics["predictions"])

        return epoch_loss, epoch_balanced_accuracy
    

    @torch.no_grad()
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:

        """
        Runs validation for 1 epoch.

        Parameters
        ----------
        val_loader: DataLoader
            The data loader for validation.

        Returns
        -------
        epoch_loss: float
            The loss for the epoch.

        epoch_balanced_accuracy: float
            The average balanced accuracy for the given epoch.  
        """

        metrics = {
            "running_loss": 0,
            "predictions": [],
            "targets": []
        }

        self.model.eval()
        pbar = tqdm(val_loader, desc="Validation in progress")
        for patch, target, *_ in pbar:
            patch = patch.to(self.device)
            target = target.to(self.device)

            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.use_amp):
                logits = self.model.fc(patch) if self.embedding_mode else self.model(patch)
                loss = self.criterion(logits, target)

            confidence = F.softmax(logits, dim=1)
            pred = torch.argmax(confidence, dim=1)

            metrics["running_loss"] += loss.detach().cpu().item()
            metrics["predictions"].extend(pred.cpu().numpy())
            metrics["targets"].extend(target.cpu().numpy())

            pbar.set_postfix({"step_loss": loss.detach().cpu().item()})

        epoch_loss = metrics["running_loss"] / len(val_loader)
        epoch_balanced_accuracy = balanced_accuracy_score(metrics["targets"], metrics["predictions"])

        return epoch_loss, epoch_balanced_accuracy
    
    @torch.no_grad()
    def inference(
        self, 
        inference_loader: DataLoader, 
        save_dir: str = None,
        filename: str = None
        ) -> Tuple[float, float]:

        """
        Performs inference and optionally saves the results as a parquet table
        for downstream analysis.

        Parameters
        ----------
        inference_loader: DataLoader
            The data loader for inference.

        save_dir: str
            The directory to save results.

        filename: str
            The filename to save the results into.

        Returns
        -------
        iteration_loss: float
            The average loss during inference.

        iteration_balanced_accuracy:
            The average balanced accuracy during inference.
        """

        if save_dir and not filename:
            raise ValueError("filename cannot be empty if save dir is specified.")
        
        if filename and not save_dir:
            raise ValueError(f"save_dir must be provided to save results as {filename}")

        metrics = {
            "loss": [],
            "confidence_score": [],
            "prediction": [],
            "target": [],
            "area": [],
            "x": [],
            "y": [],
            "w": [],
            "h": [],
            "img_idx": []
        }

        self.model.eval()
        pbar = tqdm(inference_loader, desc="Inference in progress")
        for patch, target, metadata in pbar:
            patch = patch.to(self.device)
            target = target.to(self.device)

            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.use_amp):
                logits = self.model.fc(patch) if self.embedding_mode else self.model(patch)
                loss = self.criterion(logits, target)

            confidence = F.softmax(logits, dim=1)
            pred = torch.argmax(confidence, dim=1)

            metrics["loss"].extend(loss.detach().cpu().numpy())
            metrics["confidence_score"].extend(confidence.cpu().numpy())
            metrics["prediction"].extend(pred.cpu().numpy())
            metrics["target"].extend(target.cpu().numpy())
            
            metrics["area"].extend(metadata["area"].cpu().numpy())
            metrics["x"].extend(metadata["x"].cpu().numpy())
            metrics["y"].extend(metadata["y"].cpu().numpy())
            metrics["w"].extend(metadata["w"].cpu().numpy())
            metrics["h"].extend(metadata["h"].cpu().numpy())
            metrics["img_idx"].extend(metadata["img_idx"].cpu().numpy())

        iteration_loss = sum(metrics["loss"]) / len(metrics["loss"])
        iteration_balanced_accuracy = balanced_accuracy_score(metrics["target"], metrics["prediction"])

        if save_dir and filename:
            save_table(metrics, save_dir, filename)

        return iteration_loss, iteration_balanced_accuracy
    
    @torch.no_grad()
    def predict(
        self,
        pred_loader: DataLoader,
        save_dir: str = None,
        filename: str = None
        ):
    
        """
        This method is used when the dataset contains samples with no labels.
        Outputs predictions for each patch without performing evaluation.

        Optionally, will save results as a parquet table for downstream analysis.

        Parameters
        ----------
        pred_loader: DataLoader
            The data loader to iterate over.
        
        save_dir: str
            The directory to save results.

        filename: str
            The filename to save the results into.
        """

        if save_dir and not filename:
            raise ValueError("filename cannot be empty if save dir is specified.")
        
        if filename and not save_dir:
            raise ValueError(f"save_dir must be provided to save results as {filename}")
        
        metrics = {
            "confidence_score": [],
            "prediction": [],
            "target": [],
            "area": [],
            "x": [],
            "y": [],
            "w": [],
            "h": []
        }

        self.model.eval()
        pbar = tqdm(pred_loader, desc="Prediction in progress")
        for patch, target, metadata in pbar:
            patch = patch.to(self.device)
            target = target.to(self.device)

            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.use_amp):
                logits = self.model.fc(patch) if self.embedding_mode else self.model(patch)
            
            confidence = F.softmax(logits, dim=1)
            pred = torch.argmax(confidence, dim=1)

            metrics["confidence_score"].extend(confidence.cpu().numpy())
            metrics["prediction"].extend(pred.cpu().numpy())
            metrics["target"].extend(target.cpu().numpy())
            
            metrics["area"].extend(metadata["area"].cpu().numpy())
            metrics["x"].extend(metadata["x"].cpu().numpy())
            metrics["y"].extend(metadata["y"].cpu().numpy())
            metrics["w"].extend(metadata["w"].cpu().numpy())
            metrics["h"].extend(metadata["h"].cpu().numpy())

        if save_dir and filename:
            save_table(metrics, save_dir, filename)

    @torch.no_grad()
    def extract_embeddings(
        self, 
        embed_loader: DataLoader,
        deeplake_ds: DeepLakeDataset,
        img_idx: int
        ):

        """
        Extracts and saves embeddings in a deep lake dataset.

        Assumes the existence of three inputs:
            - image patch
            - label
            - file key (an index that maps to a file to trace each embedding back to the original image)

        Parameters
        ----------
        embed_loader: DataLoader
            The data loader to iterate throught the dataset.

        deeplake_ds: DeepLakeDataset
            The deeplake dataset to store the embeddings.

        img_idx: int
            The index id of the slide associated with the patch.
        """

        self.model.eval()
        pbar = tqdm(embed_loader, desc="Extracting embeddings")
        for patch, label, metadata in pbar:
            patch = patch.to(self.device)

            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.use_amp):
                embedding = self.model.encoder(patch)

            embedding = embedding.detach().cpu()

            area = metadata["area"]
            x = metadata["x"]
            y = metadata["y"]
            w = metadata["w"]
            h = metadata["h"]
            img_idx_batched = np.broadcast_to(img_idx, x.shape)

            deeplake_ds.append({
                "embedding": embedding.numpy(),
                "label": label.numpy(),
                "area": area.numpy(),
                "x": x.numpy(),
                "y": y.numpy(),
                "w": w.numpy(),
                "h": h.numpy(),
                "img_idx": img_idx_batched
            })

def save_checkpoint(
    save_dir: str,
    model: nn.Module, 
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    balanced_accuracy: torch.Tensor | float,
    loss: torch.Tensor | float,
    min_val_loss: torch.Tensor | float,
    max_val_accuracy: torch.Tensor | float
    ):

    training_state = {
        "model": model.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "balanced_accuracy": balanced_accuracy,
        "loss": loss,
        "min_val_loss": min_val_loss,
        "max_val_accuracy": max_val_accuracy
    }

    torch.save(training_state, os.path.join(save_dir, "checkpoint.pth"))



# ==================
#   img_transform_fn
# ==================

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

#from .chunk_helpers import foreground_patch
#from .image_migration_helpers import match_distribution

class ToPILCheck:
    def __call__(self, img):
        if isinstance(img, Image.Image):
            return img

        else:
            return transforms.ToPILImage()(img)
        

class RGBCheck:
    def __call__(self, img: Image):
        return img.convert("RGB")
def normalize(img: np.ndarray | Image.Image) -> torch.Tensor:
    
    img_transform = transforms.Compose([
        ToPILCheck(),
        RGBCheck(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    transformed = img_transform(img)

    return transformed

def img_transform_fn(
    row: Dict[str, Any],
    apply_augmentation: bool = False,
    preprocess_fn: Optional[Callable] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:

    """
    Performs patch-level processing for embedding extraction.
    """

    img = row["patch"].copy()
    
    if preprocess_fn is not None:
        img = preprocess_fn(img)

    if apply_augmentation:
        img = augment_fn(img)

    img = normalize(img)
    label = torch.tensor(row["label"], dtype=torch.long)
    
    area = torch.tensor(row["area"], dtype=torch.long)
    x = torch.tensor(row["x"], dtype=torch.long)
    y = torch.tensor(row["y"], dtype=torch.long)
    w = torch.tensor(row["w"], dtype=torch.long)
    h = torch.tensor(row["h"], dtype=torch.long)

    metadata = {
        "area": area,
        "x": x,
        "y": y,
        "w": w,
        "h": h
    }

    return img, label, metadata