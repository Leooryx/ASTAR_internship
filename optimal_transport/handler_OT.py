# TODO: handle when different scanners than Leica are given for training while not increasing too much the size of data for one batch!
# TODO: faire une fonction display en dehors de la classe que je peux mettre facilement partout
# TODO: the code assumes im working with the patch images direclty, but sometimes patch is used for the patch embeddings!! il faut corriger ca
# TODO: concernant les batch --> mettre le nom de toutes les variables au pluriel
# TODO: data augmentation strategies?? (last)
# TODO: way too many repetitions between train and validation --> completely possible to create one function with a little if loop to compute or not gradient

# entrainer sur Akoya et Leica a chaque fois


        # attention, ici il faudra faire la distinction de cas selon que l'on applique les couches OT ou non !
        #def OT_layers() define somewhere else
        # self.bottle_neck, self.head = OT_layers(OT)


import os
from typing import (
    Tuple, 
    Literal
)
import random
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from deeplake import Dataset as DeepLakeDataset
from sklearn.metrics import balanced_accuracy_score

#from .network import Network
#from .save import save_table

from dataset_OT import multi_WSI_loader 
from architecture_OT import Neural_Network
from geomloss import SamplesLoss


# Ensuring reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False 

# Define global variables
metrics_train = {'running_loss':0, 'predictions':[], 'labels':[]}
metrics_val = {'running_loss':0, 'predictions':[], 'labels':[]}
WSI_ids_train = [1,2,3, 4, 5, 6, 7, 8, 9, 10] #to be completed
WSI_ids_val = [11,12,13,14,15]

# Defining OT-based loss function
loss_geom = SamplesLoss('sinkhorn', p=2, blur=0.1, scaling=0.95, verbose=False)
Lambda = 0.1 # strength of OT (0.1 is the value of the article)

class NetworkHandler:
    '''
    A class to handle training, inference and prediction
    '''

    def __init__(self, precision = 'mixed', freeze_encoder = True, embedding_mode = False, display = False):
        self.precision = precision
        self.freeze_encoder = freeze_encoder
        self.embedding_mode = embedding_mode
        self.display = display

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        BASE_MODEL_DIR = '/home/leolr-int/AGGCPerturbations/model_weights'
        self.model = Neural_Network(BASE_MODEL_DIR)
        self.model = self.model.to(self.device)

        self.use_amp = precision == 'mixed' and self.device == 'cuda'
        self.grad_scaler = GradScaler(enabled=self.use_amp)

    def training_no_OT(self, train_scanners):
        # here we train only using cross entropy


        # 1st part: training for one epoch

        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.03, momentum=0.9, weight_decay=0.001)
        torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        #data loading
        batch_size = 64 #Careful: the number of data loaded is batch_size * number of scanners 
        dataset_train = multi_WSI_loader(WSI_ids_train, train_scanners, train_or_test='Train')
        loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    
        self.model.train()

        #deactivate the encoder training if needed
        if self.freeze_encoder or self.embedding_mode: 
            self.model.encoder.eval()
        
        pbar = tqdm(loader_train, desc='Training with Cross-Entropy in progress')

        for patch, label, *_ in pbar:
            patch = patch.to(self.device)
            label = label.to(self.device)

            with torch.autocast(device_type = self.device, dtype = torch.float16, enabled = self.use_amp):
                #if embeddings from gigapath are already computed, we can speed up training
                logits = self.model.bottle_neck(patch) if self.embedding_mode else self.model(patch)
                loss = nn.CrossEntropy(logits, label)
            
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(optimizer)
            self.grad_scaler.step()
            optimizer.zero_grad()

            confidence = F.softmax(logits, dim=1)
            pred = torch.argmax(confidence, dim=1)
            
            #performance metrics
            metrics_train['running_loss'] += loss.detach().cpu().item()
            metrics_train['predictions'].extend(pred.cpu().numpy())
            metrics_train['labels'].extend(label.cpu().numpy())

            pbar.set_postfix({'step_loss': loss.detach().cpu().item()})
        
        epoch_loss_train = metrics_train['running_loss'] / len(loader_train)
        epoch_balanced_accuracy_train = balanced_accuracy_score(metrics_train['labels'], metrics_train['predictions'])
        

        # 2nd part: validation for one epoch (WHAT ABOUT torch.no_grad() ??????)
        
        self.model.eval()
        
        #we still work with the Train folder
        dataset_val = multi_WSI_loader(WSI_ids_val, train_scanners, train_or_test='Train')
        loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
        
        pbar = tqdm(loader_val, desc='Validation with Cross-Entropy in progress')
        for patch, label, *_ in pbar:
            patch = patch.to(self.device)
            label = label.to(self.device)

            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.use_amp):
                logits = self.model.bottle_neck(patch) if self.embedding_mode else self.model(patch)
                loss = nn.CrossEntropy(logits, label)
            
            confidence = F.softmax(logits, dim=1)
            pred = torch.argmax(confidence, dim=1)

            metrics_val["running_loss"] += loss.detach().cpu().item()
            metrics_val["predictions"].extend(pred.cpu().numpy())
            metrics_val["targets"].extend(label.cpu().numpy())

            pbar.set_postfix({"step_loss": loss.detach().cpu().item()})

        epoch_loss_val = metrics_val["running_loss"] / len(loader_val)
        epoch_balanced_accuracy_val = balanced_accuracy_score(metrics_val["targets"], metrics_val["predictions"])

        
        return epoch_loss_train, epoch_balanced_accuracy_train, epoch_loss_val, epoch_balanced_accuracy_val 
    

    def training_OT(self, train_scanners): 
        # we differentiate explicitly source and target scanner to apply the OT loss
        # training with validation
        
        batch_size = 64
        
        self.model.train()

        if self.freeze_encoder or self.embedding_mode:
            self.model.encoder.eval()

        target_scanner = ['Akoya'] if 'Akoya' in train_scanners else random.choice(train_scanners)
        train_scanners.remove(target_scanner[0])
        source_scanner = train_scanners

        dataset_target_train = multi_WSI_loader(WSI_ids_train, target_scanner, train_or_test='Train')
        dataset_source_train = multi_WSI_loader(WSI_ids_train, source_scanner, train_or_test='Train')

        # DataLoaders
        loader_target_train = DataLoader(dataset_target_train, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
        loader_source_train = DataLoader(dataset_source_train, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
        
        iterator_target_train = iter(loader_target_train)
        iterator_source_train = iter(loader_source_train)
        
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.03, momentum=0.9, weight_decay=0.001)
        torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        
        self.model.train()
        #deactivate the training for encoder if needed
        if self.freeze_encoder or self.embedding_mode: 
            self.model.encoder.eval()
    
        
        for i in range(len(loader_source_train)): #maybe change the loop?
            #t = epoch*len(train_loader_source) + i
            try: #potentially the scanner datasets do not have the same length
                patch_source_train, label_source_train, _ = next(iterator_source_train) #we dont really care about metadata here
            except StopIteration:
                iterator_source_train = iter(loader_source_train)
                patch_source_train, label_source_train, _ = next(iterator_source_train)
        
            try: #potentially the scanner datasets do not have the same length
                patch_target_train, label_target_train, _ = next(iterator_target_train) #we dont really care about metadata here
            except StopIteration:
                iterator_target_train = iter(loader_target_train)
                patch_target_train, label_target_train, _ = next(iterator_target_train)

            patch_source_train = patch_source_train.to(self.device)
            label_source_train = label_source_train.to(self.device)
            patch_target_train = patch_target_train.to(self.device)
            label_target_train = label_target_train.to(self.device)

            #pbar = tqdm(train_loader_target, desc='Training with OT in progress')

        
            with torch.autocast(device_type = self.device, dtype = torch.float16, enabled = self.use_amp):
                #if embeddings from gigapath are already computed, we can speed up training
                logits_source_train =  self.model.bottle_neck(patch_source_train) if self.embedding_mode else self.model(patch_source_train)
                logits_target_train = self.model.bottle_neck(patch_target_train) if self.embedding_mode else self.model(patch_target_train)
                
                # source features
                feat_source_train = nn.Sequential(*list(self.model.children())[:-1])(patch_source_train)
                # target features
                feat_target_train = nn.Sequential(*list(self.model.children())[:-1])(patch_target_train)

                #OT loss
                loss_g = loss_geom(feat_source_train.detach().squeeze, feat_target_train.detach().squeeze)
            
                # liberty taken here: instead of copy-pasting the code from https://github.com/kiakh93/OT-regularized-UDA/blob/main/train_OT.py
                # i decided to compute two cross entropies for source and target domain

                #CE loss
                loss_c = nn.CrossEntropyLoss(logits_source_train, label_source_train) + nn.CrossEntropyLoss(logits_target_train, label_target_train)
                
                # total loss
                loss_train = loss_c + Lambda*loss_g
                
            self.grad_scaler.scale(loss_train).backward()
            self.grad_scaler.step(optimizer)
            self.grad_scaler.step()
            optimizer.zero_grad()
            
            # performance
            confidence_source_train = F.softmax(logits_source_train, dim=1) #why not include that in the network directly in forward?
            confidence_target_train = F.softmax(logits_target_train, dim=1)
            pred_source_train = torch.argmax(confidence_source_train, dim=1)
            pred_target_train = torch.argmax(confidence_target_train, dim=1)


            #performance metrics
            metrics_train['running_loss'] += loss_train.detach().cpu().item()
            # we concatenate the predictions of source and target
            metrics_train['predictions'].extend(pred_source_train.cpu().numpy())
            metrics_train['predictions'].extend(pred_target_train.cpu().numpy())
            metrics_train['labels'].extend(label_source_train.cpu().numpy())
            metrics_train['labels'].extend(label_target_train.cpu().numpy())

            #pbar.set_postfix({'step_loss': loss_train.detach().cpu().item()})
        
        epoch_loss_train = metrics_train['running_loss'] / (len(loader_source_train) + len(loader_target_train))
        epoch_balanced_accuracy_train = balanced_accuracy_score(metrics_train['labels'], metrics_train['predictions'])


        # 2nd part: validation (WHAT ABOUT torch.no_grad() ??????)
        self.model.eval()
        
        target_dataset_val = multi_WSI_loader(WSI_ids_val, target_scanner, train_or_test='Train')
        source_dataset_val = multi_WSI_loader(WSI_ids_val, source_scanner, train_or_test='Train')

        # DataLoaders
        loader_target_val = DataLoader(target_dataset_val, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
        loader_source_val = DataLoader(source_dataset_val, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
        
        iterator_target_val = iter(loader_target_val)
        iterator_source_val = iter(loader_source_val)
    
        
        for i in range(len(loader_source_val)): #maybe change the loop?
            
            try: #potentially the scanner datasets do not have the same length
                patch_source_val, label_source_val, _ = next(iterator_source_val) #we dont really care about metadata here
            except StopIteration:
                iterator_source_val = iter(loader_source_val)
                patch_source_val, label_source_val, _ = next(iterator_source_val)
        
            try: #potentially the scanner datasets do not have the same length
                patch_target_val, label_target_val, _ = next(iterator_target_val) #we dont really care about metadata here
            except StopIteration:
                iterator_target_val = iter(loader_target_val)
                patch_target_val, label_target_val, _ = next(iterator_target_val)

            patch_source_val = patch_source_val.to(self.device)
            label_source_val = label_source_val.to(self.device)
            patch_target_val = patch_target_val.to(self.device)
            label_target_val = label_target_val.to(self.device)

            #pbar = tqdm(train_loader_target, desc='Training with OT in progress')

            with torch.autocast(device_type = self.device, dtype = torch.float16, enabled = self.use_amp):
                #if embeddings from gigapath are already computed, we can speed up training
                logits_source_val =  self.model.bottle_neck(patch_source_val) if self.embedding_mode else self.model(patch_source_val)
                logits_target_val = self.model.bottle_neck(patch_target_val) if self.embedding_mode else self.model(patch_target_val)
                # source features
                feat_source_val = nn.Sequential(*list(self.model.children())[:-1])(patch_source_val)
                # target features
                feat_target_val = nn.Sequential(*list(self.model.children())[:-1])(patch_target_val)

                #OT loss
                loss_g = loss_geom(feat_source_val.detach().squeeze, feat_target_val.detach().squeeze)
            
                # liberty taken here: instead of copy-pasting the code from https://github.com/kiakh93/OT-regularized-UDA/blob/main/train_OT.py
                # i decided to compute two cross entropies for source and target domain

                #CE loss
                loss_c = nn.CrossEntropyLoss(logits_source_val, label_source_val) + nn.CrossEntropyLoss(logits_target_val, label_target_val)
                
                # total loss
                loss_val = loss_c + Lambda*loss_g
                
            
            # performance
            confidence_source_val = F.softmax(logits_source_val, dim=1) #why not include that in the network directly in forward?
            confidence_target_val = F.softmax(logits_target_val, dim=1)
            pred_source_val = torch.argmax(confidence_source_val, dim=1)
            pred_target_val = torch.argmax(confidence_target_val, dim=1)


            #performance metrics
            metrics_val['running_loss'] += loss_val.detach().cpu().item()
            # we concatenate the predictions of source and target
            metrics_val['predictions'].extend(pred_source_val.cpu().numpy())
            metrics_val['predictions'].extend(pred_target_val.cpu().numpy())
            metrics_val['labels'].extend(label_source_val.cpu().numpy())
            metrics_val['labels'].extend(label_target_val.cpu().numpy())

            #pbar.set_postfix({'step_loss': loss_train.detach().cpu().item()})
        
        epoch_loss_val = metrics_val['running_loss'] / (len(loader_source_val) + len(loader_target_val))
        epoch_balanced_accuracy_val = balanced_accuracy_score(metrics_val['labels'], metrics_val['predictions'])
        
        return epoch_loss_train, epoch_balanced_accuracy_train, epoch_loss_val, epoch_balanced_accuracy_val





   




class NetworkHandler:

    

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
    
    # HERE  MUST LOAD THE TEST DATA and change the name to test instead of inference + matrice de confusion !
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
        "model": model.state_dict(),
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
