# how to manage training data?
# i have several scanners and all have labels so i can make it supervised somehow?
# faire en sorte que Akoya soit toujours la norme

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

class NetworkHandler:
    '''
    A class to handle training, inference and prediction
    '''

    def __init__(self, OT = False, precision = 'mixed', freeze_encoder = True, embedding_mode = False, display = False):
        self.OT = OT
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

        self.use_amp = precision == "mixed" and self.device == "cuda"
        self.grad_scaler = GradScaler(enabled=self.use_amp)
        self.model = self.model.to(self.device)


    def training(self, train_scanners, num_epochs):
        #pour le training faudra bien comprendre le repo !!
        # display graphs
        metrics = {'running_loss':0, 'predictions':[], 'targets':[]}

        self.model.train()

        if self.freeze_encoder or self.embedding_mode:
            self.model.encoder.eval()

        # indices of the slides for training
        WSI_ids_train = [1,2,3]
        batch_size = 128
        
        
        if self.OT: 
            #we differentiate explicitly source and target scanner to apply the OT loss
            
            # Defining OT-based loss function
            loss_geom = SamplesLoss('sinkhorn', p=2, blur=0.1, scaling=0.95, verbose=False)
            Lambda = 0.1 # strength of OT (0.1 is the value of the article)
            
            target_scanner = ['Akoya'] if 'Akoya' in train_scanners else random.choice(train_scanners)
            train_scanners.remove(target_scanner[0])
            source_scanner = train_scanners

            target_dataset = multi_WSI_loader(WSI_ids_train, target_scanner, train_or_test='Train')
            source_dataset = multi_WSI_loader(WSI_ids_train, source_scanner, train_or_test='Train')

            # DataLoaders
            train_loader_target = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
            train_loader_source = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
            
            iterator_train_loader_target = iter(train_loader_target)
            iterator_train_loader_source = iter(train_loader_source)
            
            optimizer = torch.optim.SGD(self.model.parameters(), lr=0.03, momentum=0.9, weight_decay=0.001)
            torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

            #keep tracks of performance evolution
            Loss_G = []
            Loss_CE = []
            A_t = []
            A_v = []

            for epoch in range(epoch, num_epochs):
                self.model.train()
                #deactivate the training for encoder if needed
                if self.freeze_encoder or self.embedding_mode: 
                    self.model.encoder.eval()
            
                
                for i in range(len(train_loader_source)):
                    #t = epoch*len(train_loader_source) + i
                    try: #potentially the scanner datasets do not have the same length
                        X1, Y1, _ = next(iterator_train_loader_source) #we dont really care about metadata here
                    except StopIteration:
                        iterator_train_loader_source = iter(train_loader_source)
                        X1, Y1, _ = next(iterator_train_loader_source)
                
                    try: #potentially the scanner datasets do not have the same length
                        X2, Y2, _ = next(iterator_train_loader_target) #we dont really care about metadata here
                    except StopIteration:
                        iterator_train_loader_target = iter(train_loader_target)
                        X2, Y2, _ = next(iterator_train_loader_target)

                    X1 = X1.to(self.device)
                    Y1 = Y1.to(self.device)
                    X2 = X2.to(self.device)
                    Y2 = Y2.to(self.device)

                    with torch.autocast(device_type = self.device, dtype = torch.float16, enabled = self.use_amp):
                        #if embeddings from gigapath are already computed, we can speed up training
                        logits = self.model.bottle_neck(patch) if self.embedding_mode else self.model(patch)
                        logits_source =  self.model.bottle_neck(X1) if self.embedding_mode else self.model(X1)
                        logits_target = self.model.bottle_neck(X2) if self.embedding_mode else self.model(X2)
                        # source features
                        feat1 = nn.Sequential(*list(self.model.children())[:-1])(X1)
                        # target features
                        feat2 = nn.Sequential(*list(self.model.children())[:-1])(X2)

                        #OT loss
                        loss_g = loss_geom(feat1.detach().squeeze, feat2.detach().squeeze)
                    
                        # liberty taken here: instead of copy-pasting the code from https://github.com/kiakh93/OT-regularized-UDA/blob/main/train_OT.py
                        # i decided to compute two cross entropies for source and target domain

                        #CE loss
                        loss_c = nn.CrossEntropyLoss(pred_source, Y1) + nn.CrossEntropyLoss(pred_target, Y2)
                        
                        # total loss
                        loss_train = loss_c + Lambda*loss_g
                        
                    self.grad_scaler.scale(loss_train).backward()
                    self.grad_scaler.step(optimizer)
                    self.grad_scaler.step()
                    optimizer.zero_grad()
                    
                    # performance
                    confidence_source = F.softmax(logits_source, dim=1) #why not include that in the network directly in forward?
                    confidence_target = F.softmax(logits_target, dim=1)
                    pred_source = torch.argmax(confidence_source, dim=1)
                    pred_target = torch.argmax(confidence_target, dim=1)


                   #performance metrics
                    metrics['running_loss'] += loss.detach().cpu().item()
                    metrics['predictions'].extend(pred.cpu().numpy())
                    metrics['labels'].extend(label.cpu().numpy())

                    pbar.set_postfix({'step_loss': loss.detach().cpu().item()})
                
                epoch_loss = metrics['running_loss'] / len(train_loader)
                epoch_balanced_accuracy = balanced_accuracy_score(metrics['labels'], metrics['predictions'])
                
                return epoch_loss, epoch_balanced_accuracy 




                    '''Loss_G.append(loss_g.item())
                    Loss_CE.append(loss_c.item())

                    #Accuracy of source
                    pred_y = outputs1.cpu().detach().numpy()
                    pred_y = np.argmax(pred_y, axis=1)
                    acc = 0
                    for i in range(len(pred_y)):
                        if pred_y[i] == Y1[i].data.cpu().numpy():
                            acc+=1
                    
                    output = self.model(X2)

                    # accuracy of target
                    pred_y = output.cpu().detach.numpy()
                    pred_y = np.argmax(pred_y, axis=1)
                    acc_v = 0
                    for i in range(len(pred_y)):
                        if pred_y[i] == Y2[i].data.cpu().numpy():
                            acc_v+=1
                    
                    A_t.append(acc)
                    A_v.append(acc_v)
                
                # TODO: performance per epoch'''

        else:
            # here we train only using cross entropy
            train_dataset = multi_WSI_loader(WSI_ids_train, train_scanners, train_or_test='Train')
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

            self.model.train()
            #deactivate the encoder training if needed
            if self.freeze_encoder or self.embedding_mode: 
                self.model.encoder.eval()
            
            pbar = tqdm(train_loader, desc='Training with Cross-Entropy in progress')

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
                metrics['running_loss'] += loss.detach().cpu().item()
                metrics['predictions'].extend(pred.cpu().numpy())
                metrics['labels'].extend(label.cpu().numpy())

                pbar.set_postfix({'step_loss': loss.detach().cpu().item()})
            
            epoch_loss = metrics['running_loss'] / len(train_loader)
            epoch_balanced_accuracy = balanced_accuracy_score(metrics['labels'], metrics['predictions'])
            
            return epoch_loss, epoch_balanced_accuracy 


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
