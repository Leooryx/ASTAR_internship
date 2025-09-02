# TODO: handle when different scanners than Leica are given for training while not increasing too much the size of data for one batch!
# TODO: faire une fonction display en dehors de la classe que je peux mettre facilement partout
# TODO: the code assumes im working with the patch images direclty, but sometimes patch is used for the patch embeddings!! il faut corriger ca
# TODO: concernant les batch --> mettre le nom de toutes les variables au pluriel
# TODO: data augmentation strategies?? (last)
# TODO: way too many repetitions between train and validation --> completely possible to create one function with a little if loop to compute or not gradient
# TODO: bien mettre les tqdm partout

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
import deeplake
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score
from geomloss import SamplesLoss
import time
import pandas as pd
from torch.utils.data import ConcatDataset
import pickle
import seaborn as sns
import matplot.pyplot as plt
from umap import UMAP

from architecture_OT import Network
from dataset_OT import patches_loader, make_multi_WSI_dataset

# Ensuring reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False 

# Defining OT-based loss function
loss_geom = SamplesLoss('sinkhorn', p=2, blur=0.1, scaling=0.95, verbose=False)
Lambda = 0.1 # strength of OT (0.1 is the value of the article)



    
# Defining OT-based loss function
loss_geom = SamplesLoss('sinkhorn', p=2, blur=0.1, scaling=0.95, verbose=False)
Lambda = 0.1 # strength of OT (0.1 is the value of the article)

loss_geom = SamplesLoss('sinkhorn', p=2, blur=0.1, scaling=0.95, verbose=False)
Lambda = 0.1 # strength of OT (0.1 is the value of the article)

class NetworkHandler:
    '''
    A class to handle training, inference and prediction
    '''

    def __init__(self, precision = 'mixed', freeze_encoder = True, emb_mode = False, display = False):
        self.precision = precision
        self.freeze_encoder = freeze_encoder
        self.emb_mode = emb_mode
        self.display = display

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = Network(emb_mode=self.emb_mode)
        self.model = self.model.to(self.device, non_blocking=True)

        #printing architecture
        '''print("Bottleneck layers:")
        print(self.model.bottle_neck)
        print("Head layer:")
        print(self.model.head)
        if self.model.encoder is not None:
            print("\nEncoder architecture:")
            print(self.model.encoder)'''

        self.use_amp = precision == 'mixed' and self.device == 'cuda'
        self.grad_scaler = GradScaler(enabled=self.use_amp)

    
    def training_no_OT(self, batch_size, num_epochs):
        # here we train only using cross entropy
        training_stats = []
        min_loss_val, max_accuracy_val = float("inf"), -float("inf")

        trainable_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        idx_range_subset1 = [i for i in range(1,52+1)] #only the 26 first slides
        random.shuffle(idx_range_subset1)
        num_train = int(np.ceil(0.7 * len(idx_range_subset1))) 
        train_range1, val_range1 = idx_range_subset1[:num_train], idx_range_subset1[num_train:]
        
        idx_range_subset3 = [i for i in range(1,26+1)] 
        random.shuffle(idx_range_subset3)
        num_train = int(np.ceil(0.7 * len(idx_range_subset3))) 
        train_range3, val_range3 = idx_range_subset3[:num_train], idx_range_subset3[num_train:]
    
        akoya_loader_train_subset1 = make_multi_WSI_loader('Subset1', train_range1, ['Akoya'], train_or_test='Train', batch_size=batch_size)
        akoya_loader_val_subset1 = make_multi_WSI_loader('Subset1', val_range1, ['Akoya'], train_or_test='Train', batch_size=batch_size)
        akoya_loader_train_subset3 = make_multi_WSI_loader('Subset3', train_range3, ['Akoya'], train_or_test='Train', batch_size=batch_size)
        akoya_loader_val_subset3 = make_multi_WSI_loader('Subset3', val_range3, ['Akoya'], train_or_test='Train', batch_size=batch_size)
        leica_loader_train = make_multi_WSI_loader('Subset3', train_range3, ['Leica'], train_or_test='Train', batch_size=batch_size)
        leica_loader_val = make_multi_WSI_loader('Subset3', val_range3, ['Leica'], train_or_test='Train', batch_size=batch_size)
    
        loader_train = ConcatDataset([akoya_loader_train_subset1.dataset, akoya_loader_train_subset3.dataset, leica_loader_train.dataset])
        loader_val = ConcatDataset([akoya_loader_val_subset1.dataset, akoya_loader_val_subset3.dataset, leica_loader_val.dataset])
    
        loader_train = DataLoader(loader_train, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True, persistent_workers=True)
        loader_val = DataLoader(loader_val, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True, persistent_workers=True)
            
        len_loader_train = len(loader_train)
        len_loader_val = len(loader_val)

        for epoch in range(1, num_epochs+1):
            

            metrics_train = {'running_loss': 0, 'predictions': [], 'labels': []}
            metrics_val   = {'running_loss': 0, 'predictions': [], 'labels': []}
            
            start = time.time()
            # 1st part: training for one epoch 
            self.model.train()
    
            #deactivate the encoder training if needed
            if self.freeze_encoder and not self.emb_mode: 
                self.model.encoder.eval()
            
            pbar = tqdm(loader_train, desc=f'Epoch:{epoch} Training with Cross-Entropy in progress')
            
            for batch in pbar:
                batch_train_start = time.time()
                
                patch = batch['embedding'] if self.emb_mode else batch['img'] 
                patch = patch.to(self.device, non_blocking=True)
                label = batch['label'].to(self.device, non_blocking=True)

                optimizer.zero_grad()
    
                with torch.autocast(device_type = self.device, dtype = torch.float16, enabled = self.use_amp):
                    #if embeddings from gigapath are already computed, we can speed up training
                    logits = self.model(patch) 
                    loss = nn.CrossEntropyLoss()(logits, label) #careful about syntax
                
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.step(optimizer)
                self.grad_scaler.update()
                
    
                confidence = F.softmax(logits, dim=1)
                pred = torch.argmax(confidence, dim=1)
                
                #performance metrics
                metrics_train['running_loss'] += loss.detach().item()
                metrics_train['predictions'].extend(pred.cpu().numpy())
                metrics_train['labels'].extend(label.cpu().numpy())
                
                
    
                #pbar.set_postfix({'step_loss': loss.detach().item()})
                
            
            epoch_loss_train = metrics_train['running_loss'] / len_loader_train
            epoch_balanced_accuracy_train = balanced_accuracy_score(metrics_train['labels'], metrics_train['predictions'])
            

            
            
            # 2nd part: validation for one epoch 
            with torch.no_grad():
                self.model.eval()
                
                #we still work with the Train folder
                
                pbar = tqdm(loader_val, desc=f'Epoch:{epoch} Validation - Cross-Entropy in progress')
                for batch in pbar:
                    patch = batch['embedding'] if self.emb_mode else batch['img']
                    patch = patch.to(self.device, non_blocking=True) 
                    label = batch['label'].to(self.device, non_blocking=True)

                    with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.use_amp):
                        logits = self.model(patch)
                        loss = nn.CrossEntropyLoss()(logits, label)
                    
                    confidence = F.softmax(logits, dim=1)
                    pred = torch.argmax(confidence, dim=1)
        
                    metrics_val["running_loss"] += loss.detach().item()
                    metrics_val["predictions"].extend(pred.cpu().numpy())
                    metrics_val["labels"].extend(label.cpu().numpy())
        
                    #pbar.set_postfix({"step_loss": loss.detach().item()})
        
                epoch_loss_val = metrics_val["running_loss"] / len_loader_val
                epoch_balanced_accuracy_val = balanced_accuracy_score(metrics_val["labels"], metrics_val["predictions"])

                scheduler.step(epoch_loss_val) 
                
                cm = confusion_matrix(metrics_val["labels"], metrics_val["predictions"], labels=[0, 1, 2, 3, 4], normalize='true')
    
                end = time.time()
    
                dic = {'epoch_loss_train': epoch_loss_train, 
                    'epoch_balanced_accuracy_train': epoch_balanced_accuracy_train, 
                    'epoch_loss_val': epoch_loss_val, 
                    'epoch_balanced_accuracy_val': epoch_balanced_accuracy_val, 
                    'time': end - start,
                    'cm':cm}
                
                training_stats.append(dic)

                min_loss_val, max_accuracy_val = end_epoch(
                                                    save_dir,
                                                    custom_name,
                                                    self.model,
                                                    optimizer,
                                                    scheduler,
                                                    epoch,
                                                    epoch_loss_train,
                                                    epoch_balanced_accuracy_train,
                                                    epoch_loss_val,
                                                    epoch_balanced_accuracy_val,
                                                    min_loss_val,
                                                    max_accuracy_val)
                
                train_plot(pd.DataFrame(training_stats), cm, custom_name=custom_name)
                torch.cuda.empty_cache()

                
            
        return epoch_loss_val, epoch_balanced_accuracy_val

    @torch.no_grad()
    def extract_embeddings(self, subset, scanners, WSI_ids, train_or_test, batch_size):
        # creates the deeplake database for embeddings
        # structure: one deeplake dataset per WSI, embeddings, scanner, WSI_id, Train or Test

        root_dir = '/home/leolr-int/nfs/transformed_data/my_embeddings'
        self.model.eval()
        if subset == 'Subset1':
            scanners = 'Akoya'
        for scanner in scanners:
            for id in WSI_ids:
                if subset == 'Subset1':
                    path = f'{subset}_{train_or_test}_{id}'
                else:
                    path = f'{subset}_{train_or_test}_{id}_{scanner}'
                final_destination = os.path.join(root_dir, path)
                os.makedirs(final_destination, exist_ok=True)
                # creation of the deeplake dataset
                embedding_ds = deeplake.create(final_destination)
                embedding_ds.add_column('embedding', dtype=deeplake.types.Embedding(1536)) 
                embedding_ds.add_column('subset', dtype=deeplake.types.Text)
                embedding_ds.add_column('scanner', dtype=deeplake.types.Text)
                embedding_ds.add_column('WSI_id', deeplake.types.Int32)
                embedding_ds.add_column('train_or_test', dtype=deeplake.types.Text)
                embedding_ds.add_column('label', dtype=deeplake.types.Int32)

                WSI = patches_loader(subset, train_or_test, id, scanner, to_torch=True, emb_mode=False)
                loader = DataLoader(WSI, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True)
                batch_records = []

                for batch in tqdm(loader, desc=f'Extracting {path}'):
                    patches = batch['img'].to(self.device).float()
                    labels = batch['label']

                    with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.use_amp):
                        embeddings = self.model.encoder(patches)
                        embeddings = embeddings.detach().cpu().numpy()

                        # accumulate all rows from batch
                        for emb, label in zip(embeddings, labels):
                            batch_records.append({
                                'embedding':emb,
                                'subset': subset,
                                'WSI_id': id, 
                                'scanner': 'Akoya' if subset == 'Subset1' else scanner,
                                'train_or_test': train_or_test,
                                'label': label
                            })

                        # append in large
                        if len(batch_records) >= 1000:
                            embedding_ds.append(batch_records)
                            batch_records.clear()

                    # append what is left
                    if batch_records:
                        embedding_ds.append(batch_records)
                    
                
    @torch.no_grad()
    def inference(self, custom_name, scanner, data_loader, visual=True):
        weights = f'/home/leolr-int/nfs/transformed_data/weights/{custom_name}/checkpoint.pth'
        checkpoint = torch.load(weights, weights_only=False, map_location=self.device)
        handler.model.load_state_dict(checkpoint["model"])        
        
        self.model.eval()
        all_preds, all_labels, all_embeddings = [], [], []
        for batch in tqdm(data_loader, desc='Inference in progress...'):
            vectors= batch['embedding'].to(self.device)
            labels = batch['label'].to(self.device)
            emb = self.model.bottle_neck(vectors)
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.use_amp):
                logits = self.model.head(emb)
                
            confidence = F.softmax(logits, dim=1)
            pred = torch.argmax(confidence, dim=1)
            
            all_preds.append(pred.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_embeddings.append(emb.cpu().numpy())
        
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate (all_labels)
        all_embeddings = np.concatenate(all_embeddings)

        acc_score = balanced_accuracy_score(all_labels, all_preds)
        print(acc_score)

        CM = better_confusion_matrix(custom_name, all_labels, all_preds, scanner, acc_score)
        if visual:
            #random selection of 20% of the sample
            N = all_embeddings.shape[0]
            sample_size = int(0.2 * N)
            indices = torch.randperm(N)[:sample_size]
            sampled_embeddings = all_embeddings[indices]
            sampled_labels = all_labels[indices]
            dim_reduc_plot(embeddings=sampled_embeddings, y_true=sampled_labels, scanner=scanner, custom_name=custom_name, n_components=2)


    def training_OT(self, batch_size, num_epochs): 
        # we differentiate explicitly source and target scanner to apply the OT loss
        # training with validation

        training_stats = []
        min_loss_val, max_accuracy_val = float("inf"), -float('inf')
        
        min_len_train = min(len(akoya_loader_train), len(leica_loader_train))
        min_len_val = min(len(akoya_loader_val), len(leica_loader_val))
        
        trainable_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        optimizer = torch.optim.AdamW(trainable_params, lr=10e-4, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        ce_loss = nn.CrossEntropyLoss()
        
        #Data loading (parallelise Akoya and Leica data loaders with resampling for Leica)
        # if this works, delete "optimised loader" (useless)
        
        

        for epoch in range(1,num_epochs+1):
            metrics_train = {'running_loss': 0, 'predictions': [], 'labels': []}
            metrics_val   = {'running_loss': 0, 
                             'predictions': [], 
                             'predictions_akoya':[],
                             'predictions_leica':[],
                             'labels_akoya':[],
                             'labels_leica':[],
                             'labels': [], 
                             'CE_target_val':0,
                            'CE_source_val':0,
                            'OT_loss_val':0,
                            'acc_target_val':0,
                            'acc_source_val':0}
            
            
            start = time.time()
            
            #1st part: train for one epoch
            self.model.train()

            #deactivate the encoder training if needed
            if self.freeze_encoder and not self.emb_mode:
                self.model.encoder.eval()

            #the target scanner is Akoya
            
            for batch_akoya, batch_leica in tqdm(zip(akoya_loader_train, leica_loader_train),
                                                desc=f"Epoch {epoch} Training - OT in progress",
                                                total=min_len_train):
            #for batch_akoya in tqdm(akoya_loader_train, desc=f'Epoch {epoch} Training - Optimal Transport in progress'):
                #batch_leica = next(leica_train_iter)

                patches_akoya = (batch_akoya['embedding'] if self.emb_mode else batch_akoya['img']).to(self.device, non_blocking=True)
                patches_leica = (batch_leica['embedding'] if self.emb_mode else batch_leica['img']).to(self.device, non_blocking=True)
                labels_akoya = batch_akoya['label'].to(self.device, non_blocking=True)
                labels_leica = batch_leica['label'].to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
    
                with torch.autocast(device_type = self.device, dtype = torch.float16, enabled = self.use_amp):
                    #if embeddings from gigapath are already computed, we can speed up training
                    #the name embedding is ambiguous, in "batch_akoya['embedding']" it refers to Gigapath, in the line below it refers to my model with bottleneck
                    embedding_akoya = self.model.bottle_neck(patches_akoya) #.to(self.device, non_blocking=True)
                    embedding_leica = self.model.bottle_neck(patches_leica) #.to(self.device, non_blocking=True)
                    logits_akoya = self.model(patches_akoya) #.to(self.device, non_blocking=True) 
                    logits_leica = self.model(patches_leica) #.to(self.device, non_blocking=True)

                    OT_loss_train = supervised_OT_loss(embedding_akoya, embedding_leica, labels_akoya, labels_leica)
                    # loss function
                    loss_train = (ce_loss(logits_akoya, labels_akoya) 
                                  +  ce_loss(logits_leica, labels_leica) #to avoid that the model specialises in Akoya
                                  + 0.01 * OT_loss_train 
                                 )
                    
                self.grad_scaler.scale(loss_train).backward()
                self.grad_scaler.step(optimizer)
                self.grad_scaler.update()
                
    
                confidence_akoya = F.softmax(logits_akoya, dim=1)
                confidence_leica = F.softmax(logits_leica, dim=1)
                pred_akoya = torch.argmax(confidence_akoya, dim=1)
                pred_leica = torch.argmax(confidence_leica, dim=1)
                
                #performance metrics
                #performance metrics
                metrics_train['running_loss'] += loss_train.detach().cpu().item()
                # we concatenate the predictions of source and target
                metrics_train['predictions'].extend(pred_akoya.detach().cpu().numpy())
                metrics_train['predictions'].extend(pred_leica.detach().cpu().numpy())
                metrics_train['labels'].extend(labels_akoya.detach().cpu().numpy())
                metrics_train['labels'].extend(labels_leica.detach().cpu().numpy())

                
            
            epoch_loss_train = metrics_train['running_loss'] / min_len_train 
            epoch_balanced_accuracy_train = balanced_accuracy_score(metrics_train['labels'], metrics_train['predictions'])               
                
          
            # 2nd part: validation for one epoch 
            with torch.no_grad():
                self.model.eval()
                
                #we still work with the Train folder
                
                for batch_akoya_val, batch_leica_val in tqdm(zip(akoya_loader_val, leica_loader_val),
                                                desc=f"Epoch {epoch} Validation - OT in progress",
                                                total=min_len_val):

                    patches_akoya_val = (batch_akoya_val['embedding'] if self.emb_mode else batch_akoya_val['img']).to(self.device, non_blocking=True)
                    patches_leica_val = (batch_leica_val['embedding'] if self.emb_mode else batch_leica_val['img']).to(self.device, non_blocking=True)
                    labels_akoya_val = batch_akoya_val['label'].to(self.device, non_blocking=True)
                    labels_leica_val = batch_leica_val['label'].to(self.device, non_blocking=True)

                    with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.use_amp):
                        embedding_akoya_val = self.model.bottle_neck(patches_akoya_val) #.to(self.device, non_blocking=True)
                        embedding_leica_val = self.model.bottle_neck(patches_leica_val) #.to(self.device, non_blocking=True)
                        logits_akoya_val = self.model(patches_akoya_val) #.to(self.device, non_blocking=True)
                        logits_leica_val = self.model(patches_leica_val) #.to(self.device, non_blocking=True)
                        
                        CE_target_val = ce_loss(logits_akoya_val, labels_akoya_val)
                        CE_source_val = ce_loss(logits_leica_val, labels_leica_val)
                        OT_loss_val = supervised_OT_loss(embedding_akoya_val, embedding_leica_val, labels_akoya_val, labels_leica_val)
                        loss_val = (
                            CE_target_val #to avoid that the model specialises in Akoya
                            + CE_source_val
                            + 0.01 * OT_loss_val
                        )
                        #here we detach both because we do not compute the gradient
                    
                    confidence_akoya_val = F.softmax(logits_akoya_val, dim=1)
                    confidence_leica_val = F.softmax(logits_leica_val, dim=1)
                    pred_akoya_val = torch.argmax(confidence_akoya_val, dim=1)
                    pred_leica_val = torch.argmax(confidence_leica_val, dim=1)

                    # Update validation metrics
                    metrics_val['running_loss'] += loss_val.detach().cpu().item()
                    metrics_val['predictions_akoya'].extend(pred_akoya_val.detach().cpu().numpy())
                    metrics_val['predictions'].extend(pred_akoya_val.detach().cpu().numpy())
                    metrics_val['predictions_leica'].extend(pred_leica_val.detach().cpu().numpy())
                    metrics_val['predictions'].extend(pred_leica_val.detach().cpu().numpy())
                    metrics_val['labels_akoya'].extend(labels_akoya_val.detach().cpu().numpy())
                    metrics_val['labels'].extend(labels_akoya_val.detach().cpu().numpy())
                    metrics_val['labels_leica'].extend(labels_leica_val.detach().cpu().numpy())
                    metrics_val['labels'].extend(labels_leica_val.detach().cpu().numpy())
                    metrics_val['CE_target_val'] += CE_target_val.detach().cpu().item()
                    metrics_val['CE_source_val'] += CE_source_val.detach().cpu().item()
                    metrics_val['OT_loss_val'] += OT_loss_val.detach().cpu().item()
        
        
                epoch_loss_val = metrics_val['running_loss'] / min_len_val
                
                epoch_balanced_accuracy_val = balanced_accuracy_score(metrics_val['labels'], metrics_val['predictions'])
                
                scheduler.step(epoch_loss_val) 
                
                cm = confusion_matrix(metrics_val["labels"], metrics_val["predictions"], labels=[0, 1, 2, 3, 4], normalize='true')
    
                end = time.time()
    
                dic = {'epoch_loss_train': epoch_loss_train, 
                    'epoch_balanced_accuracy_train': epoch_balanced_accuracy_train, 
                    'epoch_loss_val': epoch_loss_val, 
                    'epoch_balanced_accuracy_val': epoch_balanced_accuracy_val, 
                    'time': end - start,
                    'cm':cm,
                    'CE_target_val': metrics_val['CE_target_val'] / min_len_val,
                    'OT_loss_val': metrics_val['OT_loss_val'] / min_len_val,
                    'CE_source_val': metrics_val['CE_source_val'] / min_len_val,
                    'acc_target': balanced_accuracy_score(metrics_val['labels_akoya'], metrics_val['predictions_akoya']),
                    'acc_source': balanced_accuracy_score(metrics_val['labels_leica'], metrics_val['predictions_leica'])
                      }
                
                training_stats.append(dic)

                min_loss_val, max_accuracy_val = end_epoch(
                                                    save_dir,
                                                    custom_name,
                                                    self.model,
                                                    optimizer,
                                                    scheduler,
                                                    epoch,
                                                    epoch_loss_train,
                                                    epoch_balanced_accuracy_train,
                                                    epoch_loss_val,
                                                    epoch_balanced_accuracy_val,
                                                    min_loss_val,
                                                    max_accuracy_val)
                
                train_plot(pd.DataFrame(training_stats), cm, custom_name=custom_name, OT=True)
                torch.cuda.empty_cache()

                
            
        return epoch_loss_val, epoch_balanced_accuracy_val

    


def train_plot(training_stats, cm, custom_name, plot=False, OT=False):
    if OT:
        row = 3
    else:
        row = 2
    fig, axes = plt.subplots(row,2, figsize=(10,10))
    # loss curves
    axes[0,0].plot(training_stats['epoch_loss_train'], label='Training loss', color='blue')
    axes[0,0].plot(training_stats['epoch_loss_val'], label='Validation loss', color='green')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend()
    axes[0,0].set_title('Loss per epoch')

    # accuracy curves
    axes[0,1].plot(training_stats['epoch_balanced_accuracy_train'], label='Training accuracy', color='blue')
    axes[0,1].plot(training_stats['epoch_balanced_accuracy_val'], label='Validation accuracy', color='green')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Accuracy')
    axes[0,1].legend()
    axes[0,1].set_title('Accuracy per epoch')

    # time per epoch
    axes[1,0].plot(training_stats['time'], color='black')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('Time')
    axes[1,0].set_title('Time for each epoch')

    # confusion matrix
    label_name = ['Stroma', 'Normal', 'G3', 'G4', 'G5']
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_name)
    display.plot(ax=axes[1,1], cmap='PuRd', colorbar=False)
    axes[1,1].set_title('Confusion matrix for validation')

    if OT:
        axes[2,0].plot(training_stats['CE_target_val'], label='CE Akoya', color='blue')
        axes[2,0].plot(training_stats['CE_source_val'], label='CE Leica', color='green')
        axes[2,0].plot(training_stats['OT_loss_val'], label='OT', color='red')
        axes[2,0].set_xlabel('Epoch')
        axes[2,0].set_ylabel('Losses')
        axes[2,0].legend()
        axes[2,0].set_title('Decomposed validation loss')

        axes[2,1].plot(training_stats['acc_target'], label='Akoya accuracy', color='blue')
        axes[2,1].plot(training_stats['acc_source'], label='Leica accuracy', color='green')
        axes[2,1].set_xlabel('Epoch')
        axes[2,1].set_ylabel('Accuracy')
        axes[2,1].legend()
        axes[2,1].set_title('Validation accuracies of target and source per epoch')

    plt.suptitle('Training statistics')
    fig.tight_layout()
    if plot:
        plt.show()
    plt.savefig(f'training_stats_{custom_name}.pdf')
    plt.close(fig)

def save_checkpoint(
    save_dir,
    custom_name,
    model,
    optimizer,
    scheduler,
    epoch,
    balanced_accuracy,
    loss,
    min_loss_val,
    max_accuracy_val):

    os.makedirs(f'{save_dir}/{custom_name}', exist_ok=True)

    training_state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'balanced_accuracy': balanced_accuracy,
        'loss': loss,
        'min_loss_val': min_loss_val,
        'max_accuracy_val': max_accuracy_val
    }

    torch.save(training_state, os.path.join(save_dir, custom_name, f'checkpoint.pth'))

# i have to give a custom_name

def end_epoch(
    save_dir,
    custom_name,
    model,
    optimizer,
    scheduler,
    epoch,
    epoch_loss_train,
    epoch_balanced_accuracy_train,
    epoch_loss_val,
    epoch_balanced_accuracy_val,
    min_loss_val,
    max_accuracy_val,
):
    save_checkpoint(
        save_dir=save_dir,
        custom_name=custom_name,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        balanced_accuracy=epoch_balanced_accuracy_val,
        loss=epoch_loss_val,
        min_loss_val =min_loss_val,
        max_accuracy_val=max_accuracy_val,
    )

    if epoch_loss_val < min_loss_val:
        torch.save(model.state_dict(), os.path.join(save_dir, custom_name, f'lowest_loss.pth'))
        min_loss_val = epoch_loss_val
        print(f'Epoch {epoch}: new minimum for val loss = {min_loss_val}')
    if epoch_balanced_accuracy_val > max_accuracy_val:
        torch.save(model.state_dict(), os.path.join(save_dir, custom_name, f'max_accuracy.pth'))
        max_accuracy_val = epoch_balanced_accuracy_val
        print(f'Epoch {epoch}: new maximum for val accuracy = {max_accuracy_val}')
                   

    return min_loss_val, max_accuracy_val







def better_confusion_matrix(custom_name, y_true, y_pred, scanner, acc_score):
    label_name = ['Stroma', 'Normal', 'G3', 'G4', 'G5']
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3,4], normalize='true')
    #?
    cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    #?
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    precisions = precision_score(y_true, y_pred, average=None, zero_division=0)

    df = pd.DataFrame(cm, index=label_name, columns=label_name)
    df['recall'] = recalls
    df_perc = pd.DataFrame(cm_perc, index=label_name, columns=label_name)
    df_perc['precision'] = precisions

    precisions = list(precisions) + [np.nan] #this is a filler for the bottom left corner
    df.loc['precision'] = precisions
    df_perc.loc['precision'] = precisions


    fig, ax = plt.subplots(figsize=(6,6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_name)
    disp.plot(ax=ax, cmap='PuRd')
    #display percentage
    '''for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            disp.text_[i,j].set_text(cm_perc[i,j])'''
    
    plt.title(f'Confusion matrix using {custom_name} for {scanner} \n balanced accuracy = {round(acc_score*100, 2)}%')
    plt.plot()
    plt.savefig(f'/home/leolr-int/nfs/transformed_data/weights/{custom_name}/confusion_matrix_{scanner}.pdf')
    
    return df, df_perc

def dim_reduc_plot(embeddings, y_true, scanner, custom_name, n_components=2):
    
    umap = UMAP(n_components=2, init='random', random_state=42, verbose=True)
    with open(f'/home/leolr-int/nfs/transformed_data/weights/{custom_name}/umap_{scanner}_axis.pkl', 'wb') as f:
        pickle.dump(umap, f)
    reduced = umap.fit_transform(embeddings)
    
    label_map = {0:'Stroma', 1:'Normal', 2:'G3', 3:'G4', 4:'G5'}
    labels = [label_map[y] for y in y_true]    
    
    plt.figure(figsize=(6,6))
    palette = sns.color_palette("colorblind")
    ax = sns.scatterplot(
        x=reduced[:,0], 
        y=reduced[:,1], 
        hue=labels, 
        palette=palette, 
        alpha=0.6, 
        s=20
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, title='Labels')
    
    plt.legend(title="Label", labels=['Normal', 'Stroma', 'G3', 'G4', 'G5'], loc="upper left")
    plt.title(f'UMAP plot for {scanner}')
    plt.savefig(f'/home/leolr-int/nfs/transformed_data/weights/{custom_name}/umap_{scanner}.png')
    plt.plot()

    #usage example:
    '''with open('umap_{scanner}_axis.pkl', 'rb') as f:
        loaded_umap = pickle.load(f)
    new_reduced = loaded_umap.transform(new_embeddings)'''


def supervised_OT_loss(embedding_akoya, embedding_leica, labels_akoya, labels_leica):
    OT_loss = embedding_akoya.new_zeros(()) #creates a new tensor with the same dtype and device, with () --> scalar shape
    #OT_loss = torch.tensor(0.0).to('cuda')
    device = embedding_akoya.device
    
    uniq_akoya = torch.unique(labels_akoya)
    uniq_leica = torch.unique(labels_leica)
    common_labels = uniq_akoya[torch.isin(uniq_akoya, uniq_leica)]

    
    for label in common_labels:
        akoya_idx = labels_akoya == label
        leica_idx = labels_leica == label
        akoya_emb = embedding_akoya[akoya_idx]
        leica_emb = embedding_leica[leica_idx]
        
        label_OT = loss_geom(akoya_emb, leica_emb)
        OT_loss.add_(label_OT) #in place operation
        
    return OT_loss   

   

    






# Example:
'''
#for training
loader_train, loader_val = create_optimized_loaders()
print(f"Training batches per epoch: {len(loader_train)}") #34k
print(f"Validation batches per epoch: {len(loader_val)}") #~15k
scanners_train = ['Akoya', 'Leica'] #add Leica later
train_or_test = 'Train'
WSI_ids_train = [i for i in range(1,26+1)] #for testing
WSI_ids_val = [i for i in range(1,26+1)] #i have to be sure that all labels are represented in validation data
batch_size = 64
#subset = 'Subset1'
training_stats = []
handler = NetworkHandler(emb_mode=True)
save_dir = '/home/leolr-int/nfs/transformed_data/weights'
custom_name = 'baseline_train_val_sep'

num_epochs = 50
handler.training_no_OT(loader_train, loader_val, num_epochs, custom_name)


'''

'''scanners = ['Akoya', 'Leica', 'KFBio']
train_or_test = 'Train'
WSI_ids = [i for i in range(1,26+1)]
batch_size = 128
NetworkHandler().extract_embeddings(scanners, WSI_ids, train_or_test, batch_size)'''

'''
#for inference:
inference = True

if inference:
    custom_name = 'optimal_transport_loss_v2'
    batch_size = 64
    id_ranges = [i for i in range(1, 26+1)]
    
    scanner = 'KFBio'
    loader = make_multi_WSI_dataset('Subset3', WSI_ids=id_ranges, scanners=[scanner], train_or_test='Train', batch_size=batch_size)
    loader = DataLoader(loader, batch_size=batch_size, num_workers=6, pin_memory=True, persistent_workers=True)
    handler = NetworkHandler(emb_mode=True)
    handler.inference(custom_name=custom_name, scanner=scanner, data_loader=loader, visual=True)
    
    scanner = 'KFBio_FDA'
    loader = make_multi_WSI_dataset('Subset3', WSI_ids=id_ranges, scanners=[scanner], train_or_test='Train', batch_size=batch_size)
    loader = DataLoader(loader, batch_size=batch_size, num_workers=6, pin_memory=True, persistent_workers=True)
    handler = NetworkHandler(emb_mode=True)
    handler.inference(custom_name=custom_name, scanner=scanner, data_loader=loader, visual=True)
'''



# new embeddings extraction: one deeplake dataset + config_extraction
@torch.no_grad()
def extract_embeddings(self, extraction_config, batch_size):
    
    root_dir = '/home/leolr-int/nfs/transformed_data/all_embeddings'
    os.makedirs(root_dir, exist_ok=True)
    embedding_ds = deeplake.create(root_dir)
    embedding_ds.add_column('embedding', dtype=deeplake.types.Embedding(1536)) 
    embedding_ds.add_column('subset', dtype=deeplake.types.Text)
    embedding_ds.add_column('scanner', dtype=deeplake.types.Text)
    embedding_ds.add_column('WSI_id', deeplake.types.Int32)
    embedding_ds.add_column('train_or_test', dtype=deeplake.types.Text)
    embedding_ds.add_column('label', dtype=deeplake.types.Int32)
    
    self.model.eval()
    
    for config in extraction_config:
        subset = config['subset']
        scanner = config['scanner'] 
        WSI_ids = config['WSI_ids']
        train_or_test = config['train_or_test']
        
        for wsi_id in WSI_ids:

            WSI = patches_loader(subset, train_or_test, wsi_id, scanner, to_torch=True, emb_mode=False)
            loader = DataLoader(WSI, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True)
            batch_records = []

            for batch in tqdm(loader, desc=f'{subset}_{wsi_id}_{scanner[0]}'):
                patches = batch['img'].to(self.device).float()
                labels = batch['label']

                with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.use_amp):
                    embeddings = self.model.encoder(patches)
                    embeddings = embeddings.detach().cpu().numpy()

                    # accumulate all rows from batch
                    for emb, label in zip(embeddings, labels):
                        batch_records.append({
                            'embedding':emb,
                            'subset': subset,
                            'WSI_id': wsi_id, 
                            'scanner': 'Akoya' if subset == 'Subset1' else scanner,
                            'train_or_test': train_or_test,
                            'label': label})

                    # append in large
                    if len(batch_records) >= 1000:
                        embedding_ds.append(batch_records)
                        batch_records.clear()

                # append what is left
                if batch_records:
                    embedding_ds.append(batch_records)
    print('finish')



# Example of data loading 
batch_size = 512 

print('Loading starting...')
idx_range_subset1 = [i for i in range(1,52+1)]
random.shuffle(idx_range_subset1)
num_train = int(np.ceil(0.7 * len(idx_range_subset1))) 
train_range1, val_range1 = idx_range_subset1[:num_train], idx_range_subset1[num_train:]

idx_range_subset3 = [i for i in range(1,26+1)] 
random.shuffle(idx_range_subset3)
num_train = int(np.ceil(0.7 * len(idx_range_subset3))) 
train_range3, val_range3 = idx_range_subset3[:num_train], idx_range_subset3[num_train:]

akoya_loader_train_subset1 = make_multi_WSI_dataset('Subset1', train_range1, ['Akoya'], train_or_test='Train', batch_size=batch_size)
akoya_loader_val_subset1 = make_multi_WSI_dataset('Subset1', val_range1, ['Akoya'], train_or_test='Train', batch_size=batch_size)
akoya_loader_train_subset3 = make_multi_WSI_dataset('Subset3', train_range3, ['Akoya'], train_or_test='Train', batch_size=batch_size)
akoya_loader_val_subset3 = make_multi_WSI_dataset('Subset3', val_range3, ['Akoya'], train_or_test='Train', batch_size=batch_size)
leica_loader_train = make_multi_WSI_dataset('Subset3', train_range3, ['Leica'], train_or_test='Train', batch_size=batch_size)
leica_loader_val = make_multi_WSI_dataset('Subset3', val_range3, ['Leica'], train_or_test='Train', batch_size=batch_size)

akoya_loader_train = ConcatDataset([akoya_loader_train_subset1, akoya_loader_train_subset3])
akoya_loader_val = ConcatDataset([akoya_loader_val_subset1, akoya_loader_val_subset3])

len_akoya_train = len(akoya_loader_train)
len_akoya_val = len(akoya_loader_val)

len_leica_train = len(leica_loader_train)
len_leica_val = len(leica_loader_val)

len_train = len_akoya_train + len_leica_train
len_val = len_akoya_val + len_leica_val

B_A_train = round(batch_size * len_akoya_train / (len_akoya_train + len_leica_train))
B_L_train = batch_size - B_A_train 
B_A_val = round(batch_size * len_akoya_val / (len_akoya_val + len_leica_val))
B_L_val = batch_size - B_A_val

akoya_loader_train = DataLoader(akoya_loader_train, batch_size=B_A_train, shuffle=True, num_workers=6, pin_memory=True, persistent_workers=True, prefetch_factor=4)
akoya_loader_val = DataLoader(akoya_loader_val, batch_size=B_A_val, num_workers=6, pin_memory=True, persistent_workers=True, prefetch_factor=4)
leica_loader_train = DataLoader(leica_loader_train, batch_size=B_L_train, shuffle=True, num_workers=6, pin_memory=True, persistent_workers=True, prefetch_factor=4)
leica_loader_val = DataLoader(leica_loader_val, batch_size=B_L_val, num_workers=6, pin_memory=True, persistent_workers=True, prefetch_factor=4)

#leica_train_iter = itertools.cycle(iter(leica_loader_train))
#leica_val_iter = itertools.cycle(iter(leica_loader_val)) #????

print("Train batches Akoya:", len(akoya_loader_train), 'batch size:', B_A_train)
print("Train batches Leica:", len(leica_loader_train), 'batch size:', B_L_train)
print("Validation batches Akoya:", len(akoya_loader_val), 'batch size:', B_A_val)
print("Validation batches Leica:", len(leica_loader_val), 'batch size:', B_L_val)

#in samples
print('len train:', len_train)





































