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
from sklearn.decomposition import PCA
from geomloss import SamplesLoss
import time
import pandas as pd
from torch.utils.data import ConcatDataset, RandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
import pickle
import seaborn as sns
import itertools
import os
from typing import (
    Tuple, 
    Literal
)

from dataset_OT import make_multi_WSI_dataset
from architecture_OT import Network
from handler_OT import train_plot, end_epoch, better_confusion_matrix, dim_reduc_plot

seed = 42
torch.manual_seed(seed)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False 


batch_size = 512 

ids_subset1 = [i for i in range(1,52+1)]
random.shuffle(ids_subset1)
num_train = int(np.ceil(0.7 * len(ids_subset1))) 
train_range1, val_range1 = ids_subset1[:num_train], ids_subset1[num_train:]

ids_subset3 = [i for i in range(1,26+1)]
random.shuffle(ids_subset3)
num_train = int(np.ceil(0.7 * len(ids_subset3))) 
train_range3, val_range3 = ids_subset3[:num_train], ids_subset3[num_train:]


def data_train_val(subset, ids, scanner):
    if subset == 'Subset1':
        train_range, val_range = train_range1, val_range1
    else:
        train_range = [i for i in train_range3 if i in ids]
        val_range = [i for i in val_range3 if i in ids]

    #train_range = [1]
    #val_range = [5]
    
    train_dataset = make_multi_WSI_dataset(subset, train_range, [scanner], train_or_test='Train', batch_size=batch_size)
    val_dataset = make_multi_WSI_dataset(subset, val_range, [scanner], train_or_test='Train', batch_size=batch_size)
    return train_dataset, val_dataset

akoya_data_train_subset1, akoya_data_val_subset1 = data_train_val('Subset1', [i for i in range(1,52+1)], 'Akoya')
akoya_data_train_subset3, akoya_data_val_subset3 = data_train_val('Subset3', [i for i in range(1,26+1)], 'Akoya')
leica_data_train, leica_data_val = data_train_val('Subset3', [i for i in range(1,26+1)], 'Leica')
leica_data_train, leica_data_val = data_train_val('Subset3', [i for i in range(1,26+1)], 'Leica')
philips_data_train, philips_data_val = data_train_val('Subset3', [i for i in range(1,26+1)], 'Philips')
olympus_data_train, olympus_data_val = data_train_val('Subset3', [i for i in range(1,26+1) if i != 20], 'Olympus')
zeiss_data_train, zeiss_data_val = data_train_val('Subset3', [1,5,6,7,8,9,10,11,12,13,14,16,21,23,25], 'Zeiss')

akoya_data_train = ConcatDataset([akoya_data_train_subset1, akoya_data_train_subset3])
akoya_data_val = ConcatDataset([akoya_data_val_subset1, akoya_data_val_subset3])

len_akoya_train = len(akoya_data_train)
len_akoya_val = len(akoya_data_val)

len_leica_train = len(leica_data_train)
len_leica_val = len(leica_data_val)

len_philips_train = len(philips_data_train) 
len_philips_val = len(philips_data_val)

len_olympus_train = len(olympus_data_train)
len_olympus_val = len(olympus_data_val)

len_zeiss_train = len(zeiss_data_train)
len_zeiss_val = len(zeiss_data_val)

len_train = len_akoya_train + len_leica_train + len_philips_train + len_olympus_train + len_zeiss_train
len_val = len_akoya_val + len_leica_val + len_philips_val + len_olympus_val + len_zeiss_val

#batch sizes for train
B_A_train = round(batch_size * len_akoya_train / len_train)
B_L_train = round(batch_size * len_leica_train / len_train)
B_P_train = round(batch_size * len_philips_train / len_train)
B_O_train = round(batch_size * len_olympus_train / len_train)
B_Z_train = batch_size - B_A_train - B_L_train - B_P_train - B_O_train


#batch sizes for validation
B_A_val = round(batch_size * len_akoya_val / len_val)
B_L_val = round(batch_size * len_leica_val / len_val)
B_P_val = round(batch_size * len_philips_val / len_val)
B_O_val = round(batch_size * len_olympus_val / len_val)
B_Z_val = batch_size - B_A_val - B_L_val - B_P_val - B_O_val


def make_loader(dataset, auto_batch_size):
    return DataLoader(dataset, batch_size=auto_batch_size, shuffle=True) #, num_workers=1, pin_memory=True, persistent_workers=True, prefetch_factor=4)

akoya_loader_train = make_loader(akoya_data_train, B_A_train)
leica_loader_train = make_loader(leica_data_train, B_L_train)
philips_loader_train = make_loader(philips_data_train, B_P_train)
olympus_loader_train = make_loader(olympus_data_train, B_O_train)
zeiss_loader_train = make_loader(zeiss_data_train, B_Z_train)



print("Train batches Akoya:", len(akoya_data_train), 'batch size:', B_A_train)
print("Train batches Leica:", len(leica_data_train), 'batch size:', B_L_train)
print("Train batches Philips:", len(philips_data_train), 'batch size:', B_P_train)
print("Train batches olympus:", len(olympus_data_train), 'batch size:', B_O_train)
print("Train batches Zeiss:", len(zeiss_data_train), 'batch size:', B_Z_train)
print('len train:', len_train)


akoya_loader_val = make_loader(akoya_data_val, B_A_val)
leica_loader_val = make_loader(leica_data_val, B_L_val)
philips_loader_val = make_loader(philips_data_val, B_P_val)
olympus_loader_val = make_loader(olympus_data_val, B_O_val)
zeiss_loader_val = make_loader(zeiss_data_val, B_Z_val)

print("Val batches Akoya:", len(akoya_data_val), 'batch size:', B_A_val)
print("Val batches Leica:", len(leica_data_val), 'batch size:', B_L_val)
print("Val batches Philips:", len(philips_data_val), 'batch size:', B_P_val)
print("Val batches olympus:", len(olympus_data_val), 'batch size:', B_O_val)
print("Val batches Zeiss:", len(zeiss_data_val), 'batch size:', B_Z_val)
print('len train:', len_val)




# Defining OT-based loss function
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

    
                  
    @torch.no_grad()
    def inference(self, custom_name, scanner, data_loader, visual=True):
        weights = f'/home/leolr-int/nfs/transformed_data/weights/{custom_name}/checkpoint.pth'
        checkpoint = torch.load(weights, weights_only=False, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])        
        
        self.model.eval()
        all_preds, all_labels, all_embeddings = [], [], []
        for batch in tqdm(data_loader, desc='Inference in progress...'):
            vectors= batch['embedding'].to(self.device)
            labels = batch['label'].to(self.device)
            
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.use_amp):
                emb = self.model.bottle_neck(vectors)
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


    def training_OT(self, custom_name, num_epochs=20): 
        # we differentiate explicitly source and target scanner to apply the OT loss
        # training with validation

        training_stats = []
        min_loss_val, max_accuracy_val = float("inf"), -float('inf')
        
        #min_len_train = min(len(akoya_loader_train), len(leica_loader_train))
        #min_len_val = min(len(akoya_loader_val), len(leica_loader_val))
        
        trainable_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        optimizer = torch.optim.AdamW(trainable_params, lr=10e-4, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        ce_loss = nn.CrossEntropyLoss()  

        for epoch in range(1,num_epochs+1):
            metrics_train = {'running_loss': 0, 'predictions': [], 'labels': []}
            metrics_val   = {'running_loss': 0, 'predictions': [], 'labels': []} 

            count_train = 0
            count_val = 0
                            
            start = time.time()
            
            #1st part: train for one epoch
            self.model.train()

            #deactivate the encoder training if needed
            if self.freeze_encoder and not self.emb_mode:
                self.model.encoder.eval()

            #the target scanner is Akoya

            del_count = 0
            
            for batch_akoya, batch_leica, batch_philips, batch_olympus, batch_zeiss in tqdm(zip(akoya_loader_train, leica_loader_train, philips_loader_train, olympus_loader_train, zeiss_loader_train),
                                                    desc=f"Epoch {epoch} - Training Multi Scanner",
                                                    total = min(len(akoya_loader_train), len(leica_loader_train), len(philips_loader_train), len(olympus_loader_train), len(zeiss_loader_train))):
            
                del_count += 1
                count_train += len(batch_akoya) + len(batch_leica) + len(batch_philips) + len(batch_olympus) + len(batch_zeiss)
                #patches_akoya = (batch_akoya['embedding'] if self.emb_mode else batch_akoya['img']).to(self.device, non_blocking=True)
                #patches_leica = (batch_leica['embedding'] if self.emb_mode else batch_leica['img']).to(self.device, non_blocking=True)
        
                patch_akoya = batch_akoya['embedding'].to(self.device, non_blocking=True) 
                patch_leica = batch_leica['embedding'].to(self.device, non_blocking=True)   
                patch_philips = batch_philips['embedding'].to(self.device, non_blocking=True) 
                patch_olympus = batch_olympus['embedding'].to(self.device, non_blocking=True)
                patch_zeiss = batch_zeiss['embedding'].to(self.device, non_blocking=True)

                labels_akoya = batch_akoya['label'].to(self.device, non_blocking=True)
                labels_leica = batch_leica['label'].to(self.device, non_blocking=True)
                labels_philips = batch_philips['label'].to(self.device, non_blocking=True)
                labels_olympus = batch_olympus['label'].to(self.device, non_blocking=True)
                labels_zeiss = batch_zeiss['label'].to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
    
                with torch.autocast(device_type = self.device, dtype = torch.float16, enabled = self.use_amp):
                    
                    embedding_akoya = self.model.bottle_neck(patch_akoya) #.to(self.device, non_blocking=True)
                    embedding_leica = self.model.bottle_neck(patch_leica) #.to(self.device, non_blocking=True)
                    embedding_philips = self.model.bottle_neck(patch_philips)
                    embedding_olympus = self.model.bottle_neck(patch_olympus)
                    embedding_zeiss = self.model.bottle_neck(patch_zeiss)

                    logits_akoya = self.model(patch_akoya) #.to(self.device, non_blocking=True) 
                    logits_leica = self.model(patch_leica) #.to(self.device, non_blocking=True)
                    logits_philips = self.model(patch_philips)
                    logits_olympus = self.model(patch_olympus)
                    logits_zeiss = self.model(patch_zeiss)
                    
                    #OT_loss_train = loss_geom(embedding_akoya, embedding_leica)
                    #OT_loss_train = supervised_OT_loss(embedding_akoya, embedding_leica, labels_akoya, labels_leica)
                    #selective OT
                    #cost_fn_high = make_cost_fn(labels_akoya, labels_leica, p=p_penalty)
                    #loss_high = SamplesLoss(loss="sinkhorn", p=2, blur=0.05, scaling=0.95, backend="tensorized", cost=cost_fn_high)
                    #OT_loss_train = loss_high(embedding_akoya.detach(), embedding_leica)

                    loss_train = (ce_loss(logits_akoya, labels_akoya)
                                  + ce_loss(logits_leica, labels_leica)
                                  + ce_loss(logits_philips, labels_philips)
                                  + ce_loss(logits_olympus, labels_olympus)
                                  + ce_loss(logits_zeiss, labels_zeiss)
                                  + 0.1 * loss_geom(embedding_akoya, embedding_leica)
                                  + 0.1 * loss_geom(embedding_akoya, embedding_philips)
                                  + 0.1 * loss_geom(embedding_akoya, embedding_olympus)
                                  + 0.1 * loss_geom(embedding_akoya, embedding_zeiss))
                    
                self.grad_scaler.scale(loss_train).backward()
                self.grad_scaler.step(optimizer)
                self.grad_scaler.update()
                
                pred_akoya = torch.argmax(F.softmax(logits_akoya, dim=1), dim=1)
                pred_leica = torch.argmax(F.softmax(logits_leica, dim=1), dim=1)
                pred_philips = torch.argmax(F.softmax(logits_philips, dim=1), dim=1)
                pred_olympus = torch.argmax(F.softmax(logits_olympus, dim=1), dim=1)
                pred_zeiss = torch.argmax(F.softmax(logits_zeiss, dim=1), dim=1)
                
                #performance metrics
                
                metrics_train['running_loss'] += loss_train.detach().cpu().item()
                # we concatenate the predictions of source and target
                metrics_train['predictions'].extend(
                    np.concatenate([
                        pred_akoya.detach().cpu().numpy(),
                        pred_leica.detach().cpu().numpy(),
                        pred_philips.detach().cpu().numpy(),
                        pred_olympus.detach().cpu().numpy(),
                        pred_zeiss.detach().cpu().numpy()
                    ])
                )
                metrics_train['labels'].extend(
                    np.concatenate([
                        labels_akoya.detach().cpu().numpy(),
                        labels_leica.detach().cpu().numpy(),
                        labels_philips.detach().cpu().numpy(),
                        labels_olympus.detach().cpu().numpy(),
                        labels_zeiss.detach().cpu().numpy()
                    ])
                )

                if del_count % 50 == 0:  # Every 50 batches
                    torch.cuda.empty_cache()
                    
                # Delete intermediate tensors
                del patch_akoya, patch_leica, patch_philips, patch_olympus, patch_zeiss
                del labels_akoya, labels_leica, labels_philips, labels_olympus, labels_zeiss
                del logits_akoya, logits_leica, logits_philips, logits_olympus, logits_zeiss
                del pred_akoya, pred_leica, pred_philips, pred_olympus, pred_zeiss
                del loss_train

            epoch_loss_train = metrics_train['running_loss'] / count_train 
            epoch_balanced_accuracy_train = balanced_accuracy_score(metrics_train['labels'], metrics_train['predictions'])               
                
          
            # 2nd part: validation for one epoch 
            with torch.no_grad():
                self.model.eval()
                
                #we still work with the Train folder
                del_count = 0
                
                for batch_akoya, batch_leica, batch_philips, batch_olympus, batch_zeiss in tqdm(zip(akoya_loader_val, leica_loader_val, philips_loader_val, olympus_loader_val, zeiss_loader_val),
                                                    desc=f"Epoch {epoch} - Validation Multi Scanner",
                                                    total = min(len(akoya_loader_val), len(leica_loader_val), len(philips_loader_val), len(olympus_loader_val), len(zeiss_loader_val))):

                    del_count += 1
                    count_val += len(batch_akoya) + len(batch_leica) + len(batch_philips) + len(batch_olympus) + len(batch_zeiss)
                
                    patch_akoya = batch_akoya['embedding'].to(self.device, non_blocking=True) 
                    patch_leica = batch_leica['embedding'].to(self.device, non_blocking=True)   
                    patch_philips = batch_philips['embedding'].to(self.device, non_blocking=True) 
                    patch_olympus = batch_olympus['embedding'].to(self.device, non_blocking=True)
                    patch_zeiss = batch_zeiss['embedding'].to(self.device, non_blocking=True)

                    labels_akoya = batch_akoya['label'].to(self.device, non_blocking=True)
                    labels_leica = batch_leica['label'].to(self.device, non_blocking=True)
                    labels_philips = batch_philips['label'].to(self.device, non_blocking=True)
                    labels_olympus = batch_olympus['label'].to(self.device, non_blocking=True)
                    labels_zeiss = batch_zeiss['label'].to(self.device, non_blocking=True)

        
                    with torch.autocast(device_type = self.device, dtype = torch.float16, enabled = self.use_amp):
                        
                        embedding_akoya = self.model.bottle_neck(patch_akoya) #.to(self.device, non_blocking=True)
                        embedding_leica = self.model.bottle_neck(patch_leica) #.to(self.device, non_blocking=True)
                        embedding_philips = self.model.bottle_neck(patch_philips)
                        embedding_olympus = self.model.bottle_neck(patch_olympus)
                        embedding_zeiss = self.model.bottle_neck(patch_zeiss)

                        logits_akoya = self.model(patch_akoya) #.to(self.device, non_blocking=True) 
                        logits_leica = self.model(patch_leica) #.to(self.device, non_blocking=True)
                        logits_philips = self.model(patch_philips)
                        logits_olympus = self.model(patch_olympus)
                        logits_zeiss = self.model(patch_zeiss)
                        

                        loss_val = (ce_loss(logits_akoya, labels_akoya)
                                    + ce_loss(logits_leica, labels_leica)
                                    + ce_loss(logits_philips, labels_philips)
                                    + ce_loss(logits_olympus, labels_olympus)
                                    + ce_loss(logits_zeiss, labels_zeiss)
                                    + 0.1 * loss_geom(embedding_akoya.detach(), embedding_leica)
                                    + 0.1 * loss_geom(embedding_akoya.detach(), embedding_philips)
                                    + 0.1 * loss_geom(embedding_akoya.detach(), embedding_olympus)
                                    + 0.1 * loss_geom(embedding_akoya.detach(), embedding_zeiss))
                    
                    pred_akoya = torch.argmax(F.softmax(logits_akoya, dim=1), dim=1)
                    pred_leica = torch.argmax(F.softmax(logits_leica, dim=1), dim=1)
                    pred_philips = torch.argmax(F.softmax(logits_philips, dim=1), dim=1)
                    pred_olympus = torch.argmax(F.softmax(logits_olympus, dim=1), dim=1)
                    pred_zeiss = torch.argmax(F.softmax(logits_zeiss, dim=1), dim=1)
                    
                    #performance metrics
                
                    metrics_val['running_loss'] += loss_val.detach().cpu().item()
                    # we concatenate the predictions of source and target
                    metrics_val['predictions'].extend(
                        np.concatenate([
                            pred_akoya.detach().cpu().numpy(),
                            pred_leica.detach().cpu().numpy(),
                            pred_philips.detach().cpu().numpy(),
                            pred_olympus.detach().cpu().numpy(),
                            pred_zeiss.detach().cpu().numpy()
                        ])
                    )
                    metrics_val['labels'].extend(
                        np.concatenate([
                            labels_akoya.detach().cpu().numpy(),
                            labels_leica.detach().cpu().numpy(),
                            labels_philips.detach().cpu().numpy(),
                            labels_olympus.detach().cpu().numpy(),
                            labels_zeiss.detach().cpu().numpy()
                        ])
                    )

                    if del_count % 50 == 0:
                        torch.cuda.empty_cache()
                        
                    # Delete tensors
                    del patch_akoya, patch_leica, patch_philips, patch_olympus, patch_zeiss
                    del labels_akoya, labels_leica, labels_philips, labels_olympus, labels_zeiss
                    del logits_akoya, logits_leica, logits_philips, logits_olympus, logits_zeiss
                    del pred_akoya, pred_leica, pred_philips, pred_olympus, pred_zeiss
                    del loss_val
        
        
                epoch_loss_val = metrics_val['running_loss'] / count_val
                
                epoch_balanced_accuracy_val = balanced_accuracy_score(metrics_val['labels'], metrics_val['predictions'])
                
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
                
                train_plot(pd.DataFrame(training_stats), cm, custom_name=custom_name, OT=False)
                torch.cuda.empty_cache()

        return epoch_loss_val, epoch_balanced_accuracy_val


        
if __name__ == '__main__':
    handler = NetworkHandler(emb_mode=True)
    save_dir = '/home/leolr-int/nfs/transformed_data/weights'
    custom_name = 'OT_multi_no_detach'
    num_epochs = 20
    handler.training_OT(custom_name, num_epochs=num_epochs)