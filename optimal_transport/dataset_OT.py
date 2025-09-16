import numpy as np
from torch.utils.data import Dataset, ConcatDataset
import matplotlib.pyplot as plt
import deeplake 
import torch
from PIL import Image
from torchvision import transforms


class ToPILCheck:
    def __call__(self, img):
        if isinstance(img, Image.Image):
            return img

        else:
            return transforms.ToPILImage()(img)
        

class RGBCheck:
    def __call__(self, img: Image):
        return img.convert("RGB")


        
def resize(img: np.ndarray | Image.Image) -> torch.Tensor:
    
    img_transform = transforms.Compose([
        ToPILCheck(),
        RGBCheck(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()])
        

    transformed = img_transform(img)

    return transformed

'''transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))'''


class patches_loader(Dataset):
    '''
    Class to handle patches stored as deeplake objects
    '''
    def __init__(self, subset, train_or_test, WSI_id, scanner, to_torch=False, emb_mode=True):
        self.subset = subset
        self.train_or_test = train_or_test
        self.WSI_id = WSI_id
        self.scanner = scanner
        self.to_torch = to_torch
        self.emb_mode = emb_mode
        
        
        if self.emb_mode:
            directory = '/home/leolr-int/nfs/transformed_data/my_embeddings'
        else:
            directory = f"/home/leolr-int/nfs/data/data/patched/dim_256/{train_or_test}"
        
        #here we consider only Subset3
        
        if self.subset == 'Subset1':
            WSI = f'{subset}_{train_or_test}_{WSI_id}'
            scanner = 'Akoya'
        else:
            WSI = f'{subset}_{train_or_test}_{WSI_id}_{scanner}'
        self.patches = deeplake.open_read_only(f'{directory}/{WSI}')

        
        #test to delete later maybe
        '''ds = deeplake.open_read_only('/home/leolr-int/nfs/transformed_data/all_embeddings')
        sql_query = f"SELECT * WHERE scanner = '{scanner}' AND subset = '{subset}' AND WSI_id = {WSI_id}"
        self.patches = ds.query(sql_query)'''

    def summary(self):
        return self.patches.summary()
    
    # to specify a label and then access the columns of the deeplake dataset
    def __getitem__(self, idx):
        if self.emb_mode:
            patch = self.patches[idx]
            embedding = torch.tensor(patch["embedding"], dtype=torch.float)
            label = torch.tensor(patch["label"], dtype=torch.long)
            return {'embedding':embedding, 'label':label} 
        
        else: 
            if self.to_torch:
                patch = self.patches[idx]
                
                img = patch["patch"].copy()
        
                '''if preprocess_fn is not None:
                    img = preprocess_fn(img)
            
                if apply_augmentation:
                    img = augment_fn(img)'''
            
                img = resize(img)
                
                #img = torch.tensor(img)
                label = torch.tensor(patch["label"], dtype=torch.long)
                area = torch.tensor(patch["area"], dtype=torch.long)
                x = torch.tensor(patch["x"], dtype=torch.long)
                y = torch.tensor(patch["y"], dtype=torch.long)
                w = torch.tensor(patch["w"], dtype=torch.long)
                h = torch.tensor(patch["h"], dtype=torch.long)
                
                metadata = {
                    "area": area,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    }
                
                dic = {'img':img, 'label':label, 'metadata': metadata}

                if self.emb_mode:
                    # connection to embeddings
                    directory = '/home/leolr-int/nfs/transformed_data/my_embeddings'
                    
                    if self.subset == 'Subset1':
                        WSI = f'{self.subset}_{self.train_or_test}_{self.WSI_id}'
                    else:
                        WSI = f'{self.subset}_{self.train_or_test}_{self.WSI_id}_{self.scanner}'
                    embedding_ds = deeplake.open_read_only(f'{directory}/{WSI}')
                    embedding = embedding_ds[idx]['embedding']
                    embedding = torch.tensor(embedding, dtype=torch.float)
                    dic['embedding'] = embedding
                
                return dic
                
            else: 
                #deeplake object
                return self.patches[idx]
                # Example: patches[idx]['label']

    def __len__(self):
        return len(self.patches)
    
    def display(self, idx): 
        fig, axes = plt.subplots(figsize=(4, 4))
        axes.imshow(self.patches[idx]["patch"])
        plt.show()

    
    def to_embedding(self, idx=None):
        # connection to embeddings
        directory = '/home/leolr-int/nfs/transformed_data/my_embeddings'
        if self.subset == 'Subset1':
            WSI = f'{self.subset}_{self.train_or_test}_{self.WSI_id}'
        else:
            WSI = f'{self.subset}_{self.train_or_test}_{self.WSI_id}_{self.scanner}'
        embedding_ds = deeplake.open_read_only(f'{directory}/{WSI}')
        if idx == None:
            embeddings_np = np.array(embedding_ds['embedding'])  # stack into 1 array
            embeddings_tensor = torch.from_numpy(embeddings_np).float()
            return embeddings_tensor 
        else:
            embedding = embedding_ds[idx]['embedding']
            embedding = torch.tensor(embedding, dtype=torch.float)
        return embedding

class multi_WSI_dataset(Dataset):
    '''
    Class to handle several WSI from different scanners
    Used for training a neural network
    '''

    def __init__(self, subset, WSI_ids, scanner, train_or_test='Train'):
        self.subset = subset
        self.train_or_test = train_or_test
        self.WSI_ids = WSI_ids
        self.scanner = scanner
        
        # dictionary to store all the patches_loader objects
        self.datasets = []
        
        for WSI_id in WSI_ids: 
            ds = patches_loader(subset, train_or_test, WSI_id, scanner, to_torch=True)
            _ = len(ds)
            self.datasets.append(ds)

        # index mapping
        self.index_map = []
        for ds_idx, ds in enumerate(self.datasets):
            for i in range(len(ds)):
                self.index_map.append((ds_idx, i))


    # define indexing so that the dataloader can access data    
    def __getitem__(self, idx):
        ds_idx, patch_idx = self.index_map[idx]
        return self.datasets[ds_idx][patch_idx] #which is a patches_loader object
    
    def __len__(self):
        return len(self.index_map)


def make_multi_WSI_dataset(subset, WSI_ids, scanners, train_or_test, batch_size):
    datasets = []
    
    for scanner in scanners:
        dataset = multi_WSI_dataset(subset, WSI_ids, scanner, train_or_test)
        datasets.append(dataset)
    
    datasets = ConcatDataset(datasets)
    #loader = DataLoader(datasets, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True, persistent_workers=True)

    return datasets


