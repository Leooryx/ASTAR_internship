import numpy as np
from torch.utils.data import Dataset, DataLoader
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
    def __init__(self, train_or_test, WSI_id, scanner, to_torch=False, emb_mode=True):
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
        WSI = f'Subset3_{train_or_test}_{WSI_id}_{scanner}'
        self.patches = deeplake.open_read_only(f'{directory}/{WSI}')
        

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

                if self.emb:
                    # connection to embeddings
                    directory = '/home/leolr-int/nfs/transformed_data/my_embeddings'
                    WSI = f'Subset3_{self.train_or_test}_{self.WSI_id}_{self.scanner}'
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
        WSI = f'Subset3_{self.train_or_test}_{self.WSI_id}_{self.scanner}'
        embedding_ds = deeplake.open_read_only(f'{directory}/{WSI}')
        if idx == None:
            embeddings_np = np.array(embedding_ds['embedding'])  # stack into 1 array
            embeddings_tensor = torch.from_numpy(embeddings_np).float()
            return embeddings_tensor 
        else:
            embedding = embedding_ds[idx]['embedding']
            embedding = torch.tensor(embedding, dtype=torch.float)
        return embedding

'''
idx=1500
file_test = patches_loader('Train', 1, 'Leica', to_torch=True)
file_test.display(idx)
file_test[idx]['img'].shape
#file_test.summary()
#print(file_test.to_embedding())
#file_test[idx]['embedding'] #use this form only when to_torch = True
#file_test.to_embedding(idx)'''


class multi_WSI_loader(Dataset):
    '''
    Class to handle several WSI from different scanners
    Used for training a neural network
    '''

    def __init__(self, WSI_ids, scanner, train_or_test='Train'):
        self.train_or_test = train_or_test
        self.WSI_ids = WSI_ids
        self.scanner = scanner
        
        # dictionary to store all the patches_loader objects
        self.datasets = []
        
        for WSI_id in WSI_ids: 
            ds = patches_loader(train_or_test, WSI_id, scanner, to_torch=True)
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


def make_multi_WSI_loader(WSI_ids, scanners, train_or_test, batch_size):
    datasets = []
    
    for scanner in scanners:
        dataset = multi_WSI_loader(WSI_ids, scanner, train_or_test)
        datasets.append(dataset)
    
    datasets = ConcatDataset(datasets)
    loader = DataLoader(datasets, batch_size=64, shuffle=True, num_workers=6)

    return loader



'''Test = False

if Test: 
    train_scanners = ['Akoya', 'Leica', 'KFBio']
    WSI_ids = [1,2]
    target_scanner = ['Akoya'] if 'Akoya' in train_scanners else random.choice(train_scanners)
    train_scanners.remove(target_scanner[0])
    source_scanner = train_scanners
    
    print(target_scanner)
    print(source_scanner)
    
    target_dataset = multi_WSI_loader(WSI_ids, target_scanner, train_or_test='Train')
    source_dataset = multi_WSI_loader(WSI_ids, source_scanner, train_or_test='Train')
    
    
    # DataLoaders
    batch_size = 16
    train_loader_source = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    train_loader_target = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    
    
    for batch in train_loader_source:
        images = batch['img']
        labels = batch['label']
        
        print("Source batch - Images shape:", images.shape, "Labels:", labels)
        break
    
    for batch in train_loader_target:
        images = batch['img']
        labels = batch['label']
        embeddings = batch['embedding']
        
        print("Target batch - Images shape:", images.shape, "Labels:", labels, "Emb:", embeddings)
        break
'''