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

#deprecated
def make_multi_WSI_loader(subset, WSI_ids, scanners, train_or_test, batch_size):
    datasets = []
    
    for scanner in scanners:
        dataset = multi_WSI_loader(subset, WSI_ids, scanner, train_or_test)
        datasets.append(dataset)
    
    datasets = ConcatDataset(datasets)
    loader = DataLoader(datasets, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True, persistent_workers=True)

    return loader



'''Test = False

if Test: 
    subset = 'Subset3'
    train_scanners = ['Akoya', 'Leica', 'KFBio']
    WSI_ids = [1,2]
    target_scanner = ['Akoya'] if 'Akoya' in train_scanners else random.choice(train_scanners)
    train_scanners.remove(target_scanner[0])
    source_scanner = train_scanners
    
    print(target_scanner)
    print(source_scanner)
    
    target_dataset = multi_WSI_loader(subset, WSI_ids, target_scanner, train_or_test='Train')
    source_dataset = multi_WSI_loader(subset, WSI_ids, source_scanner, train_or_test='Train')
    
    
    # DataLoaders
    batch_size = 16
    train_loader_source = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    train_loader_target = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    
    
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










### SQL BASED QUERIES TO DATA LOADER (mine)
extraction_config = [
            {'subset': 'Subset1', 'scanners': 'Akoya', 'WSI_ids': list(range(1,52+1)), 'train_or_test': 'Train'},
            {'subset': 'Subset3', 'scanners': 'Akoya', 'WSI_ids': list(range(1,26+1)), 'train_or_test': 'Train'},
            {'subset': 'Subset3', 'scanners': 'Leica', 'WSI_ids': list(range(1,26+1)), 'train_or_test': 'Train'}
        ]

def random_split(idx_range, split_ratio=0.7):
    random.shuffle(idx_range)
    num_train = int(np.ceil(split_ratio * len(idx_range)))
    return idx_range[:num_train], idx_range[num_train:]

def embedding_transform_fn(row):
    embedding = torch.tensor(row["embedding"])
    label = torch.tensor(row["label"], dtype=torch.long)
    #subset = torch.tensor(row["subset"], dtype=torch.str)
    #scanner = torch.tensor(row["scanner"], dtype=torch.str)
    #WSI_id = torch.tensor(row["WSI_id"], dtype=torch.long)
    #train_or_test = torch.tensor(row["train_or_test"], dtype=torch.str)

    return {'embedding': embedding, 'label': label}


   
def split(extraction_config, batch_size):
    #no need to specify 'Train' for the dataset because the split is obvioulsy for it
    line_query_train = []
    line_query_val = []

    for config in extraction_config:
        config['train_idx'], config['val_idx'] = random_split(config['WSI_ids'])

        line_query_train.append(f"(scanner = '{config['scanners'][0]}' AND subset = '{config['subset']}' AND WSI_id IN {tuple(config['train_idx'])})")
        line_query_val.append(f"(scanner = '{config['scanners'][0]}' AND subset = '{config['subset']}' AND WSI_id IN {tuple(config['val_idx'])})")

    # Join the conditions with " OR "
    where_clause_train = " OR ".join(line_query_train)
    where_clause_val = " OR ".join(line_query_val)
    
    # Construct the full SQL query
    sql_query_train = f"SELECT * WHERE {where_clause_train}"
    sql_query_val = f"SELECT * WHERE {where_clause_val}"
    print(sql_query_val)

    #from deeplake to dataloader
    ds = deeplake.open_read_only('/home/leolr-int/nfs/transformed_data/all_embeddings')
    print(ds.summary())

    ds_train = ds.query(sql_query_train).pytorch(transform=embedding_transform_fn)
    ds_val = ds.query(sql_query_val).pytorch(transform=embedding_transform_fn)

    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    loader_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True, persistent_workers=True, prefetch_factor=4)

    return loader_train, loader_val

TRAIN = False 
if TRAIN:
    batch_size=64
    loader_train, loader_val = split(extraction_config, batch_size)
    print(len(loader_train))
    print(len(loader_val))
    
    
    
    scanners_train = ['Akoya', 'Leica'] #add Leica later
    train_or_test = 'Train'
    WSI_ids_train = [i for i in range(1,26+1)] #for testing
    WSI_ids_val = [i for i in range(1,26+1)] #i have to be sure that all labels are represented in validation data
    batch_size = 64
    subset = 'Subset3'
    training_stats = []
    handler = NetworkHandler(emb_mode=True)
    save_dir = '/home/leolr-int/nfs/transformed_data/weights'
    custom_name = 'test_deeplake_all'
    
    num_epochs = 50
    handler.training_no_OT(scanners_train, batch_size, num_epochs)
#####################################


### Loading data technique that works fast, used for baseline:

from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset, WeightedRandomSampler
import numpy as np
from typing import List, Tuple, Optional
import multiprocessing as mp

class OptimizedMultiWSILoader(Dataset):
    '''
    Optimized class to handle several WSI from different scanners
    with improved memory efficiency and faster indexing
    '''
    def __init__(self, subset, WSI_ids, scanner, train_or_test='Train'):
        self.subset = subset
        self.train_or_test = train_or_test
        self.WSI_ids = WSI_ids
        self.scanner = scanner
        
        # Store dataset info instead of full datasets to save memory
        self.dataset_info = []
        self.cumulative_lengths = [0]
        
        for WSI_id in WSI_ids:
            # Only store the parameters needed to create the dataset
            dataset_params = {
                'subset': subset,
                'train_or_test': train_or_test,
                'WSI_id': WSI_id,
                'scanner': scanner,
                'to_torch': True
            }
            
            # Create dataset temporarily to get length, then discard
            temp_ds = patches_loader(**dataset_params)
            length = len(temp_ds)
            
            self.dataset_info.append((dataset_params, length))
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + length)
            del temp_ds  # Free memory immediately
        
        self.total_length = self.cumulative_lengths[-1]
        
        # Lazy loading cache for datasets
        self._dataset_cache = {}
    
    def _get_dataset(self, ds_idx: int):
        """Lazy load dataset and cache it"""
        if ds_idx not in self._dataset_cache:
            params, _ = self.dataset_info[ds_idx]
            self._dataset_cache[ds_idx] = patches_loader(**params)
        return self._dataset_cache[ds_idx]
    
    def __getitem__(self, idx: int):
        # Fast binary search to find which dataset this index belongs to
        ds_idx = np.searchsorted(self.cumulative_lengths[1:], idx, side='right')
        local_idx = idx - self.cumulative_lengths[ds_idx]
        
        dataset = self._get_dataset(ds_idx)
        return dataset[local_idx]
    
    def __len__(self):
        return self.total_length

def make_dataset(subset, indices, scanner, train_or_test):
    return OptimizedMultiWSILoader(subset, indices, scanner, train_or_test)

def split_dataset(dataset, split_ratio=0.7, seed=42):
    """Split dataset with fixed seed for reproducibility"""
    generator = torch.Generator().manual_seed(seed) if 'torch' in globals() else None
    n_total = len(dataset)
    n_train = int(n_total * split_ratio)
    n_val = n_total - n_train
    return random_split(dataset, [n_train, n_val], generator=generator)

def create_balanced_sampler(train_datasets: List) -> WeightedRandomSampler:
    """Create a more efficient balanced sampler"""
    lengths = [len(ds) for ds in train_datasets]
    total_length = sum(lengths)
    
    # More efficient weight calculation
    weights = np.concatenate([
        np.full(length, 1.0 / length, dtype=np.float32) 
        for length in lengths
    ])
    
    return WeightedRandomSampler(
        weights=weights,
        num_samples=total_length,
        replacement=True
    )

def get_optimal_num_workers() -> int:
    """Determine optimal number of workers based on system"""
    cpu_count = mp.cpu_count()
    # Use 75% of available CPUs, but cap at 12 for diminishing returns
    return min(max(1, int(cpu_count * 0.75)), 12)

def make_loaders(
    datasets: List[Tuple], 
    batch_size: int = 64, 
    num_workers: Optional[int] = None, 
    split_ratio: float = 0.7,
    pin_memory: bool = None,
    prefetch_factor: int = 3,
    persistent_workers: bool = True
):
    """
    Optimized loader creation with smart defaults
    
    Args:
        datasets: List of (subset, indices, scanner, mode) tuples
        batch_size: Batch size for training
        num_workers: Number of worker processes (auto-detected if None)
        split_ratio: Train/validation split ratio
        pin_memory: Pin memory for GPU transfer (auto-detected if None)
        prefetch_factor: How many batches to prefetch per worker
        persistent_workers: Keep workers alive between epochs
    """
    
    # Auto-detect optimal settings if not provided
    if num_workers is None:
        num_workers = get_optimal_num_workers()
    
    if pin_memory is None:
        # Auto-detect if CUDA is available
        try:
            import torch
            pin_memory = torch.cuda.is_available()
        except ImportError:
            pin_memory = False
    
    print(f"Using {num_workers} workers, pin_memory={pin_memory}")
    
    # Create datasets with progress indication
    train_datasets, val_datasets = [], []
    print(f"Creating {len(datasets)} dataset splits...")
    
    for i, (subset, indices, scanner, mode) in enumerate(datasets):
        print(f"Processing dataset {i+1}/{len(datasets)}: {subset}_{scanner}")
        
        ds = make_dataset(subset, indices, scanner, mode)
        ds_train, ds_val = split_dataset(ds, split_ratio)
        
        train_datasets.append(ds_train)
        val_datasets.append(ds_val)
    
    # Combine datasets
    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)
    
    print(f"Total training samples: {len(train_dataset)}")
    print(f"Total validation samples: {len(val_dataset)}")
    
    # Create balanced sampler
    sampler = create_balanced_sampler(train_datasets)
    
    # Common loader arguments
    common_args = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'persistent_workers': persistent_workers and num_workers > 0,
        'prefetch_factor': prefetch_factor if num_workers > 0 else 2,
    }
    
    # Training loader with sampler
    loader_train = DataLoader(
        train_dataset,
        sampler=sampler,
        **common_args
    )
    
    # Validation loader with shuffle=False
    loader_val = DataLoader(
        val_dataset,
        shuffle=False,
        **common_args
    )
    
    return loader_train, loader_val

# Enhanced usage example with better configuration
def create_optimized_loaders():
    """Example usage with optimized settings"""
    
    # Configuration
    batch_size = 64
    subset = "Subset3"
    
    # Use more efficient range creation
    idx_subset1 = list(range(1, 53))  # 1 to 52 inclusive
    idx_subset3_akoya = list(range(1, 27))  # 1 to 26 inclusive  
    idx_subset3_leica = list(range(1, 27))  # 1 to 26 inclusive
    
    datasets = [
        ('Subset1', idx_subset1, "Akoya", "Train"),
        (subset, idx_subset3_akoya, "Akoya", "Train"),
        (subset, idx_subset3_leica, "Leica", "Train"),
    ]
    
    # Create loaders with optimized settings
    loader_train, loader_val = make_loaders(
        datasets, 
        batch_size=batch_size,
        split_ratio=0.7,
        # num_workers will be auto-detected
        # pin_memory will be auto-detected
    )
    
    return loader_train, loader_val

# Example usage
if __name__ == "__main__":
    loader_train, loader_val = create_optimized_loaders()
    print(f"Training batches per epoch: {len(loader_train)}") #34k
    print(f"Validation batches per epoch: {len(loader_val)}") #~15k