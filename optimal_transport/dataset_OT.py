import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import deeplake 
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset


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
    def __init__(self, train_or_test, WSI_id, scanner, to_torch=False, emb=True):
        self.train_or_test = train_or_test
        self.WSI_id = WSI_id
        self.scanner = scanner
        directory = f"/home/leolr-int/nfs/data/data/patched/dim_256/{train_or_test}"
        #here we consider only Subset3
        WSI = f'Subset3_{train_or_test}_{WSI_id}_{scanner}'
        self.patches = deeplake.open_read_only(f'{directory}/{WSI}')
        self.to_torch = to_torch
        self.emb = emb

    def summary(self):
        return self.patches.summary()
    
    # to specify a label and then access the columns of the deeplake dataset
    def __getitem__(self, idx):
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
      


# Example
'''idx=1500
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
    loader = DataLoader(datasets, batch_size=64, shuffle=True)

    return loader
    

# Example 
'''
train_scanners = ['Akoya', 'Leica']
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
    images = batch['img'].permute(0,3,2,1)
    labels = batch['label']
    embeddings = batch['embedding']
    
    print("Target batch - Images shape:", images.shape, "Labels:", labels, "Emb:", embeddings.shape)
    break
'''






'''fig, axes = plt.subplots(1, 2, figsize=(10, 5))
dataset_path_akoya_1 = f"/home/leolr-int/nfs/data/data/patched/dim_256/Train/Subset3_Train_1_Akoya"
akoya_1 = deeplake.open_read_only(dataset_path_akoya_1)
axes[0].imshow(akoya_1[200]["patch"])
#plt.show()
dataset_path_KFbio_1 = f"/home/leolr-int/nfs/data/data/patched/dim_256/Train/Subset3_Train_1_KFBio"
KFBio_1 = deeplake.open_read_only(dataset_path_KFbio_1)
axes[1].imshow(KFBio_1[200]["patch"])
plt.suptitle(f"Subset3_Train_1_ Akoya and KFBio")
plt.savefig("/home/leolr-int/nfs/ASTAR_internship/Fourier_Domain_Adaptation/images/akoya_vs_KFBio.svg")
plt.show()'''        

#example of the deeplake dataset for embeddings
'''embed_ds = deeplake.create(save_dir)
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





'''