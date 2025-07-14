

'''import utils 
import pyvips

print("✅ pyvips importé avec succès")

# Crée une image vide (bleue) de 100x100
image = pyvips.Image.black(100, 100).new_from_image([0, 0, 255])  # RGB bleu

# Applique une opération simple (rotation)
rotated = image.rot90()

# Sauvegarde en PNG
rotated.write_to_file("test_output.png")

print(torch.__version__)
print(torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())'''

import time 
start_time = time.time()

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from FDA import Fourier_Domain_Adaptation
import deeplake
import numpy as np
from PIL import Image

# Load datasets
akoya_1 = deeplake.open_read_only("/home/leolr-int/data/data/patched/dim_256/Train/Subset3_Train_1_Akoya")
KFBio_1 = deeplake.open_read_only("/home/leolr-int/data/data/patched/dim_256/Train/Subset3_Train_1_KFBio")

# Get images and apply FDA
src_img = KFBio_1[200]["patch"].transpose((2, 0, 1))  # (3, H, W)
trg_img = akoya_1[200]["patch"].transpose((2, 0, 1))
output_folder = '/home/leolr-int/ASTAR_internship/Fourier_Domain_Adaptation/images'
fda_img = Fourier_Domain_Adaptation(src_img, save=False, output_folder=output_folder, display=False)  # still (3, H, W)

# Convert back to (H, W, C) and to uint8 (if needed)
fda_img = np.transpose(fda_img, (1, 2, 0)).astype(np.uint8)
fda_img_pil = Image.fromarray(fda_img)

# Torch preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

img_tensor = transform(fda_img_pil).unsqueeze(0)  # shape: (1, 3, 256, 256)

# Load pretrained model
model = models.resnet18(pretrained=True)
model.eval()

# Remove classification head
embedding_model = torch.nn.Sequential(*list(model.children())[:-1])  # (1, 512, 1, 1)

# Get embedding
with torch.no_grad():
    embedding = embedding_model(img_tensor).squeeze()  # shape: (512,)

print("Embedding shape:", embedding.shape)
print("Embedding vector:", embedding)



end_time = time.time()


print(f"Time taken for FDA: {end_time - start_time} seconds") 

'''
encoder_dir = os.path.join(BASE_MODEL_DIR, "pre_trained_weights")
model = Network(args["encoder"], encoder_dir)
network_handler = NetworkHandler(model, precision=args["precision"])
embedding_dim = model.fc.head.in_features  # assume final FC input size

# Init speed log
speed_table = {
    "slide_id": [],
    "num_patches": [],
    "start_time": [],
    "end_time": [],
    "elapsed_time": []
}

# Loop over i = 1 to 26
for i in range(1, 27):
    img_id = f"Subset3_Train_{i}_Akoya"
    ds_path = os.path.join(source_root, img_id)
    save_dir = os.path.join(target_root, img_id)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'-' * BORDER_WIDTH}\n  Extracting embeddings for {img_id}  \n{'-' * BORDER_WIDTH}\n")

    try:
        patch_ds = deeplake.open_read_only(ds_path).pytorch(transform=dataset.img_transform_fn)
        patch_loader = DataLoader(patch_ds, batch_size=args["batch_size"], shuffle=False)

        embed_ds = deeplake.create(save_dir)
        embed_ds.add_column("embedding", dtype=deeplake.types.Embedding(embedding_dim))
        embed_ds.add_column("label", dtype=deeplake.types.Int32)

        # Metadata columns
        for field in ["area", "x", "y", "w", "h", "img_idx"]:
            embed_ds.add_column(field, dtype=deeplake.types.Int32)

        start_time = datetime.now()
        img_idx = id_table[img_id]
        network_handler.extract_embeddings(patch_loader, embed_ds, img_idx=img_idx)
        end_time = datetime.now()

        elapsed = end_time - start_time
        speed_table["slide_id"].append(img_id)
        speed_table["num_patches"].append(len(patch_ds))
        speed_table["start_time"].append(start_time)
        speed_table["end_time"].append(end_time)
        speed_table["elapsed_time"].append(elapsed)

    except Exception as e:
        print(f"❌ Error on {img_id}: {e}")

# Save profiling
save_table(
    speed_table,
    save_dir=os.path.join(PROFILING_DIR, "embedding", f"{args['precision']}_precision"),
    filename=f"{args['encoder']}_akoya_speed_table")'''