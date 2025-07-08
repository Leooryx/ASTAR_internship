import os
from pathlib import Path

# ==================
#   PATH CONSTANTS
# ==================

#ROOT_DIR = str(Path(__file__).resolve().parents[2])


ROOT_DIR = "/home/leolr-int/AGGCPerturbations"
PATCH_DIR = "/home/leolr-int/data/data/patched/dim_256/Train"
METADATA_DIR = "/home/leolr-int/data/data/metadata"
EMBEDDING_DIR = "/home/leolr-int/transformed_data/new_embeddings"


SRC_DIR = os.path.join(ROOT_DIR, "src")
ASSET_DIR = os.path.join(ROOT_DIR, "assets")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
PROFILING_DIR = os.path.join(ROOT_DIR, "profiling")

DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
RUN_DIR = os.path.join(ROOT_DIR, "tensorboard")
SPLIT_DIR = os.path.join(DATA_DIR, "splits")
CONFIG_DIR = os.path.join(ROOT_DIR, "configs")
#PATCH_DIR = os.path.join(DATA_DIR, "patched")
#METADATA_DIR = os.path.join(DATA_DIR, "metadata")
#EMBEDDING_DIR = os.path.join(ROOT_DIR, "embeddings")
BASE_MODEL_DIR = os.path.join(ROOT_DIR, "model_weights")
FULL_PATCH_DIR = os.path.join(DATA_DIR, "patched_full") # only used for downstream analysis


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








