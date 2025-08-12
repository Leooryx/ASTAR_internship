import os

import timm
import torch
import torch.nn as nn
from timm.layers import SwiGLUPacked
from timm.models.vision_transformer import VisionTransformer 



class Neural_Network(nn.Module):
    """
    Initialises an Artificial Neural Network with the foundation encoder Gigapath 
    and 2 layers for classification (one to create embeddings, one to classify)

    """

    def __init__(self, BASE_MODEL_DIR, freeze_encoder: bool = True, OT: bool = False, num_classes: int = 5):
        super().__init__() #super constructor for ANN in PyTorch

        self.freeze_encoder = freeze_encoder
        self.OT = OT #maybe not useful here
        
        # Define encoder
        encoder_name = 'gigapath'
        encoder_dir = os.path.join(BASE_MODEL_DIR, "pre_trained_weights")
        encoder_path = os.path.join(encoder_dir, f"{encoder_name}.pth")
        encoder = torch.load(encoder_path, map_location=torch.device("cpu"), weights_only=False)
        self.encoder = encoder

        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Define bottle neck / embeddings
        # fixed parameter value for Gigapath
        in_dim = 1536
        self.bottle_neck = nn.Linear(in_dim, 1024)

        # Define classification head
        out_dim = num_classes
        self.head = nn.Linear(1024, out_dim)

        # Define sequential architecture
        def forward(self, x): 
            if self.freeze_encoder: 
                with torch.no_grad():
                    embedding = self.bottle_neck(self.encoder(x))
            else:
                embedding = self.bottle_neck(self.encoder(x))
            logits = self.head(embedding)
            return logits




