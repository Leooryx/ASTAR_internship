import torch
import torch.nn as nn
import random

import torch
import torch.nn as nn
import random

class Network(nn.Module):
    """
    Initialises an Artificial Neural Network with the foundation encoder Gigapath 
    and 2 layers for classification (one to create embeddings, one to classify)

    """

    def __init__(self, emb_mode: bool = False, freeze_encoder: bool = True, OT: bool = False, num_classes: int = 5):
        super().__init__() #super constructor for ANN in PyTorch

        self.freeze_encoder = freeze_encoder
        self.OT = OT #maybe not useful here
        self.emb_mode = emb_mode
        
        # Define encoder

        if not self.emb_mode:
            encoder_name = 'gigapath'
            BASE_MODEL_DIR = '/home/leolr-int/AGGCPerturbations/model_weights'
            encoder_dir = os.path.join(BASE_MODEL_DIR, "pre_trained_weights")
            encoder_path = os.path.join(encoder_dir, f"{encoder_name}.pth")
            encoder = torch.load(encoder_path, map_location=torch.device("cpu"), weights_only=False)
            self.encoder = encoder

            if self.freeze_encoder:
                for param in self.encoder.parameters():
                    param.requires_grad = False
        
        else:
            #no need for the encoder
            self.encoder = None #I didnt know we could do this

        # Define bottle neck / embeddings
        # fixed parameter value for Gigapath
        in_dim = 1536
        self.bottle_neck = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(), #do not forget ReLU
            nn.Dropout(p=0.5))

        # Define classification head
        self.head = nn.Linear(1024, num_classes)

    # Define sequential architecture
    def forward(self, x): 
        if self.emb_mode:
            #x is  a vector here
            embedding = self.bottle_neck(x)
        else:
            #x is an image here
            if self.freeze_encoder: 
                with torch.no_grad():
                    encoded = self.encoder(x)
            else:
                encoded = self.encoder(x)
            embedding = self.bottle_neck(encoded)
        logits = self.head(embedding)
        return logits

        

    

