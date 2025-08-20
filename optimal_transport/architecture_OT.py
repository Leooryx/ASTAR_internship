import torch
import torch.nn as nn
import random

class Gigapath_Network(nn.Module):
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
        self.bottle_neck = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(), #do not forget ReLU
            nn.Dropout(p=0.5))

        # Define classification head
        out_dim = num_classes
        self.head = nn.Linear(1024, out_dim)

        # Define sequential architecture
        def forward(self, x): 
            if self.freeze_encoder: 
                with torch.no_grad():
                    encoded = self.encoder(x)
                embedding = self.bottle_neck(encoded)
            else:
                embedding = self.bottle_neck(self.encoder(x))
            logits = self.head(embedding)
            return logits




class Network(nn.Module):
    """
    Initialises an Artificial Neural Network with the foundation encoder Gigapath 
    and 2 layers for classification (one to create embeddings, one to classify)

    """

    def __init__(self, BASE_MODEL_DIR, emb_mode: bool = False, freeze_encoder: bool = True, OT: bool = False, num_classes: int = 5):
        super().__init__() #super constructor for ANN in PyTorch

        self.freeze_encoder = freeze_encoder
        self.OT = OT #maybe not useful here
        self.emb_mode = emb_mode
        
        # Define encoder

        if not self.emb_mode:
            encoder_name = 'gigapath'
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
        out_dim = num_classes
        self.head = nn.Linear(1024, out_dim)

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
                embedding = self.bottle_neck(self.encoder(x))
            logits = self.head(embedding)
            return logits
    

