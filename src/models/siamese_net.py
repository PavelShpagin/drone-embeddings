import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class GeM(nn.Module):
    def __init__(self, num_channels=2048, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(num_channels) * p)
        self.eps = eps

    def forward(self, x):
        # Get actual number of channels from input
        num_channels = x.size(1)
        # Use only the needed number of parameters
        p = self.p[:num_channels].view(-1, 1, 1)
        
        x = x.clamp(min=self.eps)
        x = x.pow(p)
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))
        x = x.pow(1./p)
        return x

class SiameseNet(nn.Module):
    def __init__(self, backbone_name='shufflenet_v2_x1_0', pretrained=True, gem_p=3.0, embedding_dim=512):
        super(SiameseNet, self).__init__()
        
        # Get the truncated backbone
        self.backbone = self._get_backbone(backbone_name, pretrained)
        num_features = self._get_num_features(backbone_name)
        
        print(f"Initializing model with {backbone_name}, num_features: {num_features}")
        
        # Initialize GeM pooling with the correct number of channels
        self.gem_pooling = GeM(num_channels=num_features, p=gem_p)
        
        # Add embedding layer
        self.embedding = nn.Linear(num_features, embedding_dim)
        
    def _get_backbone(self, backbone_name, pretrained):
        if backbone_name.startswith('shufflenet'):
            if backbone_name == 'shufflenet_v2_x1_0':
                model = models.shufflenet_v2_x1_0(pretrained=pretrained)
            elif backbone_name == 'shufflenet_v2_x0_5':
                model = models.shufflenet_v2_x0_5(pretrained=pretrained)
            elif backbone_name == 'shufflenet_v2_x1_5':
                model = models.shufflenet_v2_x1_5(pretrained=pretrained)
            elif backbone_name == 'shufflenet_v2_x2_0':
                model = models.shufflenet_v2_x2_0(pretrained=pretrained)
            else:
                raise ValueError(f"Unsupported ShuffleNet variant: {backbone_name}")
            
            # Get all layers
            layers = list(model.children())
            
            # For ShuffleNet, we want conv1, maxpool, stage2, stage3
            backbone = nn.Sequential(
                layers[0],  # conv1
                layers[1],  # maxpool
                layers[2],  # stage2
                layers[3]   # stage3
            )
            
        elif backbone_name.startswith('resnet'):
            if backbone_name == 'resnet18':
                model = models.resnet18(pretrained=pretrained)
            elif backbone_name == 'resnet34':
                model = models.resnet34(pretrained=pretrained)
            elif backbone_name == 'resnet50':
                model = models.resnet50(pretrained=pretrained)
            elif backbone_name == 'resnet101':
                model = models.resnet101(pretrained=pretrained)
            elif backbone_name == 'resnet152':
                model = models.resnet152(pretrained=pretrained)
            else:
                raise ValueError(f"Unsupported ResNet variant: {backbone_name}")
            
            backbone = nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu,
                model.maxpool,
                model.layer1,
                model.layer2,
                model.layer3
            )
        else:
            raise ValueError(f"Unsupported backbone architecture: {backbone_name}")
            
        return backbone
    
    def _get_num_features(self, backbone_name):
        # Return the number of output features after stage3/layer3
        feature_dims = {
            'shufflenet_v2_x0_5': 192,   # channels after stage3
            'shufflenet_v2_x1_0': 232,   # channels after stage3
            'shufflenet_v2_x1_5': 352,   # channels after stage3
            'shufflenet_v2_x2_0': 464,   # channels after stage3
            'resnet18': 256,    # channels after layer3
            'resnet34': 256,    # channels after layer3
            'resnet50': 1024,   # channels after layer3
            'resnet101': 1024,  # channels after layer3
            'resnet152': 1024   # channels after layer3
        }
        if backbone_name not in feature_dims:
            raise ValueError(f"Unknown backbone: {backbone_name}")
        return feature_dims[backbone_name]

    def forward_one(self, x):
        # Extract features using the backbone
        x = self.backbone(x)
        
        # Debug print
        print(f"Feature shape after backbone: {x.shape}")
        
        # Apply GeM pooling
        x = self.gem_pooling(x)
        
        # Flatten and get embeddings
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        
        # Normalize embeddings
        x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, query, positive, negative):
        query_embed = self.forward_one(query)
        positive_embed = self.forward_one(positive)
        negative_embed = self.forward_one(negative)
        return query_embed, positive_embed, negative_embed 