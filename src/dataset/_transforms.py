import torchvision.transforms as T
import torch

class MapTransform:
    def __init__(self, google_stats, azure_stats):
        self.google_stats = google_stats
        self.azure_stats = azure_stats
        
        # Base transforms that are common for both sources
        self.base_transform = T.Compose([
            T.ToTensor(),
        ])
        
        # Create specific normalizations for each source
        self.google_normalize = T.Normalize(
            mean=self.google_stats['mean'],
            std=self.google_stats['std']
        )
        
        self.azure_normalize = T.Normalize(
            mean=self.azure_stats['mean'],
            std=self.azure_stats['std']
        )
    
    def __call__(self, img, source='google'):
        # First apply base transforms
        img = self.base_transform(img)
        
        # Then apply source-specific normalization
        if source == 'google':
            return self.google_normalize(img)
        elif source == 'azure':
            return self.azure_normalize(img)
        else:
            raise ValueError(f"Unknown source: {source}. Must be 'google' or 'azure'")