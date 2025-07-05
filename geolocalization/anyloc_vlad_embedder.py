import torch
from torchvision import transforms as T
from PIL import Image
from third_party.AnyLoc.utilities import DinoV2ExtractFeatures, VLAD
import numpy as np

class AnyLocVLADEmbedder:
    """
    Wrapper for DINOv2 + VLAD embedding pipeline from AnyLoc.
    Allows dynamic vocabulary fitting and patch embedding for geolocalization.
    Now supports both concatenated VLAD vectors and Chamfer similarity.
    """
    def __init__(self, model_type="dinov2_vits14", layer=11, facet="key", device=None, n_clusters=32):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DinoV2ExtractFeatures(model_type, layer, facet, device=self.device)
        self.n_clusters = n_clusters
        self.vlad = None
        self.desc_dim = None
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.Resize((224, 224)),
        ])

    def fit_vocabulary(self, images):
        """
        Fit VLAD vocabulary (K-means) on a list of PIL images (patches).
        """
        feats = []
        with torch.no_grad():
            for img in images:
                timg = self.transform(img).unsqueeze(0).to(self.device)
                f = self.model(timg)  # [1, N_patches, D]
                f = torch.nn.functional.normalize(f, dim=2)
                feats.append(f.cpu())
        feats = torch.cat(feats, dim=0)  # [N, N_patches, D]
        feats_flat = feats.reshape(-1, feats.shape[-1])
        self.desc_dim = feats_flat.shape[1]
        self.vlad = VLAD(self.n_clusters, desc_dim=self.desc_dim, device=self.device)
        self.vlad.fit(feats_flat)

    def get_embedding(self, img):
        """
        Get concatenated VLAD embedding (original behavior).
        """
        feats = self.extract_dense_features(img)
        # VLAD expects torch tensor on CPU
        if isinstance(feats, np.ndarray):
            feats = torch.from_numpy(feats)
        feats = feats.cpu()  # Ensure on CPU
        vlad_desc = self.vlad.generate(feats)  # [Nc*D]
        return vlad_desc.cpu().numpy()

    def get_vlad_vectors(self, img):
        """
        Get individual VLAD cluster vectors for Chamfer similarity.
        Returns: numpy array of shape [n_clusters, desc_dim]
        """
        feats = self.extract_dense_features(img)
        if isinstance(feats, np.ndarray):
            feats = torch.from_numpy(feats)
        feats = feats.cpu()
        
        # Generate VLAD but return individual cluster vectors instead of concatenated
        vlad_desc = self.vlad.generate(feats)  # [Nc*D]
        
        # Reshape to [n_clusters, desc_dim]
        vlad_vectors = vlad_desc.reshape(self.n_clusters, self.desc_dim)
        return vlad_vectors.cpu().numpy()

    @staticmethod
    def chamfer_similarity(query_vectors, doc_vectors):
        """
        Compute Chamfer similarity between two sets of vectors.
        Args:
            query_vectors: numpy array of shape [n_query, dim]
            doc_vectors: numpy array of shape [n_doc, dim]
        Returns:
            Chamfer similarity score (float)
        """
        # Compute pairwise inner products: [n_query, n_doc]
        similarities = np.dot(query_vectors, doc_vectors.T)
        
        # For each query vector, find max similarity with any doc vector
        max_similarities = np.max(similarities, axis=1)
        
        # Sum over all query vectors
        chamfer_score = np.sum(max_similarities)
        
        return chamfer_score

    def extract_dense_features(self, image):
        """
        Given a PIL image, return dense DINOv2 features as a numpy array [N_patches, D].
        """
        with torch.no_grad():
            timg = self.transform(image).unsqueeze(0).to(self.device)
            feats = self.model(timg)  # [1, N_patches, D]
            feats = torch.nn.functional.normalize(feats, dim=2)
        return feats[0].cpu().numpy()

    def fit_vocabulary_from_features(self, features):
        """
        Fit VLAD vocabulary (K-means) directly from a numpy array of features [N, D].
        """
        self.desc_dim = features.shape[1]
        self.vlad = VLAD(self.n_clusters, desc_dim=self.desc_dim, device=self.device)
        self.vlad.fit(features)

    def load_vocabulary(self, centroids_path):
        """
        Load VLAD cluster centers from a .pt file and set as the vocabulary.
        """
        centers = torch.load(centroids_path, map_location=self.device)
        centers = centers.to(self.device)
        self.desc_dim = centers.shape[1]
        self.vlad = VLAD(self.n_clusters, desc_dim=self.desc_dim, device=self.device)
        self.vlad.c_centers = centers
        self.vlad.kmeans.centroids = centers 