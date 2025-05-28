import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from keras.models import load_model
from keras.models import Model
import torch
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import ViTModel, ViTImageProcessor
from PIL import Image
import torch

class MultimodalClusteringPipeline:
    def __init__(self, sbert_model_name='paraphrase-MiniLM-L6-v2',vit_model_name='google/vit-base-patch16-224'):
        """
        Initialize the multimodal clustering pipeline
        
        Args:
            sbert_model_name: Name of the SBERT model for text encoding
        """
        # Initialize SBERT model for text encoding
        self.sbert_model = SentenceTransformer(sbert_model_name)
        self.scaler = StandardScaler()
        # Initialize ResNet50 for image encoding (remove top layer for features)
        #custom_resnet = load_model('resnet50_csv_model.h5')
# Extract the ResNet50 base (adjust layer index if needed)
        #resnet_base = Model(
        #inputs=custom_resnet.input,
        #outputs=custom_resnet.layers[-2].output
        #)
        #self.resnet_model = resnet_base
        self.vit_model = ViTModel.from_pretrained(vit_model_name)
        self.feature_extractor = ViTImageProcessor.from_pretrained(vit_model_name)
        
    def encode_text(self, texts):
        """
        Encode texts using SBERT
        
        Args:
            texts: List of text strings
            
        Returns:
            numpy array of text embeddings
        """
        # Combine multiple text fields if needed
        text_embeddings = self.sbert_model.encode(texts)
        np.save(
        "text_embeddings.npy",text_embeddings
        )
        return text_embeddings
    
    def get_vit_embeddings(self, image_paths):
        model = ViTModel.from_pretrained('google/vit-base-patch16-224')
        processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        model.eval()
        embeddings = []
        i=0
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            inputs = processor(images=img, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            # CLS token embedding
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_emb)
            i=i+1
            if(i%50==0):
                print(i)
        return np.vstack(embeddings)

    
    def create_multimodal_embeddings(self, df, text_column, image_column):
        """
        Create combined multimodal embeddings
        
        Args:
            df: DataFrame containing the data
            text_columns: List of column names containing text
            image_column: Column name containing image paths
            
        Returns:
            Combined embeddings, text embeddings, image embeddings
        """
        # Prepare text data
        
        # Encode text and images
        print("Encoding texts...")
        #text_embeddings = self.encode_text(df[text_column].tolist())
        text_embeddings = np.load("text_embeddings.npy")
        print("Encoding images...")
        #image_embeddings = self.get_vit_embeddings(df[image_column].tolist())
        #np.save("imageembed.npy",image_embeddings)
        image_embeddings = np.load("imageembed.npy")
        import glob

        #all_embeddings = []
        #for file in sorted(glob.glob('embeddings_batches/embeddings_*.npy')):
        #    all_embeddings.append(np.load(file))
        #all_embeddings = np.concatenate(all_embeddings, axis=0)
        
        # Normalize embeddings to same scale
        text_embeddings = self.scaler.fit_transform(text_embeddings)
        print(text_embeddings.shape)
        image_embeddings = self.scaler.fit_transform(image_embeddings)
        print(image_embeddings.shape)
        # Concatenate embeddings
        multimodal_embeddings = np.concatenate([text_embeddings, image_embeddings], axis=1)
        np.save(
        "embeddings1.npy",multimodal_embeddings
        )
        print("saved")
        

        return multimodal_embeddings
    
    def perform_clustering(self, embeddings, n_clusters=None, method='kmeans'):
        """
        Perform clustering on embeddings
        
        Args:
            embeddings: Input embeddings
            n_clusters: Number of clusters (if None, will be estimated)
            method: Clustering method ('kmeans' or 'dbscan')
            
        Returns:
            Cluster labels and clustering model
        """
        #pca = PCA(n_components=100, random_state=42)
        #embeddings = pca.fit_transform(embeddings)
        if method == 'kmeans':
            clustering_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = clustering_model.fit_predict(embeddings)
        return cluster_labels, clustering_model
    
    def find_optimal_clusters(self, embeddings, max_clusters=10):
        """
        Find optimal number of clusters using silhouette score
        """
        silhouette_scores = []
        cluster_range = range(2, min(max_clusters + 1, len(embeddings)))
        
        for n_clusters in cluster_range:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            score = silhouette_score(embeddings, labels)
            silhouette_scores.append(score)
        
        optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
        return optimal_clusters
    
    def optimize_dbscan_params(self, embeddings):
        """
        Optimize DBSCAN parameters using heuristics
        """
        from sklearn.neighbors import NearestNeighbors
        
        # Estimate eps using k-distance graph
        k = 4
        nbrs = NearestNeighbors(n_neighbors=k).fit(embeddings)
        distances, indices = nbrs.kneighbors(embeddings)
        distances = np.sort(distances[:, k-1], axis=0)
        
        # Use knee point as eps estimate
        eps = np.percentile(distances, 90)
        min_samples = max(2, int(0.01 * len(embeddings)))
        
        return eps, min_samples

def clean_dataframe(df, text_column):
    # Remove rows with NA in any text column
    df_clean = df.dropna(subset=text_column)
    
    # Remove rows where any text column contains '-' or is a float
    def is_invalid(row):
        val = row[text_column]
        if val == '-':
            return True
        if isinstance(val, float):
            return True
        return False
    
    df_clean = df_clean[~df_clean.apply(is_invalid, axis=1)]
    return df_clean

# Example usage
def main():
    # Load your CSV data
    df = pd.read_csv('data.csv')
    df['image_path'] = 'data/' + df['image'].astype(str)
    # Specify your column names we have image path, category(ground truth),description, and we have display name which we drop
    text_column = "description"
    image_column = "image_path"
    df = clean_dataframe(df, text_column)
    # Initialize the pipeline
    pipeline = MultimodalClusteringPipeline()
    
    #multimodal_embeddings = np.load('embeddings.npy')
    # Create multimodal embeddings
    multimodal_embeddings= pipeline.create_multimodal_embeddings(df, text_column, image_column)
    print(multimodal_embeddings.shape)
    # Perform clustering
    cluster_labels, clustering_model = pipeline.perform_clustering(multimodal_embeddings,n_clusters=142)
    print(cluster_labels)
    # Add results to dataframe
    #df['predicted_cluster'] = cluster_labels
    y_true = df['category'].values
    ari = adjusted_rand_score(y_true, cluster_labels)
    nmi = normalized_mutual_info_score(y_true, cluster_labels)

    print(f"Adjusted Rand Index: {ari:.3f}")  
    print(f"Normalized Mutual Information: {nmi:.3f}")
    
    return df, pipeline

if __name__ == "__main__":
    df_results, pipeline = main()
