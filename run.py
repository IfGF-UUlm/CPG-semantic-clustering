import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import warnings


def get_embeddings(corpus, embedder):
    """
    Create sentence embeddings from a text corpus using the given embedding model.

    Args:
        corpus (list[str]): Text corpus.
        embedder (sentence_transformers.SentenceTransformer): Embedding model to use.

    Returns:
       ndarray: Normalized sentence embeddings.
    """
    corpus_embeddings = embedder.encode(corpus)
    corpus_embeddings = corpus_embeddings / \
        np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
    return corpus_embeddings


def reduce_dimensions(vector, n_components=3):
    """
    Reduce dimensions of high dimensional data by applying the UMAP algorithm.

    Args:
        vector (numpy.ndarray): High dimensional data to be reduced.
        n_components (int): Number of components to reduce to. Default is 3.

    Returns:
        numpy.ndarray: Reduced data.
    """
    reduced_embeddings = umap.UMAP(
        n_components=n_components, random_state=42).fit_transform(vector)
    return reduced_embeddings


def get_clusters(data, max_clusters=50):
    """
    Cluster high dimensional data using the KMeans clustering algorithm.

    Args:
        data (numpy.ndarray): Data to be clustered.
        max_clusters (int): Maximum number of clusters to try. Default is 50.

    Returns:
        numpy.ndarray: Array of labels indicating which cluster each data point belongs to.
    """
    silhouette_score_list = []
    cluster_list = []
    for i in range(2, max_clusters+1):
        clusters = KMeans(n_clusters=i, n_init=1000,
                          max_iter=1000, random_state=42).fit(data)
        silhouette_score_list.append(silhouette_score(data, clusters.labels_))
        cluster_list.append(clusters)
    return cluster_list[silhouette_score_list.index(max(silhouette_score_list))].labels_


def assign_colors(labels, random_state=0):
    """
    Helper function to randomly assign colors to clusters.

    Args:
        labels (numpy.ndarray): Labels indicating which cluster each data point belongs to.
        random_state (int): Random seed. Default is 0.

    Returns:
        Dict[int, int]: Dictionary mapping cluster numbers to assigned colors.
    """
    cluster_color = {}
    for i in range(max(labels)+1):
        cluster_color.update({i: i})
    shuffled = shuffle(list(cluster_color.values()), random_state=random_state)
    return dict(zip(cluster_color, shuffled))


def create_figure(data, labels):
    """
    Create a 3D scatter plot with x, y, z axes representing the first three dimensions of the reduced data.
    The points are colored based on their corresponding cluster labels.

    Args:
        data (numpy.ndarray): Reduced data.
        labels (numpy.ndarray): Feature to use for color coding.

    Returns:
        None
    """
    plt.style.use('default')

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')

    ax.scatter(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        # Choose a different `random_state` to assign colors differently
        c=labels.map(assign_colors(labels, random_state=0)),
        alpha=0.6,
    )

    ax.set_xlabel('UMAP dimension 1 [a.u.]', fontsize=9, labelpad=8)
    ax.set_ylabel('UMAP dimension 2 [a.u.]', fontsize=9, labelpad=8)
    ax.set_zlabel('UMAP dimension 3 [a.u.]', fontsize=9, labelpad=8)
    ax.set_box_aspect(None, zoom=0.9)
    ax.tick_params(axis='both', which='major', labelsize=6)

    # modify these values to your liking
    ax.set_xlim(-1.5, 6)
    ax.set_ylim(3.5, 10)
    ax.set_zlim(5, 9)
    ax.xaxis.set_ticks(np.arange(-1, 6, 2))
    ax.yaxis.set_ticks(np.arange(4, 11, 2))
    ax.zaxis.set_ticks(np.arange(6, 9, 2))

    fig.savefig('AI_clusters.png', transparent=False,
                dpi=400, bbox_inches="tight")

    return None


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=(UserWarning))
    warnings.filterwarnings("ignore", category=(FutureWarning))
    torch.cuda.is_available = lambda: False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Data
    data = pd.read_csv('./recommendations.csv')

    # Get Embeddings from SentenceTransformer
    embedder = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    embeddings = get_embeddings(data['Recommendation'], embedder)

    # Reduce Embedding Dimensions
    reduced_embeddings = reduce_dimensions(embeddings)

    # Find Recommendation Clusters
    data['AI Clustering'] = get_clusters(reduced_embeddings)

    # Create a 3D Projection Plot of Recommendation Embeddings
    create_figure(reduced_embeddings, data['AI Clustering'])

    # Save to File
    data.to_csv('./recommendations.csv', index=False)
