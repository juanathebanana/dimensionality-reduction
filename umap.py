import umap
import time
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

%matplotlib inline

#data
df4=pd.read_csv('/home/juana/Downloads/forestfires.csv')


su = df4[['FFMC', 'DMC', 'DC', 'ISI', 'month']]

# Compute UMAP embeddings
reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(su.drop('month', axis=1))

# Plot the UMAP embeddings colored by month
plt.figure(figsize=(10, 7))
sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=su['month'], palette='viridis', s=60, edgecolor='k')
plt.title("UMAP Embedding Colored by Month")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.show()
