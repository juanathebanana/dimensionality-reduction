from sklearn import manifold
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


md_scaling = manifold.MDS(
n_components=2,
max_iter=50,
n_init=4,
random_state=0
)
mds_results = md_scaling.fit_transform(leukemia_use)
labs = list(leukemia['Class'].values)
leukemia_use['mds-2d-one'] = mds_results[:, 0]
leukemia_use['mds-2d-two'] = mds_results[:, 1]
# --- mds plot ---
plt.figure(figsize=(16, 10))
sns.scatterplot(
x="mds-2d-one", y="mds-2d-two",
hue=labs,
palette=sns.color_palette("hls", 4),
data=leukemia_use,
# legend="full",
alpha=0.7,
)
plt.show()
