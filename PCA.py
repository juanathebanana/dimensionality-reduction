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


##file

food=pd.read_csv('/home/juana/Downloads/food.csv')


#elaboration data
sub = food.iloc[:, 1:]
feat_cols=sub.columns
sub = sub.fillna(0)

#pca
pca = PCA(n_components=2)#number of components
pca_result = pca.fit_transform(sub[feat_cols].values)

sub['pca-one'] = pca_result[:, 0]
sub['pca-two'] = pca_result[:, 1] 
sub['pca-three'] = pca_result[:, 1] 

print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
#plot 2d
plt.figure(figsize=(12, 10))

sns.scatterplot(
    x="pca-one", y="pca-two",
    data=df,
    hue="y",
    palette=sns.color_palette("hls", 10),
    # legend="full",
    alpha=0.3,
)

plt.show()



#plot 3 
fig = plt.figure(figsize=(12, 10))

ax = fig.add_subplot(1, 1, 1, projection='3d')

ax.scatter(
    xs=df["pca-one"], 
    ys=df["pca-two"], 
    zs=df["pca-three"], 
    c=df["y"], 
    cmap='tab10'
)

ax.set_xlabel('pca-one')
ax.set_ylabel('pca-two')
ax.set_zlabel('pca-three')

plt.show()
