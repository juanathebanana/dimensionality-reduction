import time
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.preprocessing import StandardScaler

%matplotlib inline



#data 

cancer=pd.read_csv('/home/juana/Downloads/breast_cancer.csv')

#selection columns to analyze
can=cancer = cancer[['type', 'NM_001127688', 'ENST00000394512', 'ENST00000508993']]
sub=can[['NM_001127688', 'ENST00000394512', 'ENST00000508993']]
#tsne
time_start = time.time()
tsne = TSNE(n_components=3, verbose=1, perplexity=10, n_iter=300)
tsne_results = tsne.fit_transform(sub)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
tsne_results[:, 0]


#plot 2d
df_subset['tsne-2d-one'] = tsne_results[:, 0]
df_subset['tsne-2d-two'] = tsne_results[:, 1]
plt.figure(figsize=(16, 10))

sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue=df1['patient diagnosis'],
    palette=sns.color_palette("tab10", 10),
    data=df_subset,
    # legend="full",
    alpha=0.3,
)

plt.show()



#plot 3d
df_subset['tsne-3d-one'] = tsne_results[:, 0]
df_subset['tsne-3d-two'] = tsne_results[:, 1]
df_subset['tsne-3d-three'] = tsne_results[:, 2]
color_map = {'M': 1, 'B': 0}  # 1 for Malignant and 0 for Benign
df_subset['color_values'] = df1['patient diagnosis'].map(color_map)


fig = plt.figure(figsize=(12, 10))

ax = fig.add_subplot(1, 1, 1, projection='3d')


ax.scatter(
    xs=df_subset['tsne-3d-one'], 
    ys=df_subset['tsne-3d-two'], 
    zs=df_subset['tsne-3d-three'], 
    c=df_subset['color_values'], 
    cmap='tab10'
)

plt.show()



#another way of visualization
df7=pd.read_csv('/home/juana/Downloads/countries.csv')
df8= df7.iloc[:, 2:]
df8= df8.applymap(lambda x: float(str(x).replace(',', '.')) if ',' in str(x) else x)
df8=df8.fillna(0)


#features = features.fillna(features.median())

# Standardize the features
scaled_features = StandardScaler().fit_transform(features)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(scaled_features)

# Visualization
plt.figure(figsize=(15,10))

# For simplicity, let's color by a continuous variable, say 'GDP ($ per capita)'. This will give a gradient color.
colors = df8['Phones (per 1000)']
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=colors, cmap='viridis', edgecolor='k')
plt.colorbar().set_label('GDP ($ per capita)')

plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('t-SNE visualization based on GDP ($ per capita)')
plt.show()
