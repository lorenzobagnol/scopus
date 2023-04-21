import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import scipy
from sentence_transformers import SentenceTransformer


distance_function="cosine"
color_palette = ["#ff0000", "#00ff00", "#0000ff", "#00ffff", "#ffff00", "#ff00ff", "#ff8888", "#88ff88", "#8888ff", "#aaaaaa"]
class_dict={}
color_dict={}
def plot_embedding_different_subject(df, class_list):
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    for i in range(len(class_list)):
        class_dict[class_list[i]] = i
        color_dict[class_list[i]] = color_palette[i]
        
    embeddings = np.array(model.encode(df["description"].values))

    #plot embeddings
    print("Plotting PCA of embeddings")
    colors = [color_dict[c] for c in df["class"]]
    pca_model = PCA(n_components = 50)
    pca_result = pca_model.fit_transform(embeddings)
    tsne_model = TSNE(n_components = 3, perplexity=2)
    tsne_result = tsne_model.fit_transform(pca_result)
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(tsne_result[:,0], tsne_result[:,1], tsne_result[:,2], c=colors)
    plt.show()

    #obtain class of each embedding
    print("Calculating intra-cluster and inter-cluster distances")
    classes = df["class"].values.flatten()
    #calculate intra-cluster and inter-cluster average distances
    dist_int = {}
    dist_ext = {}
    dist_ratio = {}
    #initialize lists
    for c in class_list:
        dist_int[c] = []
        dist_ext[c] = []
    #iterate over classes and embeddings
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i == j:
                dist_int[classes[i]].append(0.0)
            elif classes[i] == classes[j]:
                dist_int[classes[i]].append(get_distance(embeddings[i], embeddings[j], {}))
            else:
                dist_ext[classes[i]].append(get_distance(embeddings[i], embeddings[j], {}))
    #average every distance vector
    for c in class_list:
        dist_int[c] = np.average(dist_int[c])
        dist_ext[c] = np.average(dist_ext[c])
        dist_ratio[c] = float(dist_int[c])/float(dist_ext[c])
    #print results
    print("Intra-cluster vs Inter-cluster distance ratio for each class:")
    print(dist_ratio)
        
        
        
#euclidean distance function
def get_euclidean_distance(a, b):
   return np.linalg.norm(a - b)
   
#cosine similarity distance function
def get_cosine_distance(a, b):
    return scipy.spatial.distance.cosine(a, b)

#function that returns the defined distance
def get_distance(a, b, estimators_dict):
    if distance_function == "euclidean": return get_euclidean_distance(a, b)
    if distance_function == "cosine": return get_cosine_distance(a, b)

#function that calculates the entire distance matrix
def calculate_distance_matrix(embeddings, estimators_dict):
    distance_matrix = np.zeros((len(embeddings),len(embeddings)))
    iter_count = 0
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            iter_count = iter_count + 1
            print("Calculating distance "+str(iter_count)+" of "+str(int(len(embeddings)*(len(embeddings)-1)/2)), end='\r')
            estimate = get_distance(embeddings[i], embeddings[j], estimators_dict)
            distance_matrix[i,j] = estimate
            distance_matrix[j,i] = estimate
    print("")
    return distance_matrix
