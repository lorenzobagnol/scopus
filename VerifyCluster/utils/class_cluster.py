import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import scipy
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from sklearn.metrics import pairwise_distances


sections_list = ["A","B","C","D","E","F","G","H"]

color_palette = ["#ff0000", "#00ff00", "#0000ff", "#00ffff", "#ffff00", "#ff00ff", "#ff8888", "#88ff88", "#8888ff", "#aaaaaa"]
class_dict={}
color_dict={}
el_per_class=1000

def plot_embedding_different_classes(df, classes_list, distance_function):
    
    
    for i in range(len(classes_list)):
        class_dict[classes_list[i]] = i
    embeddings=[np.fromstring(el.replace('\n','')
                            .replace('[','')
                            .replace(']','')
                            .replace('  ',' '), sep=' ') for  el in df["embedding"]] 
    embeddings=np.array(embeddings)
    """    
    color_dict[classes_list[i]] = color_palette[i]

    plot embeddings
    print("Plotting PCA of embeddings for different classes of section "+SECTION)
    colors = [color_dict[eval(c)[0]["class"]] for c in df["ipcr_classifications"]]
    pca_model = PCA(n_components = 50)
    pca_result = pca_model.fit_transform(embeddings)
    tsne_model = TSNE(n_components = 3, perplexity=2)
    tsne_result = tsne_model.fit_transform(pca_result)
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(tsne_result[:,0], tsne_result[:,1], tsne_result[:,2], c=colors)
    plt.show(block=True)"""

    #obtain class of each embedding
    classes= [eval(c)[0]["class"] for c in df["ipcr_classifications"]]
    assert (embeddings.shape[0]==len(classes)), "review the code please"
    #calculate intra-cluster and inter-cluster average distances
    dist_int = {}
    dist_ext = {}
    dist_ratio = {}
    #initialize lists
    for c in classes_list:
        dist_int[c] = []
        dist_ext[c] = []
    distance_matrix=pairwise_distances(embeddings, metric=distance_function)
    #iterate over classes and embeddings
    for i in tqdm(range(len(classes)), leave=False):
        for j in range(len(classes)):
            if i==j:
                continue
            if classes[i] == classes[j]:
                dist_int[classes[i]].append(distance_matrix[i,j])
            else:
                dist_ext[classes[i]].append(distance_matrix[i,j])
    #average every distance vector
    for c in classes_list:
        dist_int[c] = np.average(dist_int[c])
        dist_ext[c] = np.average(dist_ext[c])
        dist_ratio[c] = float(dist_int[c])/float(dist_ext[c])
    #print results
    print("Intra-cluster vs Inter-cluster distance ratio for each class with "+ distance_function+" metric:")
    print(dist_ratio)



def show_classes_from_section(df, sec):
    print("Finding different classes within section "+sec)
    classes={}
    for c in df["ipcr_classifications"]:
        if eval(c)[0]["class"] in classes.keys():
            classes[eval(c)[0]["class"]]+=1
        else:
            classes[eval(c)[0]["class"]]=1
    print("Classes for section "+sec+" are: ")
    print(classes)
    return classes

def balance_dataframe_classes(df, classes_list, sect):
    balanced_df=pd.DataFrame(columns=["id","date","title","abstract","ipcr_classifications", "embedding"])
    print("Balancing dataset with a maximum of "+str(el_per_class)+" elements per class")
    for cl in classes_list:
        print("\tsearching "+str(el_per_class)+" elements for class "+str(cl))
        mask=[eval(c)[0]["class"]==cl for c in df["ipcr_classifications"]]
        temp=df.loc[mask][:el_per_class]
        df=df.loc[[not el for el in mask]]
        if len(temp)<el_per_class:
            print("\tOnly "+str(len(temp))+" elements are present in the dataset for class "+cl+" in section "+sect+". Try to download more documents.")
        balanced_df = pd.concat([balanced_df, temp], join="outer", ignore_index=True)     
    print("We now have "+str(len(balanced_df))+" documents with section "+sect)
    return balanced_df


