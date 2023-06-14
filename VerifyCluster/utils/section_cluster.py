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
el_per_class=2000


def plot_embedding_different_sections(df,distance_function):
    
    for i in range(len(sections_list)):
        class_dict[sections_list[i]] = i
        color_dict[sections_list[i]] = color_palette[i]


    embeddings=[np.fromstring(el.replace('\n','')
                            .replace('[','')
                            .replace(']','')
                            .replace('  ',' '), sep=' ') for  el in df["embedding"]] 
    embeddings=np.array(embeddings)
    """
    #plot embeddings
    colors = [color_dict[eval(c)[0]["section"]] for c in df["ipcr_classifications"]]
    print("Plotting PCA of embeddings for different sections")
    pca_model = PCA(n_components = 50)
    pca_result = pca_model.fit_transform(embeddings)
    tsne_model = TSNE(n_components = 2, perplexity=2)
    tsne_result = tsne_model.fit_transform(pca_result)
    #fig = plt.figure(figsize=(12, 12))
    #ax = fig.add_subplot(projection='3d')
    # plt.scatter(tsne_result[:,0], tsne_result[:,1], tsne_result[:,2], c=colors)
    plt.scatter(tsne_result[:,0], tsne_result[:,1], c=colors)
    plt.show()"""

    #obtain class of each embedding
    print("Calculating intra-cluster and inter-cluster distances using "+distance_function+" metric")
    classes= [eval(c)[0]["section"] for c in df["ipcr_classifications"]]
    assert (embeddings.shape[0]==len(classes)), "review the code please"
    #calculate intra-cluster and inter-cluster average distances
    dist_int = {}
    dist_ext = {}
    dist_ratio = {}
    #initialize lists
    for c in sections_list:
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
    for c in sections_list:
        dist_int[c] = np.average(dist_int[c])
        dist_ext[c] = np.average(dist_ext[c])
        dist_ratio[c] = float(dist_int[c])/float(dist_ext[c])
    #print results
    print("Intra-cluster vs Inter-cluster distance ratio for each class using "+distance_function+" metric:")
    print(dist_ratio)



def balance_dataframe_sections(df):
    balanced_df=pd.DataFrame(columns=["id","date","title","abstract","ipcr_classifications", "embedding"])
    balanced=True
    print("trying to balance the dataset...")
    for cl in sections_list:
        print("searching "+str(el_per_class)+" elements for section "+str(cl))
        mask=[eval(c)[0]["section"]==cl for c in df["ipcr_classifications"]]
        temp=df.loc[mask][:el_per_class]
        df=df.loc[[not el for el in mask]]
        if len(temp)<el_per_class:
            balanced=False
            print("Only "+str(len(temp))+" elements are present in the dataset for section "+cl+". Try to download more documents.")
        balanced_df = pd.concat([balanced_df, temp], join="outer", ignore_index=True)     
    if balanced:
        print("the dataset has been balanced")
    return balanced_df
 
def show_sections(df):
    sections={}
    for c in df["ipcr_classifications"]:
        if eval(c)[0]["section"] in sections.keys():
            sections[eval(c)[0]["section"]]+=1
        else:
            sections[eval(c)[0]["section"]]=1
    print("Documents within each section are: ")
    print(sections)