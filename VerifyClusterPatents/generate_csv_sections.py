import pandas as pd
import os
import json
from tqdm import tqdm
import numpy as np
from sklearn.metrics import pairwise_distances



metric="euclidean"


# read and save dataset e num_docs dictionary for each subclasses
balanced=pd.read_csv("./final_balanced_subclasses.csv",index_col=False)
print("dataset loaded")
with open("./final_num_docs.json") as js:
    num_docs=json.load(js)
print("num documents loaded")


# obtain section of each embedding and insert section in the dataset instead of a list of classification
classes= [eval(c)[0]["section"] for c in balanced["ipcr_classifications"]]
balanced["ipcr_classifications"]=classes


# order dataset with respect to sections
balanced=balanced.sort_values(by="ipcr_classifications")
classes= [c for c in balanced["ipcr_classifications"]]
print("dataset sorted")

# count how may docs for each section
num_docs_sections={c:0 for c in classes}
for key in num_docs.keys():
    num_docs_sections[eval(key)[0]]+=num_docs[key]

# read and save embeddings from dataset
embeddings=[np.fromstring(el.replace('\n','')
                        .replace('[','')
                        .replace(']','')
                        .replace('  ',' '), sep=' ') for  el in balanced["embedding"]] 
embeddings=np.array(embeddings)

# prepare the final dataframe 
results=pd.DataFrame(columns=["section","num-docs","dist_int", "dist_ext_different_section"])  
#calculate distances
print("calculating distance matrix...")
distance_matrix=pairwise_distances(embeddings, metric=metric)

dist_int = {c : list() for c in classes}
dist_ext_sect = {c : list() for c in classes}
current_class=classes[0]
# calculate external distance
for i in tqdm(range(len(classes)),desc="sections"):
    if classes[i]!=current_class:
        dist_ext_sect[current_class] = np.average(dist_ext_sect[current_class])
        current_class=classes[i]  
    for j in range(len(classes)): 
        if classes[i] != classes[j]:
            dist_ext_sect[classes[i]].append(distance_matrix[i,j])
for c in dist_ext_sect.keys():
    dist_ext_sect[c] = np.average(dist_ext_sect[c])     
# calculate internal distance
current_class=classes[0]          
for i in tqdm(range(len(classes)), desc="internal distances"):        
    if classes[i]!=current_class:
        dist_int[current_class] = np.average(dist_int[current_class])
        current_class=classes[i]
    for j in range(len(classes)):
        if i==j:
            continue
        if classes[i] == classes[j]:
            dist_int[classes[i]].append(distance_matrix[i,j])
for c in dist_int.keys():
    dist_int[c] = np.average(dist_int[c])

# save distances to dataframe
for c in dist_int.keys():
    results.loc[len(results)]=[c,num_docs_sections[c],dist_int[c],dist_ext_sect[c]] 
if metric=="cosine":
    path="./section_cosine_distances.csv"
else:
    path="./section_euclidean_distances.csv"
print("saving results for "+metric+" distance")
results.to_csv(path, index=False)