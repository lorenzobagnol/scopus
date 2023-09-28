import pandas as pd
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


# obtain section, class and subclass of each embedding and insert section in the dataset instead of a list of classification
classes= [(eval(c)[0]["section"],eval(c)[0]["class"],eval(c)[0]["subclass"]) for c in balanced["ipcr_classifications"]]
balanced["ipcr_classifications"]=classes


# order dataset with respect to sections, classes and subclasses
balanced=balanced.sort_values(by="ipcr_classifications")
classes= [c for c in balanced["ipcr_classifications"]]
print("dataset sorted")







# read and save embeddings from dataset
embeddings=[np.fromstring(el.replace('\n','')
                        .replace('[','')
                        .replace(']','')
                        .replace('  ',' '), sep=' ') for  el in balanced["embedding"]] 
embeddings=np.array(embeddings)

# prepare the final dataframe 
results=pd.DataFrame(columns=["section","class","subclass","num-docs","dist_int", "dist_ext_different_subclass","dist_ext_different_class","dist_ext_different_section"])  
#calculate distances
print("calculating distance matrix...")
distance_matrix=pairwise_distances(embeddings, metric=metric)

dist_int = {c : list() for c in classes}
dist_ext_subclass = {c : list() for c in classes}
dist_ext_class = {c : list() for c in classes}
dist_ext_sect = {c : list() for c in classes}
# calculate dist_ext_different_section
current_class=classes[0]
for i in tqdm(range(len(classes)),desc="sections"):
    if classes[i]!=current_class:
        dist_ext_sect[current_class] = np.average(dist_ext_sect[current_class])
        current_class=classes[i]  
    for j in range(len(classes)): 
        if classes[i][0] != classes[j][0]:
            dist_ext_sect[classes[i]].append(distance_matrix[i,j])
for c in dist_ext_sect.keys():
    dist_ext_sect[c] = np.average(dist_ext_sect[c])     
# calculate dist_int
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
# calculate dist_ext_different_subclass
current_class=classes[0]   
for i in tqdm(range(len(classes)),desc="subclasses"):
    if classes[i]!=current_class:
        dist_ext_subclass[current_class]  = np.average(dist_ext_subclass[current_class])
        current_class=classes[i]
    for j in range(len(classes)):   
        if classes[i][:2] == classes[j][:2] and classes[i][2] != classes[j][2]:
            dist_ext_subclass[classes[i]].append(distance_matrix[i,j])
for c in dist_int.keys():
    dist_ext_subclass[c] = np.average(dist_ext_subclass[c])
# calculate dist_ext_different_class
current_class=classes[0]   
for i in tqdm(range(len(classes)), desc="classes"):
    if classes[i]!=current_class:
        dist_ext_class[current_class] = np.average(dist_ext_class[current_class])
        current_class=classes[i]
    for j in range(len(classes)):   
        if classes[i][0] == classes[j][0] and classes[i][1]!=classes[j][1]:
            dist_ext_class[classes[i]].append(distance_matrix[i,j])
for c in dist_int.keys():
    dist_ext_class[c] = np.average(dist_ext_class[c])
    
# save distances to dataframe
for c in dist_int.keys():
    results.loc[len(results)]=[c[0],c[1],c[2],num_docs[str(c)],dist_int[c],dist_ext_subclass[c],dist_ext_class[c],dist_ext_sect[c]] 
if metric=="cosine":
    path="./final_cosine_distances.csv"
else:
    path="./final_euclidean_distances.csv"
print("saving results for "+metric+" distance")
results.to_csv(path, index=False)