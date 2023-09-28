import pandas as pd
import os
import json
from tqdm import tqdm
import numpy as np
from sklearn.metrics import pairwise_distances
import argparse
import gc



parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
parser.add_argument('-m', '--metric', type=str, help="Your input string", choices=["euclidean", "cosine"], default="euclidean")
args=parser.parse_args()

metric=args.metric


balanced=pd.read_csv("/home/bagnol/progetti/Scopus/VerifyClusterPapers/Clustering/balanced.csv")
print("dataset loaded")
with open("/home/bagnol/progetti/Scopus/VerifyClusterPapers/Clustering/num_docs.json") as js:
    num_docs=json.load(js)
print("num documents loaded")


# order dataset with respect to sections
balanced=balanced.sort_values(by="macroarea")
# obtain section of each embedding and insert section in the dataset instead of a list of classification
classes= [macroarea for macroarea in balanced["macroarea"]]
print("dataset sorted")

# count how may docs for each section
num_docs_sections={c:0 for c in classes}
for cl in classes:
    num_docs_sections[cl]+=1

# read and save embeddings from dataset
embeddings=[np.fromstring(el.replace('\n','')
                        .replace('[','')
                        .replace(']','')
                        .replace('  ',' '), sep=' ') for  el in balanced["embedding"]] 
embeddings=np.array(embeddings)

# prepare the final dataframe 
results=pd.DataFrame(columns=["section","num-docs","dist_int", "dist_ext_different_section"])  
#calculate distances (l1, altre, chebishev)
print("calculating distance matrix...")
distance_matrix=pairwise_distances(embeddings, metric=metric)

dist_int = {c : list() for c in classes}
dist_ext_sect = {c : list() for c in classes}
current_class=classes[0]
# calculate external distance
for i in tqdm(range(len(classes)),desc="macroareas"):
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
    path="/home/bagnol/progetti/Scopus/VerifyClusterPapers/Clustering/macroarea_cosine_distances.csv"
else:
    path="/home/bagnol/progetti/Scopus/VerifyClusterPapers/Clustering/macroarea_euclidean_distances.csv"
gc.collect()
print("saving results for "+metric+" distance")
results.to_csv(path, index=False)
print("saved to csv")