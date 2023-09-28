import pandas as pd
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
balanced=balanced.sort_values(by=["macroarea", "subject","topic"])
# obtain section of each embedding and insert section in the dataset instead of a list of classification
classes= [(macroarea, subject, topic) for macroarea, subject, topic in zip(balanced["macroarea"],balanced["subject"],balanced["topic"])]
print("dataset sorted")


# read and save embeddings from dataset
embeddings=[np.fromstring(el.replace('\n','')
                        .replace('[','')
                        .replace(']','')
                        .replace('  ',' '), sep=' ') for  el in balanced["embedding"]] 
embeddings=np.array(embeddings)

# prepare the final dataframe 
results=pd.DataFrame(columns=["macroarea","subject","topic","num-docs","dist_int", "dist_ext_different_topic","dist_ext_different_subject","dist_ext_different_macroarea"])  
#calculate distances
print("calculating distance matrix...")
distance_matrix=pairwise_distances(embeddings, metric=metric)

dist_int = {c : list() for c in classes}
dist_ext_subclass = {c : list() for c in classes}
dist_ext_class = {c : list() for c in classes}
dist_ext_sect = {c : list() for c in classes}
# calculate dist_ext_different_section
current_class=classes[0]
for i in tqdm(range(len(classes)),desc="tot"):
    if classes[i]!=current_class:
        dist_ext_sect[current_class] = np.average(dist_ext_sect[current_class])
        dist_int[current_class] = np.average(dist_int[current_class])
        dist_ext_subclass[current_class]  = np.average(dist_ext_subclass[current_class])
        dist_ext_class[current_class] = np.average(dist_ext_class[current_class])
        current_class=classes[i]  
    for j in range(len(classes)): 
        if classes[i][0] != classes[j][0]:
            dist_ext_sect[classes[i]].append(distance_matrix[i,j])
        if i==j:
            continue
        if classes[i] == classes[j]:
            dist_int[classes[i]].append(distance_matrix[i,j])
        if classes[i][:2] == classes[j][:2] and classes[i][2] != classes[j][2]:
            dist_ext_subclass[classes[i]].append(distance_matrix[i,j])
        if classes[i][0] == classes[j][0] and classes[i][1]!=classes[j][1]:
            dist_ext_class[classes[i]].append(distance_matrix[i,j])

gc.collect()   

for c in dist_ext_sect.keys():
    dist_ext_sect[c] = np.average(dist_ext_sect[c])  
    dist_int[c] = np.average(dist_int[c])
    dist_ext_subclass[c] = np.average(dist_ext_subclass[c])
    dist_ext_class[c] = np.average(dist_ext_class[c])

















"""
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
gc.collect()
# calculate dist_ext_different_subclass
current_class=classes[0]   
for i in tqdm(range(len(classes)),desc="topics"):
    if classes[i]!=current_class:
        dist_ext_subclass[current_class]  = np.average(dist_ext_subclass[current_class])
        current_class=classes[i]
    for j in range(len(classes)):   
        if classes[i][:2] == classes[j][:2] and classes[i][2] != classes[j][2]:
            dist_ext_subclass[classes[i]].append(distance_matrix[i,j])
for c in dist_int.keys():
    dist_ext_subclass[c] = np.average(dist_ext_subclass[c])
gc.collect()
# calculate dist_ext_different_class
current_class=classes[0]   
for i in tqdm(range(len(classes)), desc="subjects"):
    if classes[i]!=current_class:
        dist_ext_class[current_class] = np.average(dist_ext_class[current_class])
        current_class=classes[i]
    for j in range(len(classes)):   
        if classes[i][0] == classes[j][0] and classes[i][1]!=classes[j][1]:
            dist_ext_class[classes[i]].append(distance_matrix[i,j])
for c in dist_int.keys():
    dist_ext_class[c] = np.average(dist_ext_class[c])
"""

gc.collect()
    
# save distances to dataframe
for c in dist_int.keys():
    results.loc[len(results)]=[c[0],c[1],c[2],num_docs[str(c)],dist_int[c],dist_ext_subclass[c],dist_ext_class[c],dist_ext_sect[c]] 
if metric=="cosine":
    path="/home/bagnol/progetti/Scopus/VerifyClusterPapers/Clustering/topic_cosine_distances.csv"
else:
    path="/home/bagnol/progetti/Scopus/VerifyClusterPapers/Clustering/topic_euclidean_distances.csv"
gc.collect()
print("saving results for "+metric+" distance")
results.to_csv(path, index=False)
print("saved to csv")
gc.collect()