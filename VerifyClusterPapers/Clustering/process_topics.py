#coding=utf-8!
import sys
import os
import pandas as pd
import json
from tqdm import tqdm
import scipy
import math
import numpy as np
from sklearn.metrics import pairwise_distances
import pickle

#parameters
metric="euclidean"
subject_areas = ["AGRI", "ARTS", "BIOC", "BUSI", "CENG", "CHEM", "COMP", "DECI", "DENT", "EART", "ECON", "ENER", "ENGI", "ENVI", "HEAL", "IMMU", "MATE", "MATH", "MEDI", "NEUR", "NURS", "PHAR", "PHYS", "PSYC", "SOCI", "VETE", "MULT"]

#distance function
def euclidean_distance(e1, e2):
    return scipy.spatial.distance.euclidean(e1, e2, w=None)

#prepare distance containers
dist_macroareas = {"Multidisciplinary":{"INT":0.0, "EXT":0.0}, "Life Sciences":{"INT":0.0, "EXT":0.0}, "Social Sciences & Humanities":{"INT":0.0, "EXT":0.0}, "Physical Sciences":{"INT":0.0, "EXT":0.0}, "Health Sciences":{"INT":0.0, "EXT":0.0}}
dist_subjects = {}
for s in subject_areas:
    dist_subjects[s] = {"INT":0.0, "EXT":0.0, "UP1":0.0}
dist_topics = {}
macroareas = {"Multidisciplinary":["MULT"], "Life Sciences":["AGRI", "BIOC", "IMMU", "NEUR", "PHAR"], "Social Sciences & Humanities":["ARTS", "BUSI", "DECI", "ECON", "PSYC", "SOCI"], "Physical Sciences":["CENG", "CHEM", "COMP", "EART", "ENER", "ENGI", "ENVI", "MATE", "MATH", "PHYS"], "Health Sciences":["DENT", "HEAL", "MEDI", "NURS", "VETE"]}
subjects = {}
for s in subject_areas:
    subjects[s] = []
topics = {}
# read one subject at a time
ii = 0
for s in subject_areas:
    print("Subject: "+s+" ("+str(ii+1)+" of "+str(len(subject_areas))+")")
    subjects[s] = []
    print("Reading File")
    df = pd.read_table("../Subjects and Embeddings/"+s+".csv")
    print("Scanning topics")
    for jj in df.index:
        print("Reading Embedding "+str(jj+1)+" of "+str(len(df.index)), end="\r")
        t = df.at[jj, "topic"]
        if t not in topics.keys():
            topics[t] = []
            dist_topics[t] = {"INT":0.0, "EXT":0.0, "UP1":0.0, "UP2":0.0}
            subjects[s].append(t)
        topics[t].append(np.fromstring(df.at[jj, "embedding"]))
    print("\n")
    ii += 1

### DEBUG START ###
'''
#print hierarchy
for m in macroareas.keys():
    print("Macroarea : "+m)
    for s in macroareas[m]:
        print("\tSubject : "+s)
        for t in subjects[s]:
            print("\t\tTopic : "+str(t)+"   len = "+str(len(topics[t])))
        print("")
    print("")
sys.exit()
'''
### DEBUG STOP ###
            
#calculate sum of internal distance of topics
for m in macroareas.keys():
    for s in macroareas[m]:
        for t in subjects[s]:
            print("Calculating sum of internal distances of topic "+str(t), end="\r")
            for i in range(len(topics[t])):
                i_d = 0.0
                for j in range(len(topics[t])):
                    if i != j:
                        i_d += euclidean_distance(topics[t][i], topics[t][j])
                dist_topics[t]["INT"] += i_d
                dist_subjects[s]["INT"] += i_d
                dist_macroareas[m]["INT"] += i_d              
print("")

#calculate distance between different topics of the same subject
for m in macroareas.keys():
    for s in macroareas[m]:
        print("Calculating sum of external distances of topics in subject "+str(s))
        tot_s = 0
        for i in range(len(subjects[s])):
            print("Calculating sum of external distances for topic "+str(i+1)+" of "+str(len(subjects[s])), end="\r")
            t_i = topics[subjects[s][i]]
            tot_s += len(t_i)
            for j in range(len(t_i)):
                d_j = 0.0
                for k in range(len(subjects[s])):
                    if i == k:
                        continue
                    t_k = topics[subjects[s][k]]
                    for l in range(len(t_k)):
                        d_j += euclidean_distance(t_i[j], t_k[l])
                dist_topics[subjects[s][i]]["EXT"] += d_j
                dist_subjects[s]["INT"] += d_j
                dist_macroareas[m]["INT"] += d_j
        print("")
    
#calculate distance between topics of different subjects in the same macroarea
for m in macroareas.keys():
    print("Calculating sum of up-one-level distances of topics in macroarea "+str(m))
    for s in macroareas[m]:
        print("Calculating sum of up-one-level distances of topics in subject "+str(s))
        for i in range(len(subjects[s])):
            print("Calculating sum of up-one-level distances for topic "+str(i+1)+" of "+str(len(subjects[s])), end="\r")
            t_i = topics[subjects[s][i]]
            for j in range(len(t_i)):
                d_j = 0.0
                tot_j = 0
                for r in macroareas[m]:
                    if r == s:
                        continue
                    for k in range(len(subjects[r])):
                        t_k = topics[subjects[r][k]]
                        for l in range(len(t_k)):
                            d_j += euclidean_distance(t_i[j], t_k[l])
                        tot_j += len(t_k)
                if tot_j > 0:
                    d_j = d_j / float(tot_j)
                dist_topics[subjects[s][i]]["UP1"] += d_j
                dist_subjects[s]["EXT"] += d_j
                dist_macroareas[m]["INT"] += d_j
        print("")
        
#calculate distance between topics of different subjects in different macroareas
for m in macroareas.keys():
    print("Calculating sum of up-two-levels distances of topics in macroarea "+str(m))
    for s in macroareas[m]:
        print("Calculating sum of up-two-levels distances of topics in subject "+str(s))
        for i in range(len(subjects[s])):
            print("Calculating sum of up-two-levels distances for topic "+str(i+1)+" of "+str(len(subjects[s])), end="\r")
            t_i = topics[subjects[s][i]]
            for j in range(len(t_i)):
                d_j = 0.0
                tot_j = 0
                for n in macroareas.keys():
                    if n==m:
                        continue
                    for r in macroareas[n]:        
                        for k in range(len(subjects[r])):
                            t_k = topics[subjects[r][k]]
                            for l in range(len(t_k)):
                                d_j += euclidean_distance(t_i[j], t_k[l])
                            tot_j += len(t_k)
                if tot_j > 0:
                    d_j = d_j / float(tot_j)
                dist_topics[subjects[s][i]]["UP2"] += d_j
                dist_subjects[s]["UP1"] += d_j
                dist_macroareas[m]["EXT"] += d_j
        print("")
                
#calculate total numbers of elements
print("Calculating total number of elements in each hierarchy entry")
len_tot = 0
len_m_dict = {}
len_s_dict = {}
len_t_dict = {}
for m in macroareas.keys():
    len_m = 0
    for s in macroareas[m]:
        len_s = 0
        for t in subjects[s]:
            len_t = len(topics[t])
            len_t_dict[t] = len_t
            len_s += len_t
        len_s_dict[s] = len_s
        len_m += len_s
    len_m_dict[m] = len_m
    len_tot += len_m
    
#calculate average distance of each group by dividing by the number of opposing elements
print("Averaging over number of outer elements")
for m in macroareas.keys():
    for s in macroareas[m]:
        for t in subjects[s]:
            if len_t_dict[t] > 1:
                dist_topics[t]["INT"] = dist_topics[t]["INT"]/float(len_t_dict[t]-1)
            if len(subjects[s]) > 1:
                dist_topics[t]["EXT"] = dist_topics[t]["EXT"]/float(len_s_dict[s] - len_t_dict[t])
            if len(macroareas[m]) > 1:
                dist_topics[t]["UP1"] = dist_topics[t]["UP1"]/float(len_m_dict[m] - len_s_dict[s])
            if len(macroareas.keys()) > 1:
                dist_topics[t]["UP2"] = dist_topics[t]["UP2"]/float(len_tot - len_m_dict[m])
        if len_s_dict[s] > 1:
            dist_subjects[s]["INT"] = dist_subjects[s]["INT"]/float(len_s_dict[s]-1)
        if len(macroareas[m]) > 1:
            dist_subjects[s]["EXT"] = dist_subjects[s]["EXT"]/float(len_m_dict[m] - len_s_dict[s])
        if len(macroareas.keys()):
            dist_subjects[s]["UP1"] = dist_subjects[s]["UP1"]/float(len_tot - len_m_dict[m])
    if len_m_dict[m] > 1:
        dist_macroareas[m]["INT"] = dist_macroareas[m]["INT"]/float(len_m_dict[m]-1)
    if len(macroareas.keys()):
        dist_macroareas[m]["EXT"] = dist_macroareas[m]["EXT"]/float(len_tot - len_m_dict[m])

#wrap average distances by dividing by the number of elements in the group
print("Averaging over number of inner elements")
for m in macroareas.keys():
    for s in macroareas[m]:
        for t in subjects[s]:
            dist_topics[t]["INT"] = dist_topics[t]["INT"]/float(len_t_dict[t])
            dist_topics[t]["EXT"] = dist_topics[t]["EXT"]/float(len_t_dict[t])
            dist_topics[t]["UP1"] = dist_topics[t]["UP1"]/float(len_t_dict[t])
            dist_topics[t]["UP2"] = dist_topics[t]["UP2"]/float(len_t_dict[t])
        dist_subjects[s]["INT"] = dist_subjects[s]["INT"]/float(len_s_dict[s])
        dist_subjects[s]["EXT"] = dist_subjects[s]["EXT"]/float(len_s_dict[s])
        dist_subjects[s]["UP1"] = dist_subjects[s]["UP1"]/float(len_s_dict[s])
    dist_macroareas[m]["INT"] = dist_macroareas[m]["INT"]/float(len_m_dict[m])
    dist_macroareas[m]["EXT"] = dist_macroareas[m]["EXT"]/float(len_m_dict[m])

#save results with pickle
print("Pickling distance dictionaries")
out_file = open("distance_macroareas.pkl", 'wb')
pickle.dump(dist_macroareas, out_file)
out_file.close()
out_file = open("distance_subjects.pkl", 'wb')
pickle.dump(dist_subjects, out_file)
out_file.close()
out_file = open("distance_topics.pkl", 'wb')
pickle.dump(dist_topics, out_file)
out_file.close()

#print results to file
print("Saving results as Distances.txt")
out_file = open("Distances.txt", 'w')
for m in macroareas.keys():
    out_file.write("Macroarea "+m+" :\n\tINT = "+str(dist_macroareas[m]["INT"])+"\n\tEXT = "+str(dist_macroareas[m]["EXT"])+"\n\n")
for m in macroareas.keys():
    for s in macroareas[m]:
        out_file.write("Subject "+s+" :"+"\n\tINT = "+str(dist_subjects[s]["INT"])+"\n\tEXT = "+str(dist_subjects[s]["EXT"])+"\n\tUP1 = "+str(dist_subjects[s]["UP1"])+"\n\n")
for m in macroareas.keys():
    for s in macroareas[m]:
        for t in subjects[s]:
            out_file.write("Topic "+str(t)+" :"+"\n\tINT = "+str(dist_topics[t]["INT"])+"\n\tEXT = "+str(dist_topics[t]["EXT"])+"\n\tUP1 = "+str(dist_topics[t]["UP1"])+"\n\tUP2 = "+str(dist_topics[t]["UP2"])+"\n\n")
out_file.close()

#print results to screen
print("Results")
for m in macroareas.keys():
    print("Macroarea "+m+" :"+"\n\tINT = "+str(dist_macroareas[m]["INT"])+"\n\tEXT = "+str(dist_macroareas[m]["EXT"])+"\n")
for m in macroareas.keys():
    for s in macroareas[m]:
        print("Subject "+s+" :"+"\n\tINT = "+str(dist_subjects[s]["INT"])+"\n\tEXT = "+str(dist_subjects[s]["EXT"])+"\n\tUP1 = "+str(dist_subjects[s]["UP1"])+"\n")
for m in macroareas.keys():
    for s in macroareas[m]:
        for t in subjects[s]:
            print("Topic "+str(t)+" :"+"\n\tINT = "+str(dist_topics[t]["INT"])+"\n\tEXT = "+str(dist_topics[t]["EXT"])+"\n\tUP1 = "+str(dist_topics[t]["UP1"])+"\n\tUP2 = "+str(dist_topics[t]["UP2"])+"\n")

#terminate execution
print("Execution completed successfully")