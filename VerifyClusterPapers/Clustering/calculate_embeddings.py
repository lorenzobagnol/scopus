#coding=utf-8!
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances

import pickle
from sentence_transformers import SentenceTransformer


#parameters
metric="euclidean"
subject_areas = [ "ARTS", "BIOC", "BUSI", "CENG", "CHEM", "COMP", "DECI", "DENT", "EART", "ECON", "ENER", "ENGI", "ENVI", "HEAL", "IMMU", "MATE", "MATH", "MEDI", "NEUR", "NURS", "PHAR", "PHYS", "PSYC", "SOCI", "VETE", "MULT"]
#current_subject = "AGRI"
for current_subject in subject_areas:
    #read data
    print("Loading Publications")
    df = pd.read_table("../Subjects/"+current_subject+".csv", sep='\t', engine='python', index_col=False)
    num_docs = len(df.index)
    print("Publications in this subject : "+str(num_docs))

    #read ajsc dictionary
    print("Loading AJSC Source Classifications")
    ajsc_file = open("../ajsc_dictionary.pkl", 'rb')
    ajsc_dict = pickle.load(ajsc_file)
    ajsc_file.close()

    #associate topics to publications
    print("Associating topics to publications")
    topics = []
    exclude_publications_index = []
    ii = 0
    for p in list(df[['source_id']].values):
        #drop publications from sources which are not included in the AJSC
        if p[0] not in ajsc_dict.keys():
            exclude_publications_index.append(ii)
        else:
            #process list of topics
            topic_list = ajsc_dict[p[0]]
            for jj in range(len(topic_list)):
                topic_list[jj] = topic_list[jj].strip()
            #if the list is composed of only one topic, append the topic to the global list
            if len(topic_list) == 1:
                topics.append(topic_list[0])
            #else the list is composed of more than one topic, and we exclude the publication
            else:
                exclude_publications_index.append(ii)
        ii += 1

    #drop unregistred publications from df
    print("Excluding "+str(len(exclude_publications_index))+" publications with too many topics and/or sources not registered in the AJSC")
    df = df.drop(exclude_publications_index)

    #load model
    print("Loading model")
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    #calculate embeddings
    print("Calculating abstract embeddings")
    embeddings = []
    jj = 0
    for a in df["description"].values: #REMOVE LIMIT
        print("Encoding abstract "+str(jj+1)+" of "+str(len(df["description"].values)), end="\r")
        embeddings.append(np.array(model.encode(str(a))))
        jj+=1
    print("")

    # prepare the final dataframe
    print("Compiling new dataframe with embeddings")
    results=df[['eid', 'doi', 'authkeywords']]
    #REMOVE LIMIT
    results['embedding'] = embeddings
    results['topic'] = topics #REMOVE LIMIT

    #save dataframe to file
    print("Saving Results Dataframe to File")
    results.to_csv("../Subjects and Embeddings/"+current_subject+".csv", sep="\t", index=False)
