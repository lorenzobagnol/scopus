import os
import pandas as pd
import pybliometrics
from pybliometrics.scopus import ScopusSearch
from sentence_transformers import SentenceTransformer
from pybliometrics.scopus.utils import config
from keybert import KeyBERT
from itertools import chain, combinations
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity




lorenzobgl_key="4584271ffe90d11d615ea935afcd5ee0"
ateneo_key="d005d4f5c91efe2932594d5f07bb06f1"
print(pybliometrics.scopus.utils.constants.CONFIG_FILE)
print(config['Authentication']['APIKey'])

# 2019 Roddaro
abstract_brevetto="""
An integrated circuit (1) operating in the quantum Hall effect regime to obtain a predetermined resistance standard (R*), in particular to perform calibration of standard electrical resistors. The integrated circuit (1) includes a plurality of field-effect transistors (20), a first section (10a) comprising at least a first and a second field-effect transistors (20a,20'a) connected in series and at least a second section (10b) comprising at least a first and a second field-effect transistors (20b,20'b) connected in series. The integrated circuit (1) comprises, in addition, a plurality of balancing ohmic contacts (5), at at least one of which (5in) a predetermined I-channel current is injected, which is, then, extracted at another ohmic contact (5out). The first and at least the second sections (10a,10b) are connected to each other in parallel, such that at a predetermined ohmic contact (5*) the predetermined standard of resistance (R*) is obtained
"""
kw_model = KeyBERT()
keywords=set()
for i in range(1,5):
    extracted_keywords = kw_model.extract_keywords(abstract_brevetto, keyphrase_ngram_range=(i, i), top_n=3)
    extracted_keywords=[a[0] for a in extracted_keywords]
    print(extracted_keywords)
    keywords.update(extracted_keywords)
keywords


# given n keywords returns all the subset of two or more element
def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1,3))
parts_of=list(powerset(keywords))

queries=list()
for subset in parts_of:
    query=""
    for el in subset:
        if len(el)>1 : query=query+"\" AND \""+el
    queries.append(query[7:])
print(queries)

#script parameters
target_directory="./SearchResults/"
schema=['eid', 'pubmed_id', 'title', 'author_names', 'author_ids', 'description', 'authkeywords', 'embedding', 'cosine_similarity']
#prepare directory for results
if not os.path.exists(target_directory):
   os.makedirs(target_directory) 

df = pd.DataFrame(columns=schema)
retrived_docs_dict={}

for query in queries:
    query = 'TITLE-ABS-KEY (\"'+query+'\")'
    search = ScopusSearch(query, download=False, subscriber=True)
    result_size=search.get_results_size()
    print("search '"+query+"' returns "+str(result_size)+" documents")
    if result_size>50000:
        continue
    search = ScopusSearch(query, verbose=True, download=True, subscriber=True)          # almeno 1.5 x 25 it/s
    if search.results!=None:
        results=search.results
        print("copying on dataframe...")
        list_of_docs=list()
        for i in tqdm(range(len(results))):  
            list_of_docs.append(results[i]._asdict()) 
        df_query=pd.DataFrame(list_of_docs)
        df = pd.concat([df, df_query], join="inner", ignore_index=True)
df.drop(index=df.loc[df["description"].isnull()].index, inplace=True) 
print("end of searching \n"+str(len(df))+" documents found.")               
df.drop_duplicates("eid", ignore_index=True, inplace=True)

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

input_embedding = model.encode(abstract_brevetto)
df['embedding']=None
df['cosine_similarity']=None
for index, row in tqdm(df.iterrows(), "Calculating embeddings...", total=len(df)):         # almeno 80 it/s
    df.loc[index]['embedding']=model.encode(row['description'])
    df.loc[index]['cosine_similarity']=cosine_similarity(row['embedding'].reshape(1, -1), input_embedding.reshape(1, -1))[0][0]
df.sort_values(by='cosine_similarity', ascending=False, inplace=True)
del df['embedding']
print("copying df to csv file...")
df.to_csv("./SearchResults/search_results", sep='\t')