import pandas as pd
import os
import json



balanced=pd.read_csv("./balanced.csv",index_col=False)
print("dataset loaded")

num_docs={}
for a,b,c in set(zip(balanced["macroarea"],balanced["subject"],balanced["topic"])):
    mask=[(macroarea, subject, topic)==(a,b,c) for macroarea, subject, topic in zip(balanced["macroarea"],balanced["subject"],balanced["topic"])]
    num_docs[str((a,b,c))]=sum(mask)
with open('num_docs.json', 'w') as fp:
    json.dump(num_docs, fp)