from thefuzz import fuzz
import os
import pandas as pd
from tqdm import tqdm

base_path="/home/bagnol/progetti/Scopus/"

df=pd.DataFrame(columns=["id","date","title","abstract","ipcr_classifications", "embedding"])
for year in os.listdir(base_path+"VerifyClusterPatents/csv"):
    if year!=".gitignore":
        print("collecting data from "+year)
        for csv in os.listdir(base_path+"VerifyClusterPatents/csv/"+year):
            filename = os.fsdecode(csv)          
            df_temp=pd.read_csv(base_path+"VerifyClusterPatents/csv/"+year+"/"+filename, index_col=False)
            df = pd.concat([df, df_temp], join="outer", ignore_index=True)
    if year=="2017":
        break 

df=df.loc[pd.notna(df["title"])]
del df["id"]
del df["date"]
del df["Unnamed: 0"]

same_title_authors=df[df.duplicated(subset=["title", "inventors"],keep=False)]
same_title_authors=same_title_authors.sort_values(by=["title", "inventors"], ignore_index=True)


pbar = tqdm(total=len(same_title_authors))
points_list=[0 for i in range(101)]
current_ind=0
while current_ind<len(same_title_authors):
    temp=same_title_authors[(same_title_authors["title"]==same_title_authors["title"][current_ind]) & (same_title_authors["inventors"]==same_title_authors["inventors"][current_ind])]
    temp.reset_index(inplace=True)
    for i in range(len(temp)):
        for k in range(i+1, len(temp)):
            similarity=fuzz.ratio(temp["abstract"][i], temp["abstract"][k])
            points_list[similarity]=points_list[similarity]+1
    current_ind+=len(temp)
    pbar.update(len(temp))
pbar.close()


print(points_list)
# points_list[0:100]=[0, 0, 2, 0, 0, 0, 0, 0, 0, 3, 6, 5, 4, 2, 7, 14, 7, 9, 19, 15, 18, 22, 33, 30, 39, 33, 58, 52, 68, 71, 114, 120, 127, 218, 171, 196, 252, 350, 430, 486, 590, 874, 1131, 1744, 2417, 3012, 3741, 4056, 4334, 3876, 3761, 3762, 3215, 2979, 2875, 2660, 2448, 2250, 2232, 2035, 2543, 2110, 1815, 1715, 1570, 1582, 1482, 1271, 1320, 1236, 1335, 1358, 1227, 1003, 1137, 1103, 1207, 1112, 1101, 1048, 967, 946, 998, 951, 996, 968, 1058, 1150, 913, 1028, 1055, 1082, 1156, 1201, 1286, 1600, 2095, 2339, 3732, 191617]
# points_list[0:101]=[0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 3, 6, 5, 4, 2, 7, 14, 7, 9, 19, 15, 18, 22, 33, 30, 39, 33, 58, 52, 68, 71, 114, 120, 127, 218, 171, 196, 252, 350, 430, 486, 590, 874, 1131, 1744, 2417, 3012, 3741, 4056, 4334, 3876, 3761, 3762, 3215, 2979, 2875, 2660, 2448, 2250, 2232, 2035, 2543, 2110, 1815, 1715, 1570, 1582, 1482, 1271, 1320, 1236, 1335, 1358, 1227, 1003, 1137, 1103, 1207, 1112, 1101, 1048, 967, 946, 998, 951, 996, 968, 1058, 1150, 913, 1028, 1055, 1082, 1156, 1201, 1286, 1600, 2095, 2339, 3732, 191617]
import pickle

with open("abstract_MED_list", "wb") as fp:   
    pickle.dump(points_list, fp)


from matplotlib import pyplot as plt

plt.title('istogramma distanze abstract')
plt.xlabel('Levenshtein Distance')
plt.ylabel('number of documents')
plt.bar([i for i in range(100)], points_list, width=0.5 , color='g', align="edge")
plt.savefig(base_path+"VerifyClusterPatents/hist_for_abstracts/Istogramma_0-100.png")


plt.title('istogramma distanze abstract')
plt.xlabel('Levenshtein Distance')
plt.ylabel('number of documents')
plt.bar([i for i in range(99)], points_list[:-1], width=0.5 , color='g', align="edge")
plt.savefig(base_path+"VerifyClusterPatents/hist_for_abstracts/Istogramma_0-99.png")