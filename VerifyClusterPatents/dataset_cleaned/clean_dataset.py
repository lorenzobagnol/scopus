import pandas as pd
import os

base_path="/home/bagnol/progetti/Scopus/"

df=pd.DataFrame(columns=["id","date","title","abstract","ipcr_classifications", "embedding"])
for year in os.listdir(base_path+"VerifyClusterPatents/csv"):
    if year!=".gitignore":
        print("collecting data from "+year)
        for csv in os.listdir(base_path+"VerifyClusterPatents/csv/"+year):
            filename = os.fsdecode(csv)          
            df_temp=pd.read_csv(base_path+"VerifyClusterPatents/csv/"+year+"/"+filename, index_col=False)
            df = pd.concat([df, df_temp], join="outer", ignore_index=True)

print("total number of document in the dataset: "+str(len(df)))

df=df.loc[df[["title"]].notna()["title"]]
print("number of documents with non-null title: "+str(len(df)))
print("number of duplicated elements (same title and same inventors): "+str(len(df.loc[df.duplicated(subset=["title", "inventors"],keep=False)])))
print("number of duplicated elements (same abstract): "+str(len(df.loc[df.duplicated(subset=["abstract"],keep=False)])))

df=df.sort_values(by=["publication_date"],ascending=False)
df_cleaned=df.drop_duplicates(subset=["title", "inventors"], ignore_index=True)
df_cleaned=df_cleaned.drop_duplicates(subset=["abstract"], ignore_index=True)
print("number of documents without duplicates: "+str(len(df_cleaned)))

del df_cleaned["Unnamed: 0"]
del df_cleaned["id"]
del df_cleaned["date"]
print("saving to csv")
df_cleaned.to_csv(base_path+"VerifyClusterPatents/dataset_cleaned/dataset_cleaned.csv",index=False)
print("completed")