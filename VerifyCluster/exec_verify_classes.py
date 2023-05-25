import pandas as pd
import os
import utils.class_cluster 
from utils.class_cluster import plot_embedding_different_classes
from utils.class_cluster import balance_dataframe, classes_from_section


sections_list = ["A","B","C","D","E","F","G","H"]

df=pd.DataFrame(columns=["id","date","title","abstract","ipcr_classifications", "embedding"])
for year in os.listdir("csv"):
    if year!=".gitignore":
        print("collecting data from "+year)
        for csv in os.listdir("csv/"+year):
            filename = os.fsdecode(csv)          
            df_temp=pd.read_csv("csv/"+year+"/"+filename)
            df = pd.concat([df, df_temp], join="outer", ignore_index=True)
print("data collected.")

for sec in sections_list:
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("\n\n")
    print("Verifying embeddings with classes of section "+sec)
    print("We have a total of "+str(len(df))+" documents")
    mask_section=[eval(c)[0]["section"]==sec for c in df["ipcr_classifications"]]
    df_section=df.loc[mask_section]
    print("Only "+str(len(df_section))+" of them are classified (with ipcr) with section "+sec)

    classes_list=[el for el in classes_from_section(df_section, sec).keys()]
    balanced=balance_dataframe(df_section, classes_list, sec)
    plot_embedding_different_classes(balanced, classes_list)
    print("\n\n\n")