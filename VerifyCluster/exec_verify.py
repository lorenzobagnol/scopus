import pandas as pd
import os
from utils.section_cluster import balance_dataframe_sections, plot_embedding_different_sections, show_sections
from utils.class_cluster import plot_embedding_different_classes, balance_dataframe_classes, show_classes_from_section


sections_list = ["A","B","C","D","E","F","G","H"]

# collecting data
df=pd.DataFrame(columns=["id","date","title","abstract","ipcr_classifications", "embedding"])
for year in os.listdir("csv"):
    if year!=".gitignore":
        print("collecting data from "+year)
        for csv in os.listdir("csv/"+year):
            filename = os.fsdecode(csv)          
            df_temp=pd.read_csv("csv/"+year+"/"+filename)
            df = pd.concat([df, df_temp], join="outer", ignore_index=True)
print("data collected.")

# shuffle data
df = df.sample(frac=1, random_state=10).reset_index(drop=True)

# clusters for different sections
balanced=balance_dataframe_sections(df)
show_sections(balanced)
for metric in ["cosine", "euclidean"]:
    plot_embedding_different_sections(balanced, metric)
   
# cluster for different classes within each section 
for sec in sections_list:
    print("\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("\n\n")
    print("Verifying embeddings with classes of section "+sec)
    print("We have a total of "+str(len(df))+" documents")
    mask_section=[eval(c)[0]["section"]==sec for c in df["ipcr_classifications"]]
    df_section=df.loc[mask_section]
    print("Only "+str(len(df_section))+" of them are classified (with ipcr) with section "+sec)
    classes_list=[el for el in show_classes_from_section(df_section, sec).keys()]
    balanced=balance_dataframe_classes(df_section, classes_list, sec)
    for metric in ["cosine", "euclidean"]:
        plot_embedding_different_classes(balanced, classes_list, metric)
        print("\n\n\n")