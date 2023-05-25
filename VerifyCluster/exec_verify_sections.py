import pandas as pd
import os
from utils.section_cluster import balance_dataframe
from utils.section_cluster import plot_embedding_different_sections



df=pd.DataFrame(columns=["id","date","title","abstract","ipcr_classifications", "embedding"])
for year in os.listdir("csv"):
    if year!=".gitignore":
        print("collecting data from "+year)
        for csv in os.listdir("csv/"+year):
            filename = os.fsdecode(csv)          
            df_temp=pd.read_csv("csv/"+year+"/"+filename)
            df = pd.concat([df, df_temp], join="outer", ignore_index=True)
print("data collected.")


balanced=balance_dataframe(df)
plot_embedding_different_sections(balanced)