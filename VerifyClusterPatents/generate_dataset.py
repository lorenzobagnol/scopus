import pandas as pd
import os
import json


sections_list = ["A","B","C","D","E","F","G","H"]

sections_dict={"A":['61', '62', '46', '45', '01', '47', '63', '23', '41', '24', '43', '44', '21', '42', '22'],
"B":['05', '62', '65', '44', '64', '01', '32', '60', '25', '23', '24', '29', '42', '63', '21', '67', '43', '41', '08', '31', '66', '68', '22', '02', '28', '26', '09', '27', '61', '82', '06', '04', '30', '81', '07', '03', '33'],
"C":['22', '07', '09', '11', '23', '08', '10', '25', '04', '12', '01', '03', '02', '30', '14', '21', '40', '05', '13', '06'],
"D":['06', '21', '01', '04', '03', '07', '02', '05'],
"F":['02', '01', '04', '25', '16', '41', '03', '42', '28', '24', '21', '15', '23', '22', '26', '27', '17'],
"G":['06', '05', '01', '02', '09', '08', '07', '16', '03', '11', '10', '21', '04', '12'],
"H":['04', '01', '02', '05', '03']}

df=pd.DataFrame(columns=["id","date","title","abstract","ipcr_classifications", "embedding"])
for year in os.listdir("csv"):
    if year!=".gitignore":
        print("collecting data from "+year)
        for csv in os.listdir("csv/"+year):
            filename = os.fsdecode(csv)          
            df_temp=pd.read_csv("csv/"+year+"/"+filename)
            df = pd.concat([df, df_temp], join="outer", ignore_index=True)
print("data collected.")
df = df.sample(frac=1, random_state=10).reset_index(drop=True)

df_sections_classes_subclasses=[ [ (el["section"],el["class"],el["subclass"]) for el in eval(c)] for c in df["ipcr_classifications"]]
subclass_mask=[len(set(el))==1 for el in df_sections_classes_subclasses]
df=df.loc[subclass_mask]

all_classification=set([(eval(c)[0]["section"],eval(c)[0]["class"],eval(c)[0]["subclass"]) for c in df["ipcr_classifications"]])

el_per_class=1000
def balance(df):
    balanced_df=pd.DataFrame(columns=["id","date","title","abstract","ipcr_classifications", "embedding"])
    print("Balancing dataset with a maximum of "+str(el_per_class)+" elements per class")
    num_docs={}
    for cl in all_classification:
        print("\tsearching "+str(el_per_class)+" elements for class "+str(cl))
        mask=[(eval(c)[0]["section"],eval(c)[0]["class"],eval(c)[0]["subclass"])==cl for c in df["ipcr_classifications"]]
        temp=df.loc[mask][:el_per_class]
        df=df.loc[[not el for el in mask]]
        if len(temp)<el_per_class:
            print("\tOnly "+str(len(temp))+" elements are present in the dataset for class "+str(cl)+". Try to download more documents.")
        balanced_df = pd.concat([balanced_df, temp], join="outer", ignore_index=True)   
        num_docs[str(cl)]=len(temp)
    return balanced_df, num_docs

print("saving balanced dataset of documents and their numbers")
balanced, num_docs =balance(df)
with open('final_num_docs.json', 'w') as fp:
    json.dump(num_docs, fp)
balanced.to_csv("./final_balanced_subclasses.csv")