import pandas as pd
import os

df=pd.DataFrame(columns=["eid","macroarea", "subject","topic", "embedding"])
subject_areas = ["AGRI", "ARTS", "BIOC", "BUSI", "CENG", "CHEM", "COMP", "DECI", "DENT", "EART", "ECON", "ENER", "ENGI", "ENVI", "HEAL", "IMMU", "MATE", "MATH", "MEDI", "NEUR", "NURS", "PHAR", "PHYS", "PSYC", "SOCI", "VETE", "MULT"]
macroareas = {"Multidisciplinary":["MULT"], "Life Sciences":["AGRI", "BIOC", "IMMU", "NEUR", "PHAR"], "Social Sciences & Humanities":["ARTS", "BUSI", "DECI", "ECON", "PSYC", "SOCI"], "Physical Sciences":["CENG", "CHEM", "COMP", "EART", "ENER", "ENGI", "ENVI", "MATE", "MATH", "PHYS"], "Health Sciences":["DENT", "HEAL", "MEDI", "NURS", "VETE"]}
macroareas_inv={}
for sbj_list,mc in zip(macroareas.values(),macroareas.keys()):
    macroareas_inv=macroareas_inv | {subj:mc for subj in sbj_list}


for subject in subject_areas:
    subject_df = pd.read_table("../Subjects and Embeddings/"+subject+".csv")
    print("current subject: "+subject)
    subject_df=subject_df.sample(frac=1, random_state=10).reset_index(drop=True)
    for current_topic in set(subject_df["topic"]):
        print("\tsearching "+str(1000)+" elements for topic "+str(current_topic))
        mask=[c==current_topic for c in subject_df["topic"]]
        temp=subject_df.loc[mask][:1000]
        subject_df=subject_df.loc[[not el for el in mask]]
        if len(temp)<1000:
            print("\tOnly "+str(len(temp))+" elements are present in the dataset for topic "+str(current_topic)+". Try to download more documents.")
        temp["subject"]=[subject]*len(temp)
        temp["macroarea"]=[macroareas_inv[subject]]*len(temp)
        df=pd.concat([df, temp], ignore_index=True) 
    df.to_csv("balanced.csv", index=False)
    os.remove("../Subjects and Embeddings/"+subject+".csv")
