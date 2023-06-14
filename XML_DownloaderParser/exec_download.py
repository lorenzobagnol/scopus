import xml.etree.ElementTree as ET
import pandas as pd
from sentence_transformers import SentenceTransformer
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import zipfile
import os
 

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def create_dataset(file_name, year):
    df=pd.DataFrame(columns=["id","date", "application_reference", "title","abstract","ipcr_classifications", "claims", "embedding"])
    with open("xmls/"+file_name) as input:
        n_line=0
        temp_file = open('xmls/temp/temp.xml','w')
        while True:
            line = input.readline()
            if not line: 
                break
            if line.startswith('<?xml') and n_line!=0:
                temp_file.close()
                tree = ET.parse('xmls/temp/temp.xml')
                us_patent_grant = tree.getroot()
                title=None
                ab=None
                application_ref={}
                for application_reference in us_patent_grant.iter("application-reference"):
                    application_ref["application_id"]=application_reference.find("document-id").find("doc-number").text
                    application_ref["application_date"]=application_reference.find("document-id").find("date").text
                for us_bibliographic_data_grant in us_patent_grant.iter("us-bibliographic-data-grant"):
                    for publication_reference in us_patent_grant.iter("publication-reference"):
                        publication_id=publication_reference.find("document-id").find("doc-number").text
                        publication_date=publication_reference.find("document-id").find("date").text
                    title=us_bibliographic_data_grant.find("invention-title").text
                    classifications_ipcr=list()
                    classification_locarno=list()
                    for classifications in us_bibliographic_data_grant.iter("classifications-ipcr"):
                        for classification in classifications.iter("classification-ipcr"):
                            classifications_ipcr.append({
                                "section": classification.find("section").text,
                                "class": classification.find("class").text,
                                "subclass": classification.find("subclass").text,
                                "main-group": classification.find("main-group").text
                                })
                    for classification in us_bibliographic_data_grant.iter("classifications-locarno"):
                        classification_locarno.append({
                            "edition": classification.find("edition").text,
                            "main-classification": classification.find("main-classification").text,
                            "further-classification": classification.find("further-classification").text,
                            "text": classification.find("text").text
                            })
                for abstract in us_patent_grant.iter("abstract"):
                    ab="".join(abstract.itertext())
                claims=list()
                for cls in us_patent_grant.iter("claims"):
                    for claim in cls.iter("claim"):
                        cl="".join(claim.itertext())
                        claims.append(cl)  
                if ab!=None :
                    emb=model.encode(ab)
                    df.loc[len(df)]=[ publication_id, publication_date, application_ref, title, ab, classifications_ipcr, claims, emb]
                temp_file = open('xmls/temp/temp.xml','w')
            temp_file.write(line)   
            n_line+=1
        if not os.path.isdir("csv/"+year):
            os.mkdir("csv/"+year)
        df.to_csv("csv/"+year+"/"+file_name[:-4]+".csv")
        print("\tcsv created.")
        
first=True
BulkDataStorageSystem_url = 'https://bulkdata.uspto.gov/'
BulkDataStorageSystem = requests.get(BulkDataStorageSystem_url)
BulkDataStorageSystem_soup = BeautifulSoup(BulkDataStorageSystem.text, 'html.parser')
for link in BulkDataStorageSystem_soup.find_all('a'):
    if str(link.get_text()).startswith("Patent Grant Full Text Data"):
        print("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
        print("downloading patents from "+str(link.get('href'))[-4:])
        PatentGrantFullTextData=requests.get(link.get('href'))
        PatentGrantFullTextData_soup = BeautifulSoup(PatentGrantFullTextData.text, 'html.parser')
        for table in PatentGrantFullTextData_soup.find_all('table'):
            for zip_downloader in tqdm(table.find_all('a'), total= len(table.find_all('a')), ):
                if str(zip_downloader.get_text()).startswith("ipg"):
                    if first and str(link.get('href'))+"/"+str(zip_downloader.get('href'))!= "https://bulkdata.uspto.gov/data/patent/grant/redbook/fulltext/2019/ipg190910.zip":
                        continue
                    else: first=False
                    print("\n\tdownloading patents from link "+str(link.get('href'))+"/"+str(zip_downloader.get('href')))
                    download=requests.get(str(link.get('href'))+"/"+str(zip_downloader.get('href')))
                    with open("xmls/downloaded_zip/downloaded_file.zip","wb") as downloaded_file:
                        downloaded_file.write(download.content)
                        with zipfile.ZipFile("xmls/downloaded_zip/downloaded_file.zip", 'r') as zip:
                            zip.extract(zip.namelist()[0],"xmls/")
                    print("\tcreating dataframe and calculating embeddings...")
                    create_dataset(zip.namelist()[0], str(link.get('href'))[-4:])
                    os.remove("xmls/"+zip.namelist()[0])
                    os.remove("xmls/downloaded_zip/downloaded_file.zip")