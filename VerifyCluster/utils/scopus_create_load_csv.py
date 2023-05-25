import os
import pandas as pd
from pybliometrics.scopus import ScopusSearch

target_directory="./FiveDifferentSubject/"
schema=['eid', 'doi', 'pii', 'pubmed_id', 'title', 'subtype', 'subtypeDescription', 'creator', 'afid', 'affilname', 'affiliation_city', 
        'affiliation_country', 'author_count', 'author_names', 'author_ids', 'author_afids', 'coverDate', 'coverDisplayDate', 'publicationName',
        'issn', 'source_id', 'eIssn', 'aggregationType', 'volume', 'issueIdentifier', 'article_number', 'pageRange', 'description',
        'authkeywords', 'citedby_count', 'openaccess', 'freetoread', 'freetoreadLabel', 'fund_acr', 'fund_no', 'fund_sponsor', 'class']


def create_csv(class_list):
    if not os.path.exists(target_directory):
        os.makedirs(target_directory) 
    
    for query in class_list:
        scopus_query = 'TITLE-ABS-KEY (\"'+query+'\")'
        search = ScopusSearch(scopus_query, verbose=True, download=True, subscriber=True)
        if search.results!=None:
            df = pd.DataFrame(columns=schema)
            for i in range(min(len(search.results),200)):
                    df_doc = pd.DataFrame([search.results[i]._asdict()])
                    df = pd.concat([df, df_doc], join="outer", ignore_index=True)
                    
            df.drop(columns=["doi", "pii", "pubmed_id", "subtypeDescription", "creator", "afid", "affilname", "affiliation_city",
                                    "affiliation_country", "author_count", "author_names", "author_ids", "author_afids", "coverDate", "coverDisplayDate",
                                    "publicationName", "issn", "source_id", "eIssn", "aggregationType", "volume", "issueIdentifier", "article_number",
                                    "pageRange", "citedby_count", "openaccess", "freetoread", "freetoreadLabel", "fund_acr", "fund_no",
                                    "fund_sponsor"] ,inplace=True)
            df.drop(index=df.loc[df["description"].isnull()].index, inplace=True) 
            df["class"] = [query for j in range(len(df.index))]
            df.to_csv(target_directory+query+".csv", sep='\t')


def dataframe_from_csv():
    topic_list=os.listdir(target_directory)
    df=pd.DataFrame()
    for topic in topic_list:
        df_topic=pd.read_csv(target_directory+topic, sep="\t")
        df = pd.concat([df, df_topic], join="outer", ignore_index=True)
    return df