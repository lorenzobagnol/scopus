# Brevetti


## WebApp
To run this webapp connect to ssh server via the command "ssh -L 5000:localhost:5000 bagnol@131.114.51.13". Then run the python file webapp.py by this two commands:

'export FLASK_APP=execute_webapp'

'flask run'

### Search for Paper
We call the Scopus search API with some keywords extracted by keyBERT (in practice, we use combinations of keywords), we then calculate embeddings of each results with SentenceTransformers and we return a ranking of them based on similarity.

### Search for Patent
Before proceed with patent seach we have to download all the dataset. The VerifyClusterPatents/XML_DownloaderParser/exec_download.py file will do the job. Anyway there are some difference in xml files across years, so it is necessary to look if fields we are intrested in do not change across years. This file will also calculate embeddings automatically.
Once the dataset has been downloaded, the web app will calculate all the distances between our new patent and all the elements in the dataset and return a ranking.

## Clusters
The code for performing clusters verification is largely the same for both patents and articles. The differences are in classification: for patents we used them ipcr classsification, whereas for articles we used subject division and the scope of the journal in which they were published.
In both cases we have 3 levels of classification.
In both cases we chose only documents with unique classification.
In both cases we had to balance the datasets (with a maximum of 1000 items per classification).
The plots_cluster_patents_paper.ipynb notebook shows results.

### Paper
For papers, the 3 levels of classification are: macroarea, subject, topic.

### Patent
For patents, the 3 levels of classification are: section, class, subclass.
In this case we also had to clean our dataset, before balancing it, by eliminate duplicated patents with same inventors and title and duplicated patents with same abstract.
We have also done an analysis on how abstract differs when documents have the same title and inventors. It can be found in VerifyClusterPatents/hist_for_abstracts.

