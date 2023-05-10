Ciao Pietro, ho creato questo branch per te. Tutto il necessario per scaricare i patent si trova nella cartella XML_DownloaderParser.
Lo script che scarica i documenti che ci interessano è exec_download. Facendolo partire, inizierà a scaricare i documenti a partire dal 2020 e indietro negli anni passati. 
In quel periodo (fino al 2016) usano la sempre la stessa versione di dtd: v4.5 dove è presente la classificazione ipcr.
Lo script crea dei csv (uno per ogni file .zip scaricato).
In ogni csv ci sono, in ordine:
  - id : id document
  - date : data documento
  - title
  - abstract
  - ipcr_classifications : una lista di dizionari. Ogni dizionario rappresenta una specifica classificazione del patent. 
                       Dentro ogni dizionario ci dovrebbero essere: section, class, subclass, main-group (in ordine di gerarchia)
  - embedding : fatto con SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
