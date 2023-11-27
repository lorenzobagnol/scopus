#!/bin/bash
# balance cleaned dataset 
bal="/home/bagnol/progetti/Scopus/VerifyClusterPapers/Clustering/create_balanced_dataset.py"
echo '
    Balance dataset'
python $bal


# run python files to obtain csv for clustering evaluation
python_files=("./Clustering/generate_csv_macroareas.py" "./Clustering/generate_csv_subjects.py" "./Clustering/generate_csv_topics.py")
for file in "${python_files[@]}"; do
    echo '
    Running '$file
    if [ ! -f "$file" ]; then
        echo "Error: File $file not found."
        exit 1
    fi
    if [ "$file" == "./Clustering/generate_csv_macroareas.py" ]; then
        python $file --metric 'euclidean' 
        echo $file ' completed with metric euclidean'
        python $file --metric 'cosine'
        echo $file ' completed with metric cosine'
    fi
    if [ "$file" == "./Clustering/generate_csv_subjects.py" ]; then
        python $file --metric 'euclidean' 
        echo $file ' completed with metric euclidean'
        python $file --metric 'cosine'
        echo $file ' completed with metric cosine'
    fi
    if [ "$file" == "./Clustering/generate_csv_topics.py" ]; then
        python $file --metric 'euclidean' 
        echo $file ' completed with metric euclidean'
        python $file --metric 'cosine'
        echo $file ' completed with metric cosine'
    fi
done