#!/bin/bash

# clean the downloaded dataset
cleaner="/home/bagnol/progetti/Scopus/VerifyClusterPatents/dataset_cleaned/clean_dataset.py"
echo '
    Clean dataset'
python $cleaner

# balance cleaned dataset 
bal="/home/bagnol/progetti/Scopus/VerifyClusterPatents/Clustering/balance_dataset.py"
echo '
    Balance dataset'
python $bal


# run python files to obtain csv for clustering evaluation
base_path="/home/bagnol/progetti/Scopus/VerifyClusterPatents/Clustering"
python_files=("generate_csv_sections.py" "generate_csv_classes.py" "generate_csv_subclasses.py")
# Iterate through the list and run each Python script
for file in "${python_files[@]}"; do
    full_path="$base_path/$file"
    echo '
    Running '$file
    if [ ! -f "$full_path" ]; then
        echo "Error: File $full_path not found."
        exit 1
    fi
    if [ "$file" == "generate_csv_sections.py" ]; then
        python $full_path --metric 'euclidean' 
        echo $file ' completed with metric euclidean'
        python $full_path --metric 'cosine'
        echo $file ' completed with metric cosine'
    fi
    if [ "$file" == "generate_csv_classes.py" ]; then
        python $full_path --metric 'euclidean' 
        echo $file ' completed with metric euclidean'
        python $full_path --metric 'cosine'
        echo $file ' completed with metric cosine'
    fi
    if [ "$file" == "generate_csv_subclasses.py" ]; then
        python $full_path --metric 'euclidean' 
        echo $file ' completed with metric euclidean'
        python $full_path --metric 'cosine'
        echo $file ' completed with metric cosine'
    fi
done