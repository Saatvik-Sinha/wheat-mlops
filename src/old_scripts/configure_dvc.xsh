dvc init

dvc remote add -d my_local_remote ..\dvc-storage

mkdir data\raw\

dvc stage add \
    -n download_data \
    -p data.competition,data.raw_dir \
    -d src\python\wheat_mlops\config.py \
    -d src\python\wheat_mlops\download_data.py \
    -o data\raw \
    python -m wheat_mlops.download_data

dvc repro
dvc push
