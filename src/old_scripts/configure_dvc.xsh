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

dvc stage add \
    -n prepare_data \
    -p data.processed_dir,data.val_fraction,data.seed,data.count_bucket,data.max_count_bucket,data.min_box_area,data.max_box_area,data.link_images \
    -d src\python\wheat_mlops\config.py \
    -d src\python\wheat_mlops\prepare_data.py \
    -d data\raw \
    -o data\processed \
    python -m wheat_mlops.prepare_data

dvc repro
dvc push
