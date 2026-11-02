#!/usr/bin/env bash
# run_collection_rest.sh: The rest of the dataset after run_collection.sh was stopped part-way through showcase_forest …
D=~/rs1_18_ws/eval/yolo
T=$(dirname "$0")
bash $T/stop_gz.sh
WORLD=showcase_forest bash $T/collect.sh $D/ds_showcase --scenes 250 --views 8 --seed 3 --prefix sb
bash $T/stop_gz.sh
WORLD=dense_forest bash $T/collect.sh $D/ds_dense --scenes 200 --views 8 --seed 2 --prefix de
bash $T/stop_gz.sh
WORLD=simple_trees bash $T/collect.sh $D/ds_test_simple --scenes 60 --views 8 --seed 9 --prefix te --ground 21 --val_every 1
bash $T/stop_gz.sh
echo COLLECTION_DONE
