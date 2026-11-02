#!/usr/bin/env bash
# run_collection.sh: The full dataset (train worlds + an unseen test world)
D=~/rs1_18_ws/eval/yolo
T=$(dirname "$0")
stop_gz() { bash $T/stop_gz.sh; }
stop_gz
WORLD=showcase_forest bash $T/collect.sh $D/ds_showcase --scenes 400 --views 8 --seed 1 --prefix sc
stop_gz
WORLD=dense_forest bash $T/collect.sh $D/ds_dense --scenes 200 --views 8 --seed 2 --prefix de
stop_gz
WORLD=simple_trees bash $T/collect.sh $D/ds_test_simple --scenes 60 --views 8 --seed 9 --prefix te --ground 21 --val_every 1
stop_gz
echo COLLECTION_DONE
