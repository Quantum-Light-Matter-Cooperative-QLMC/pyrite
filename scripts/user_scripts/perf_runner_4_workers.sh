#! /bin/bash

profile="coh_test"

for i in $(seq 1 10);
do
    cxr run $profile -p --nsys --workers 4 --interval 1
done

cxr performance analyze $profile
