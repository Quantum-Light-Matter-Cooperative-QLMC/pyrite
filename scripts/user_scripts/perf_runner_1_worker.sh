#! /bin/bash

profile="coh_test"

for i in $(seq 1 10);
do
    cxr run $profile -p --nsys --interval 1
done

cxr performance analyze $profile
