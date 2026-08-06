#! /bin/bash

profile="coh_test"

for i in $(seq 1 10);
do
    cxr run $profile -p --nsys
done

cxr performance analyze $profile
