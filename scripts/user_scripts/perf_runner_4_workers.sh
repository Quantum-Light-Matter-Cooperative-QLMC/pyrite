#! /bin/bash

profile="coh_test"

for i in $(seq 1 10);
do
    pyrite run $profile -p --nsys --workers 4 --interval 1
done

pyrite performance analyze $profile
