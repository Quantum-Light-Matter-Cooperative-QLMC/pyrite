#! /bin/bash

profiles=(hopg_hbn_gaussian_200fs, hopg_hbn_microtrain_200fs, hopg_hbn_compressed_microbunch)

for p ($profiles) { cxr run --incoherent $p; cxr run --coherent $p }