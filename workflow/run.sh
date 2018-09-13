#!/usr/bin/env bash

export SINGULARITY_PULLFOLDER=${PWD}/bin

singularity pull docker://papajim/hic_hq

./daxgen.py hic.dax
./plan.sh hic.dax

