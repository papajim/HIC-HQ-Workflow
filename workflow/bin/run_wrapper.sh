#!/bin/bash

tar -xzvf Results.tar.gz

cd Results/
python3 ../run_events ../$1 $2


cd ../
tar -czvf Results_$2.tar.gz Results
