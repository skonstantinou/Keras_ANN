#!/bin/sh   

for x in 100 1000 10000 100000 1000000
do
    #./sequential.py --activation relu,relu,sigmoid --neurons 36,19,1 --epochs 10000 --batchSize $x -s pdf >& res_batchSize_$x.txt &
    ./sequential.py --activation relu,relu,sigmoid --neurons 36,19,1 --epochs 1000000 --batchSize $x -s pdf >& res_batchSize_$x.txt &
done
