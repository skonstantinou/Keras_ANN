#!/bin/sh   

for ep in 1000 10000 100000 1000000
do
    for bs in 100 1000 10000 100000
    do
	echo "=== Epochs = ${ep}, BatchSize = ${bs}"
	#./sequential.py --activation relu,relu,sigmoid --neurons 36,19,1 --epochs $ep --batchSize $bs --log -s pdf
	#./sequential.py --activation relu,relu,sigmoid --neurons 36,19,1 --epochs $ep --batchSize $bs --log -s pdf & 
	./sequential.py --activation relu,relu,relu,sigmoid --neurons 36,25,19,1 --epochs $ep --batchSize $bs --log -s pdf & 
    done

done
            
