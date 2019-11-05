#!/bin/sh   

for ep in 100000 1000000
do
    for bs in 1 10 100 1000 10000 100000
    do
	echo "=== Epochs = ${ep}, BatchSize = ${bs}"
	nohup ./sequential.py --activation relu,relu,relu,relu,relu,sigmoid --neurons 1024,512,256,128,64,1 --epochs $ep --batchSize $bs --log -s pdf &
	nohup ./sequential.py --activation relu,relu,relu,relu,relu,sigmoid --neurons 256,128,64,32,16,1 --epochs $ep --batchSize $bs --log -s pdf &
	nohup ./sequential.py --activation relu,relu,relu,relu,sigmoid --neurons 128,64,32,16,1 --epochs $ep --batchSize $bs --log -s pdf &
	nohup ./sequential.py --activation relu,relu,relu,sigmoid --neurons 64,32,16,1 --epochs $ep --batchSize $bs --log -s pdf &
	nohup ./sequential.py --activation relu,relu,sigmoid --neurons 128,64,1 --epochs $ep --batchSize $bs --log -s pdf &
	nohup ./sequential.py --activation relu,relu,sigmoid --neurons 512,256,1 --epochs $ep --batchSize $bs --log -s pdf &
	nohup ./sequential.py --activation relu,relu,sigmoid --neurons 1024,512,1 --epochs $ep --batchSize $bs --log -s pdf &
    done

done


# for ep in 1000 10000 100000 1000000
# do
#     for bs in 100 1000 10000 #100000
#     do
# 	echo "=== Epochs = ${ep}, BatchSize = ${bs}"
# 	#./sequential.py --activation relu,relu,sigmoid --neurons 36,19,1 --epochs $ep --batchSize $bs --log -s pdf
# 	#./sequential.py --activation relu,relu,sigmoid --neurons 36,19,1 --epochs $ep --batchSize $bs --log -s pdf & 
# 	#./sequential.py --activation relu,relu,relu,sigmoid --neurons 36,25,19,1 --epochs $ep --batchSize $bs --log -s pdf & 
# 	nohup ./sequential.py --activation relu,relu,relu,relu,sigmoid --neurons 36,25,22,19,1 --epochs $ep --batchSize $bs --log -s pdf &
#     done
# 
# done
            
