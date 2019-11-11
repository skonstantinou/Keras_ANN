#!/bin/sh   

# for ep in 100; do #500; do
#     for bs in 50 100 200 500 1000 5000; do 
# 	echo "=== Epochs = ${ep}, BatchSize = ${bs}"
# 	# nohup ./sequential.py --activation relu,relu,relu,sigmoid --neurons 64,32,16,1 --epochs $ep --batchSize $bs --log -s pdf & # does not work
# 	# nohup ./sequential.py --activation relu,relu,sigmoid --neurons 32,16,1 --epochs $ep --batchSize $bs --log -s pdf & # ~works
# 	# nohup ./sequential.py --activation relu,relu,sigmoid --neurons 128,64,1 --epochs $ep --batchSize $bs --log -s pdf & # works!
# 	# nohup ./sequential.py --activation relu,relu,sigmoid --neurons 512,256,1 --epochs $ep --batchSize $bs --log -s pdf & # does not work
# 	# nohup ./sequential.py --activation relu,relu,sigmoid --neurons 19,190,1 --epochs $ep --batchSize $bs --log -s pdf & #works
# 	
# 	nohup ./sequential.py --activation relu,relu,sigmoid --neurons 19,190,1 --epochs $ep --batchSize $bs --log -s pdf &
# 	sleep 10
#     done
# done
# 
# sleep 10

#for bs in 100 200 500 1000; do 
for bs in 16 32 64 128 256 512 1024; do 
    echo "=== Epochs = ${ep}, BatchSize = ${bs}"
    sleep 15
    #nohup ./sequential.py --activation relu,relu,sigmoid --neurons 19,200,1 --epochs 100 --batchSize $bs --log -s pdf &
    #nohup ./sequential.py --activation relu,relu,sigmoid --neurons 19,500,1 --epochs 1000 --batchSize $bs --log -s pdf &
    nohup ./sequential.py --activation relu,relu,sigmoid --neurons 19,100,1 --epochs 1000 --batchSize $bs --log -s pdf &

    #nohup ./sequential.py --activation relu,relu,sigmoid --neurons 19,500,1 --epochs 100 --batchSize $bs --log -s pdf &
    #nohup ./sequential.py --activation elu,elu,sigmoid --neurons 19,500,1 --epochs 100 --batchSize $bs --log -s pdf &
done
