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

# v0
# for bs in 16 32 64 128 256 512 1024; do 
#     echo "=== Epochs = ${ep}, BatchSize = ${bs}"
#     sleep 15
#     #nohup ./sequential.py --activation relu,relu,sigmoid --neurons 19,200,1 --epochs 100 --batchSize $bs --log -s pdf &
#     #nohup ./sequential.py --activation relu,relu,sigmoid --neurons 19,500,1 --epochs 1000 --batchSize $bs --log -s pdf &
#     #nohup ./sequential.py --activation relu,relu,sigmoid --neurons 19,100,1 --epochs 1000 --batchSize $bs --log -s pdf &
#     nohup ./sequential.py --activation relu,relu,sigmoid --neurons 19,190,1 --epochs 5000 --batchSize $bs --log -s pdf &
#     #nohup ./sequential.py --activation relu,relu,sigmoid --neurons 19,200,1 --epochs 1000 --batchSize $bs --log -s pdf &
#     #nohup ./sequential.py --activation relu,relu,sigmoid --neurons 19,500,1 --epochs 1000 --batchSize $bs --log -s pdf &
# done

# v1
# for n in 19 20 50 100 500 1000 5000; do 
#     echo "=== Neurons = ${n}"
#     sleep 15
#     nohup ./sequential.py --activation relu,relu,sigmoid --neurons 19,$n,1 --epochs 10000 --batchSize 64 --log -s pdf &
# done

# v2
# for bs in 16 64 128 256 1000 50000 100000; do
#     echo "=== BatchSize = ${bs}"
#     sleep 15
#     #nohup ./sequential.py --activation relu,relu,sigmoid --neurons 19,1000,1 --epochs 1000 --batchSize ${bs} --log -s pdf &
#     nohup ./sequential.py --activation relu,relu,sigmoid --neurons 19,1000,1 --epochs 1000 --batchSize ${bs} --log -s pdf &
# done

# v3
# for bs in 40 50 60 64 70 80 90 100 ; do
#     echo "=== BatchSize = ${bs}"
#     sleep 15
#     nohup ./sequential.py --activation relu,relu,sigmoid --neurons 19,1000,1 --epochs 1000 --batchSize ${bs} --log -s pdf &
# done

# v4
# for bs in 1 5 10 20 40; do
#     echo "=== BatchSize = ${bs}"
#     sleep 15
#     nohup ./sequential.py --activation relu,relu,sigmoid --neurons 19,1000,1 --epochs 1000 --batchSize ${bs} --log -s pdf &
# done

# v5
nohup ./sequential.py --activation relu,relu,sigmoid --neurons 19,500,1 --epochs 1000 --batchSize 32 --log -s pdf &
sleep 15
nohup ./sequential.py --activation relu,relu,relu,sigmoid --neurons 19,500,250,1 --epochs 1000 --batchSize 32 --log -s pdf &
sleep 15
nohup ./sequential.py --activation relu,relu,relu,relu,sigmoid --neurons 19,500,250,125,1 --epochs 1000 --batchSize 32 --log -s pdf &
sleep 15
nohup ./sequential.py --activation relu,relu,relu,relu,relu,sigmoid --neurons 19,500,250,125,60,1 --epochs 1000 --batchSize 32 --log -s pdf &
sleep 15
nohup ./sequential.py --activation relu,relu,relu,relu,relu,relu,sigmoid --neurons 19,500,250,125,60,30,1 --epochs 1000 --batchSize 32 --log -s pdf &
sleep 15
nohup ./sequential.py --activation relu,relu,relu,relu,relu,relu,relu,sigmoid --neurons 19,500,250,125,60,30,19,1 --epochs 1000 --batchSize 32 --log -s pdf &

# v6
# for n in 19 38 57 95 190 1900; do
#     echo "=== Neurons = ${n}"
#     sleep 15
#     nohup ./sequential.py --activation relu,relu,sigmoid --neurons 19,${n},1 --epochs 1000 --batchSize 32 --log -s pdf &
# done

# v7
# for bs in 16 32 64 128 256; do 
#     echo "=== BatchSize = ${bs}"
#     sleep 15
#     nohup ./sequential.py --activation relu,relu,relu,relu,relu,relu,relu,sigmoid --neurons 19,500,250,125,60,30,19,1 --epochs 1000 --batchSize ${bs} --log -s pdf &
# done

# v8
#nohup ./sequential.py --activation tanh,tanh,tanh,tanh,tanh,tanh,sigmoid --neurons 19,500,250,125,60,30,1 --epochs 1000 --batchSize 32 --log -s pdf &

# v9
# nohup ./sequential.py --activation relu,relu,relu,relu,relu,relu,sigmoid --neurons 19,500,250,125,60,30,1 --epochs 1000 --batchSize 32 --log -s pdf --lossFunction "mean_squared_error" &
# sleep 20
# nohup ./sequential.py --activation relu,relu,relu,relu,relu,relu,sigmoid --neurons 19,500,250,125,60,30,1 --epochs 1000 --batchSize 32 --log -s pdf --lossFunction "mean_absolute_error" &
# sleep 20
# nohup ./sequential.py --activation relu,relu,relu,relu,relu,relu,sigmoid --neurons 19,500,250,125,60,30,1 --epochs 1000 --batchSize 32 --log -s pdf --lossFunction "mean_absolute_percentage_error" &
# sleep 20
# nohup ./sequential.py --activation relu,relu,relu,relu,relu,relu,sigmoid --neurons 19,500,250,125,60,30,1 --epochs 1000 --batchSize 32 --log -s pdf --lossFunction "mean_squared_logarithmic_error" &
# sleep 20
# nohup ./sequential.py --activation relu,relu,relu,relu,relu,relu,sigmoid --neurons 19,500,250,125,60,30,1 --epochs 1000 --batchSize 32 --log -s pdf --lossFunction "squared_hinge" &
# sleep 20
# nohup ./sequential.py --activation relu,relu,relu,relu,relu,relu,sigmoid --neurons 19,500,250,125,60,30,1 --epochs 1000 --batchSize 32 --log -s pdf --lossFunction "hinge" &
# sleep 20
# nohup ./sequential.py --activation relu,relu,relu,relu,relu,relu,sigmoid --neurons 19,500,250,125,60,30,1 --epochs 1000 --batchSize 32 --log -s pdf --lossFunction "categorical_hinge" &
# sleep 20
# nohup ./sequential.py --activation relu,relu,relu,relu,relu,relu,sigmoid --neurons 19,500,250,125,60,30,1 --epochs 1000 --batchSize 32 --log -s pdf --lossFunction "logcosh" &
# sleep 20
# nohup ./sequential.py --activation relu,relu,relu,relu,relu,relu,sigmoid --neurons 19,500,250,125,60,30,1 --epochs 1000 --batchSize 32 --log -s pdf --lossFunction "huber_loss" & 
# sleep 20
# nohup ./sequential.py --activation relu,relu,relu,relu,relu,relu,sigmoid --neurons 19,500,250,125,60,30,1 --epochs 1000 --batchSize 32 --log -s pdf --lossFunction "categorical_crossentropy" & 
# sleep 20
# nohup ./sequential.py --activation relu,relu,relu,relu,relu,relu,sigmoid --neurons 19,500,250,125,60,30,1 --epochs 1000 --batchSize 32 --log -s pdf --lossFunction "sparse_categorical_crossentropy" & 
# sleep 20
# nohup ./sequential.py --activation relu,relu,relu,relu,relu,relu,sigmoid --neurons 19,500,250,125,60,30,1 --epochs 1000 --batchSize 32 --log -s pdf --lossFunction "binary_crossentropy" &
# sleep 20
# nohup ./sequential.py --activation relu,relu,relu,relu,relu,relu,sigmoid --neurons 19,500,250,125,60,30,1 --epochs 1000 --batchSize 32 --log -s pdf --lossFunction "kullback_leibler_divergenc" &
# sleep 20
# nohup ./sequential.py --activation relu,relu,relu,relu,relu,relu,sigmoid --neurons 19,500,250,125,60,30,1 --epochs 1000 --batchSize 32 --log -s pdf --lossFunction "poisson" "cosine_proximity" &
# sleep 20
# nohup ./sequential.py --activation relu,relu,relu,relu,relu,relu,sigmoid --neurons 19,500,250,125,60,30,1 --epochs 1000 --batchSize 32 --log -s pdf --lossFunction "is_categorical_crossentropy" &
