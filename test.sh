#!/bin/bash

# Define los modelos y los tamaños de batch size
models=("unet" "attunet")
batch_sizes=(1 2 4 8 16 32)

# Ciclo para cada modelo
for model in "${models[@]}"
do
  # Ciclo para cada tamaño de batch
  for batch_size in "${batch_sizes[@]}"
  do
    echo "Ejecutando modelo $model con batch size $batch_size"
    python eval.py --weights weights/$model.pth --model $model --dataset datasets/img_preprocess --latency --batch_size $batch_size
  done
done

models=("unet_fp16" "attunet_fp16")
# Ciclo para cada modelo
for model in "${models[@]}"
do
  # Ciclo para cada tamaño de batch
  for batch_size in "${batch_sizes[@]}"
  do
    echo "Ejecutando modelo $model con batch size $batch_size"
    python eval.py --weights weights/$model.engine --model tensorrt --dataset datasets/img_preprocess --latency --batch_size $batch_size
  done
done

echo "Proceso completado."