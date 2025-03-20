#!/bin/bash

# Define los modelos y los tamaños de batch size
model_typr="attunet"
model="attunet.pth"
batch_size=32
log_name="attunet_b32_oplvl3_pm2_hot"

tegrastats --interval 1 --logfile outputs/tegrastats_log/${log_name}.txt & #sudo tegrastats si necesitas ver mas metricas
tegrastat_pid=$!
python eval.py --weights weights/$model --model $model_typr --dataset datasets/img_preprocess --latency --batch_size $batch_size
kill -9 $tegrastat_pid