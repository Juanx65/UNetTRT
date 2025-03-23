#!/bin/bash

# Define los modelos y los tamaños de batch size
model_typr="unet"
model="unet.pth"
batch_size=32
log_name="unet_base_pm0_cold"

tegrastats --interval 1 --logfile outputs/tegrastats_log/${log_name}.txt & #sudo tegrastats si necesitas ver mas metricas
tegrastat_pid=$!
python eval.py --weights weights/$model --model $model_typr --dataset datasets/conjunto_experimental/condA --latency --batch_size $batch_size > ${log_name}.txt
kill -9 $tegrastat_pid