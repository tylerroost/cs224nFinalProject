#!/bin/bash

if [ "$1" = "train_new_with_trainer" ]; then
	CUDA_VISIBLE_DEVICES=0 python run.py train --cuda --save-to=./models/algebra__linear_1d/LM/trainer/
elif [ "$1" = "train_load_with_trainer" ]; then
	CUDA_VISIBLE_DEVICES=0 python run.py train --cuda --load --load-from=./models/algebra__linear_1d/LM/trainer/checkpoint-125000 --save-to=./models/algebra__linear_1d/LM/trainer/
elif [ "$1" = "train_new_without_trainer" ]; then
	CUDA_VISIBLE_DEVICES=0 python run.py train_without_trainer --cuda --save-to=./models/algebra__linear_1d/LM/no_trainer/
elif [ "$1" = "train_load_without_trainer" ]; then
	CUDA_VISIBLE_DEVICES=0 python run.py train_without_trainer --cuda --load --load-from=./models/algebra__linear_1d/LM/no_trainer/checkpoint-34000 --save-to=./models/algebra__linear_1d/LM/no_trainer/
elif [ "$1" = "inspect" ]; then
	CUDA_VISIBLE_DEVICES=0 python run.py inspect --cuda --load-from=./models/algebra__linear_1d/LM/
elif [ "$1" = "dataset" ]; then
	CUDA_VISIBLE_DEVICES=0 python run.py dataset --cuda
elif [ "$1" = "test" ]; then
    mkdir -p outputs
    touch outputs/test_outputs.txt
    CUDA_VISIBLE_DEVICES=0 python run.py decode --cuda --load-from=./models/algebra__linear_1d/LM/checkpoint-125000
elif [ "$1" = "vocab" ]; then
    python vocab.py vocab.json
else
	echo "Invalid Option Selected"
fi
