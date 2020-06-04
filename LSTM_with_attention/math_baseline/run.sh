#!/bin/bash

if [ "$1" = "train" ]; then
	CUDA_VISIBLE_DEVICES=0 python run.py train --vocab=vocab.json --cuda --log-every=100
elif [ "$1" = "train_short" ]; then
	CUDA_VISIBLE_DEVICES=0 python run.py train --vocab=vocab.json --cuda --log-every=100 --max-epoch=2
elif [ "$1" = "test" ]; then
    mkdir -p outputs
    touch outputs/test_outputs.txt
    CUDA_VISIBLE_DEVICES=0 python run.py decode model.bin outputs/test_outputs.txt --cuda
elif [ "$1" = "vocab" ]; then
    python vocab.py vocab.json
else
	echo "Invalid Option Selected"
fi
