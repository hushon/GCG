#!/bin/bash
torchrun --nproc_per_node=1 --master_port=1111 finetune_ans.py --save_ckpt True