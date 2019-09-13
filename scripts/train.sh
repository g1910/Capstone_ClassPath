#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=3 python -m pdb -c continue supervisor_soft/main_soft_multi.py --exp_name sup_multi_soft_b1_o0.001_q0.01_l0.1 --lambda_ortho 0.001 --lambda_quant 0.01 --lambda_l1 0.1
