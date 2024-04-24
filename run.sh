module load tensorflow2-py39-cuda11.2-gcc9/
CUDA_VISIBLE_DEVICES=0 python dynamic_max_relu/main.py --batch-size 128 --n-runs 3 --base-dir /data/sooksatrak/dynamic --drelu-loc end --dataset cifar10 --training-type normal --eps 0.01 --type test --model inceptionv3 --attack-type blackbox
