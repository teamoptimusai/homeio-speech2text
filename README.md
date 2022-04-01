# homeio-speech2text

python3 -m torch.distributed.run --nproc_per_node 4 --master_port 9527 train.py --train_file datasets/train.json --valid_file datasets/valid.json --name lsandcv --epochs 20 --params default.config.yaml