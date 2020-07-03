python3 train.py -T 20 -pool 4 --n-params 40000 --name DA_autoaugment --auto-augment 
python3 train.py -T 20 -pool 4 --n-params 40000 --name DA_cutout --cutout 8 
python3 train.py -T 20 -pool 4 --n-params 40000 --name DA_cutmix --cutmix 
python3 train.py -T 20 -pool 4 --n-params 40000 --name DA_mixup --mixup 
