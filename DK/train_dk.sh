### train cmd
python main_dk.py -e alexnet_breed --is_train --model='alexnet' --n_class=2 --CUDA_VISIBLE_DEVICES='3' --cuda --save_step=5 --lr=1e-5 --lr_decay_period=10 --max_step=30 --use_pretrain

### train with valid cmd
#python main_dk.py -e alexnet_breed --is_train --model='alexnet' --n_class=2 --CUDA_VISIBLE_DEVICES='3' --cuda --save_step=5 --lr=1e-5 --lr_decay_period=10 --max_step=30 --use_pretrain --use_valid

