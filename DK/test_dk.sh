### test species classification
python main_dk.py -e 'alexnet_valid' --model='alexnet' --data_type='species' --CUDA_VISIBLE_DEVICES='3' --cuda --use_pretrain --checkpoint='./alexnet_species/checkpoint_step_29.pth'
#python main_dk.py -e 'googlenet_valid' --model='googlenet' --data_type='species' --CUDA_VISIBLE_DEVICES='3' --cuda --use_pretrain --checkpoint='./googlenet_species/checkpoint_step_29.pth'
#python main_dk.py -e 'mobilenet_valid' --model='mobilenet' --data_type='species' --CUDA_VISIBLE_DEVICES='3' --cuda --use_pretrain --checkpoint='./mobilenet_species/checkpoint_step_29.pth'
#python main_dk.py -e 'vgg19_valid' --model='vgg19' --data_type='species' --CUDA_VISIBLE_DEVICES='3' --cuda --use_pretrain --checkpoint='./vgg19_species/checkpoint_step_25.pth'
#python main_dk.py -e 'resnet18_valid' --model='resnet18' --data_type='species' --CUDA_VISIBLE_DEVICES='3' --cuda --use_pretrain --checkpoint='./resnet18_species/checkpoint_step_29.pth'

### test breeds classification
#python main_dk.py -e 'alexnet_valid' --model='alexnet' --data_type='breeds' --CUDA_VISIBLE_DEVICES='3' --cuda --use_pretrain --checkpoint='./alexnet_breed/checkpoint_step_15.pth'
#python main_dk.py -e 'googlenet_valid' --model='googlenet' --data_type='breeds' --CUDA_VISIBLE_DEVICES='3' --cuda --use_pretrain --checkpoint='./googlenet_breed/checkpoint_step_10.pth'
#python main_dk.py -e 'mobilenet_valid' --model='mobilenet' --data_type='breeds' --CUDA_VISIBLE_DEVICES='3' --cuda --use_pretrain --checkpoint='./mobilenet_breed/checkpoint_step_20.pth'
#python main_dk.py -e 'vgg19_valid' --model='vgg19' --data_type='breeds' --CUDA_VISIBLE_DEVICES='3' --cuda --use_pretrain --checkpoint='./vgg19_breed/checkpoint_step_10.pth'
#python main_dk.py -e 'resnet18_valid' --model='resnet18' --data_type='breeds' --CUDA_VISIBLE_DEVICES='3' --cuda --use_pretrain --checkpoint='./resnet18_breed/checkpoint_step_20.pth'
