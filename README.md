# PANet
Please run the following command.  
Example of training PANet on *Wikipedia* dataset:  

python train_link_prediction.py --dataset_name wikipedia --model_name PANet --num_runs 5 --gpu 0 --window_size 100000 --order1_neighbor_nums 50 --order2_neighbor_nums 50 --matrix_on_gpu --dropout 0.1 --patience 20

For other datasets, please use the following commands. Most of the used original dynamic graph datasets come from [Towards Better Dynamic Graph Learning: New Architecture and Unified Library](https://openreview.net/forum?id=xHNzWHbklj), which can be downloaded [here](https://zenodo.org/record/7213796#.Y1cO6y8r30o). 

python train_link_prediction.py --dataset_name *datasets* --model_name PANet --num_runs 5 --gpu 0 --load_best_configs --matrix_on_gpu

