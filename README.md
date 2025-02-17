# PANet
python train_link_prediction.py \
  --dataset_name <dataset_name> \
  --model_name DyConNet \
  --num_runs 1 \
  --gpu 0 \
  --window_size 100000 \
  --order1_neighbor_nums 50 \
  --order2_neighbor_nums 50 \
  --matrix_on_gpu \
  --dropout 0.1 \
  --patience 20
