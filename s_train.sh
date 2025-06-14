python main.py --data_dir '/home/ubuntu/xwy/dataset/MVSA_Single' \
--train_data_dir '/home/ubuntu/xwy/dataset/MVSA_Single' \
--test_data_dir '/home/ubuntu/xwy/dataset/MVSA_Single' \
--gpu 1 \
--save_dir 'eccv_result/all_label' \
--lr 1e-5 \
--batch_size 2 \
--num_labels 51 \
--threshold 0.95 \
--num_train_iter 256 \
--optim AdamW \
--add_ulb False \
--use_wandb True