python main.py --data_dir '/home/ubuntu/xwy/dataset/MVSA' \
--train_data_dir '/home/ubuntu/xwy/dataset/MVSA' \
--test_data_dir '/home/ubuntu/xwy/dataset/MVSA' \
--gpu 0 \
--save_dir './rebuttal/multiple/n4500' \
--lr 1e-5 \
--num_labels 4500 \
--batch_size 2 \
--num_train_iter 512 \
--threshold 0.95 \
--optim AdamW
# --eval_batch_size 32
# --class_weight [0.2, 0.5, 0.3]
# --resume \
# --load_path 'saved_models-multiple/main/model_best.pth'