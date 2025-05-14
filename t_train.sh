# python main.py --data_dir '/home/ubuntu/xwy/dataset/MVSA' \
# --train_data_dir '/home/ubuntu/xwy/dataset/MVSA' \
# --test_data_dir '/home/ubuntu/xwy/dataset/MVSA' \
# --gpu 0 \
# --save_dir './rebuttal/multiple/n4500' \
# --lr 1e-5 \
# --num_labels 4500 \
# --batch_size 2 \
# --num_train_iter 512 \
# --threshold 0.95 \
# --optim AdamW

python main.py --data_dir '/home/ubuntu/xwy/dataset/twitter' \
--train_data_dir '/home/ubuntu/xwy/dataset/twitter' \
--test_data_dir '/home/ubuntu/xwy/dataset/twitter' \
--gpu 0 \
--save_dir 'eccv_result/twitter/900/add' \
--lr 1e-5 \
--batch_size 2 \
--num_labels 900 \
--threshold 0.95 \
--num_train_iter 256 \
--dataset twitter \
--optim AdamW \
--add_ulb True