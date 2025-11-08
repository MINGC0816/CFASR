CUDA_VISIBLE_DEVICES=0  python main.py --arch="cfasr" --test_model="CRNN" --batch_size=48 --STN  --gradient  --use_distill --stu_iter=3 --vis --vis_dir='test' --mask --go_test --resume='ckpt/' --triple_clues --text_focus --vis


# resume just only needs to be filled in the folder such as 'ckpt/rstn-1/'
# multi-stage need remove --sr_share
# multi-stage --stu_iter=x   x means how many stage you want to train

