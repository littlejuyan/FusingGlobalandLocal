export CUDA_VISIBLE_DEVICES=0,1,2,3;
python train.py --name p_trained --blur_prob 0.1 --blur_sig 0.0,3.0 --jpg_prob 0.1 --jpg_method cv2,pil --jpg_qual 30,100 --dataroot xxx/xxx --batch_size 128 --lr 0.00005 --gpu_ids 0,1,2,3

