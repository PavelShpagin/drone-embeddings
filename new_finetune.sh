python train_superpoint_uav.py \
  --data_dir uav_data/ \
  --pretrained_weights superpoint_v1.pth \
  --output_dir superpoint_uav_trained \
  --epochs 20 \
  --batch_size 4 \
  --lr 1e-4