# 1. Download pretrained weights
python3 download_pretrained_superpoint.py

# 2. Generate UAV data  
python3 generate_uav_data.py --earth_imagery_dir data/earth_imagery

# 3. Train on UAV data
python3 train_superpoint_uav.py --data_dir uav_data --pretrained_weights pretrained_weights/superpoint_v1.pth