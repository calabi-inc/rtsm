import os
import numpy as np
from PIL import Image

input_dir = './test_dataset/depth/'
output_dir = './test_dataset/depth_npy/'

os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if fname.lower().endswith('.png'):
        in_path = os.path.join(input_dir, fname)
        out_fname = os.path.splitext(fname)[0] + '.npy'
        out_path = os.path.join(output_dir, out_fname)
        # Load PNG as grayscale (depth)
        img = Image.open(in_path)
        arr = np.array(img)
        np.save(out_path, arr)
