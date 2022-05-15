"""
Resize supported attributes combination in onnx-tensorflow dependency
 ____________________________________________________________________________________________________________________________________________________
| mode    | coordinate_transformation_mode | cubic_coeff_a | exclude_outside | extrapolation_value | nearest_mode      | scales        | sizes     |
|_________|________________________________|_______________|_________________|_____________________|___________________|_______________|___________|
| nearest | align_corners                  | not apply     | 0               | not apply           | round_prefer_ceil | supported (1) | supported |
|---------|--------------------------------|---------------|-----------------|---------------------|-------------------|---------------|-----------|
| nearest | asymmetric                     | not apply     | 0               | not apply           | floor             | supported (1) | supported |
|---------|--------------------------------|---------------|-----------------|---------------------|-------------------|---------------|-----------|
| nearest | tf_half_pixel_for_nn           | not apply     | 0               | not apply           | floor             | supported (1) | supported |
|---------|--------------------------------|---------------|-----------------|---------------------|-------------------|---------------|-----------|
| linear  | align_corners                  | not apply     | 0               | not apply           | not apply         | supported (1) | supported |
|---------|--------------------------------|---------------|-----------------|---------------------|-------------------|---------------|-----------|
| linear  | asymmetric                     | not apply     | 0               | not apply           | not apply         | supported (1) | supported |
|---------|--------------------------------|---------------|-----------------|---------------------|-------------------|---------------|-----------|
| linear  | half_pixel                     | not apply     | 0               | not apply           | not apply         | supported (1) | supported |
|---------|--------------------------------|---------------|-----------------|---------------------|-------------------|---------------|-----------|
| cubic   | align_corners                  | -0.5          | 1               | not apply           | not apply         | supported (1) | supported |
|---------|--------------------------------|---------------|-----------------|---------------------|-------------------|---------------|-----------|
| cubic   | asymmetric                     | -0.5          | 1               | not apply           | not apply         | supported (1) | supported |
|---------|--------------------------------|---------------|-----------------|---------------------|-------------------|---------------|-----------|
| cubic   | half_pixel                     | -0.5          | 1               | not apply           | not apply         | supported (1) | supported |
|---------|--------------------------------|---------------|-----------------|---------------------|-------------------|---------------|-----------|
| nearest | tf_crop_and_resize             | not apply     | 0               | any float value     | round_prefer_ceil | supported     | supported |
|---------|--------------------------------|---------------|-----------------|---------------------|-------------------|---------------|-----------|
| linear  | tf_crop_and_resize             | not apply     | 0               | any float value     | not apply         | supported     | supported |
|---------|--------------------------------|---------------|-----------------|---------------------|-------------------|---------------|-----------|

Dependencies:
git clone https://github.com/onnx/onnx-tensorflow.git && cd onnx-tensorflow
pip install -e .
pip install tensorflow

Usage:
python tools/onnx2tf.py \ 
    --onnx_in=work_dirs/ocrnet_hr18_512x1024_20k_labelmefacade/latest.onnx \
    --tf_out=work_dirs/ocrnet_hr18_512x1024_20k_labelmefacade/latest.pb
"""

import argparse

from onnx_tf.backend import prepare
import onnx

parser = argparse.ArgumentParser(description='Convert ONNX to Tensorflow SavedModel')
parser.add_argument('--onnx_in', help='checkpoint file', default=None)
parser.add_argument('--tf_out', help='output file', default=None)
args = parser.parse_args()

onnx_model = onnx.load(args.onnx_in)
tf_rep = prepare(onnx_model)
tf_rep.export_graph(args.tf_out)
