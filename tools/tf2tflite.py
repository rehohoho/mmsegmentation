"""
Usage:
python tools/tf2tflite.py \ 
    --tf_in=work_dirs/ocrnet_hr18_512x1024_20k_labelmefacade/latest.pb \
    --tflite_out=work_dirs/ocrnet_hr18_512x1024_20k_labelmefacade/latest.tflite
"""
import argparse

import tensorflow as tf


parser = argparse.ArgumentParser(description='Convert Tensorflow SavedModel to TFLite')
parser.add_argument('--tf_in', help='checkpoint file', default=None)
parser.add_argument('--tflite_out', help='output file', default=None)
args = parser.parse_args()

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(args.tf_in)
tflite_model = converter.convert()

# Save the model
with open(args.tflite_out, 'wb') as f:
    f.write(tflite_model)

