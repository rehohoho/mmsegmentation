"""
Serving class that consumes images and outputs segmentation bitmap.
"""

import numpy as np
import mmcv

from mmseg.apis.inference import init_segmentor, inference_segmentor


class LocalModelServer:
  """
  Encapsulates inference methods to serve model.
  Stores model to call inference on images.

  See test at main for use example.
  """
  
  def __init__(self, config_file: str, checkpoint_file: str):
    self.model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

  def infer(self, img: np.ndarray):
    return inference_segmentor(self.model, img)
  
  def save_visualise(self, img: np.ndarray, result: np.ndarray, out_file: str):
    self.model.show_result(img, result, out_file=out_file, opacity=0.5)
  
  def get_visualise(self, img: np.ndarray, result: np.ndarray):
    return self.model.show_result(img, result, opacity=0.5)


# for testing only
if __name__ == '__main__':

  IMG_FILE='data/labelme_facade/tests/20211016_150104.jpg'
  CONFIG_FILE='configs/ocrnet/ocrnet_hr18_512x1024_40k_labelmefacade.py'
  CHECKPOINT_FILE='work_dirs/ocrnet_hr18_512x1024_40k_labelmefacade/latest.pth'
  
  server = LocalModelServer(config_file=CONFIG_FILE, checkpoint_file=CHECKPOINT_FILE)
  img = mmcv.imread(IMG_FILE)
  result = server.infer(img)
  out_file = IMG_FILE[:-4] + '_result.jpg'
  server.save_visualise(img, result, out_file)
  # mmcv.imwrite(result[0], IMG_FILE[:-4] + '_result_bitmap.jpg')