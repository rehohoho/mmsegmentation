"""
Serving class that consumes images and outputs segmentation bitmap.
"""

import numpy as np
import torch
import mmcv

from mmcv.parallel import collate
from mmseg.datasets.pipelines import Compose
from mmseg.apis.inference import init_segmentor, inference_segmentor, LoadImage
from mmseg.ops import resize
from deploy_test import TfliteSegmentor


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

class TfliteModelServer:
    def __init__(self, config_file: str, checkpoint_file: str):
        self.cfg = mmcv.Config.fromfile(config_file)
        self.model = TfliteSegmentor(checkpoint_file, cfg=self.cfg, device_id=0)
    
    def infer(self, img: np.ndarray, opacity: float = 0.5):
        test_pipeline = [LoadImage()] + self.cfg.data.test.pipeline[1:]
        test_pipeline = Compose(test_pipeline)
        data = dict(img=img)
        data = test_pipeline(data)
        data_img = collate([data['img']], samples_per_gpu=1)[0]
        data_img = resize(data_img, size=(384, 512), mode='bilinear',align_corners=True, warning=False)
        seg_pred = self.model.simple_test(data_img, data['img_metas'][0], is_resize=False)
        
        seg_pred = torch.from_numpy(seg_pred[0]).float().unsqueeze(0).unsqueeze(0)
        seg_pred = resize(seg_pred, size=tuple(img.shape[:2]), mode='nearest').squeeze().squeeze()
        seg = seg_pred.long().detach().cpu().numpy()
        return seg

    def save_visualise(self, img: np.ndarray, seg: np.ndarray, out_file: str, opacity: float = 0.5):
        palette = np.array(self.cfg.PALETTE)
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        color_seg = color_seg[..., ::-1]

        vis = img * (1 - opacity) + color_seg * opacity
        vis = vis.astype(np.uint8)
        mmcv.imwrite(vis, out_file)

# for testing only
if __name__ == '__main__':

  IMG_FILE='data/labelmefacade/tests/Picture2.png'
  CONFIG_FILE='configs/ocrnet/ocrnet_hr18_512x1024_80k_labelmefacade.py'
  CHECKPOINT_FILE='../bushierbrows.tflite'
  
  if CHECKPOINT_FILE.endswith('tflite'):
    server = TfliteModelServer(config_file=CONFIG_FILE, checkpoint_file=CHECKPOINT_FILE)
  else:
    server = LocalModelServer(config_file=CONFIG_FILE, checkpoint_file=CHECKPOINT_FILE)  
  img = mmcv.imread(IMG_FILE)
  result = server.infer(img)
  print(result)
  out_file = IMG_FILE[:-4] + '_result.jpg'
  server.save_visualise(img, result, out_file)
  mmcv.imwrite(result[0], IMG_FILE[:-4] + '_result_bitmap.jpg')