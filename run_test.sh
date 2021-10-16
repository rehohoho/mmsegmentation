export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$(pwd)
CONFIG_FILE=configs/ocrnet/ocrnet_hr18_512x1024_40k_labelmefacade.py
GPU_NUM=1
CHECKPOINT=work_dirs/ocrnet_hr18_512x1024_40k_labelmefacade/latest.pth
OUTPUT=work_dirs/ocrnet_hr18_512x1024_40k_ntuwalls_results

python tools/test.py ${CONFIG_FILE} ${CHECKPOINT} \
  --show-dir ${OUTPUT}