export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=$(pwd)
CONFIG_FILE=configs/ocrnet/ocrnet_hr18_512x1024_40k_labelmefacade.py
GPU_NUM=2
RESUME=work_dirs/ocrnet_hr18_512x1024_40k_labelmefacade/latest.pth

./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM}
  # --resume-from=${RESUME}