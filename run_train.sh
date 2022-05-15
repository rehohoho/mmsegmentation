export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$(pwd)
CONFIG_FILE=configs/ocrnet/ocrnet_hr18_512x1024_80k_labelmefacade.py
GPU_NUM=1
RESUME=work_dirs/ocrnet_hr18_512x1024_80k_labelmefacade/latest.pth

bash tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM}
  # --resume-from=${RESUME}
# nohup sh run_train.sh &> nohup.out &