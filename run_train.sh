export PYTHONPATH=$(pwd)
CONFIG_FILE=configs/ocrnet/ocrnet_hr18_512x1024_40k_labelmefacade.py
GPU_NUM=1
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM}