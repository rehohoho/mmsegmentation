CONFIG_FILE=configs/ocrnet/ocrnet_hr18_512x1024_40k_labelmefacade.py
CHECKPOINT_FILE=work_dirs/ocrnet_hr18_512x1024_40k_labelmefacade/mlda_d4ddy_bushier_seg.pth
ONNX_FILE=work_dirs/ocrnet_hr18_512x1024_40k_labelmefacade/mlda_d4ddy_bushier_seg.onnx
INPUT_IMG=data/labelmefacade/tests/IMG_4303.jpg
INPUT_SHAPE="3024 4032"
RESCALE_SHAPE="512 384"

python tools/pytorch2onnx.py \
    ${CONFIG_FILE} \
    --checkpoint ${CHECKPOINT_FILE} \
    --output-file ${ONNX_FILE} \
    --input-img ${INPUT_IMG} \
    --shape ${INPUT_SHAPE} \
    --rescale-shape ${RESCALE_SHAPE} \
    --show \
    --verify \
    --dynamic-export \
    --cfg-options \
      model.test_cfg.mode="whole"

MODEL_FILE=$ONNX_FILE
BACKEND=onnxruntime
OUTPUT_FILE=work_dirs/ocrnet_hr18_512x1024_40k_labelmefacade/onnx_out.pkl
EVALUATION_METRICS=mIoU
SHOW_DIRECTORY=work_dirs/ocrnet_hr18_512x1024_40k_labelmefacade/onnx_out_dir
OPACITY=0.5

python tools/deploy_test.py \
    ${CONFIG_FILE} \
    ${MODEL_FILE} \
    --backend ${BACKEND} \
    --out ${OUTPUT_FILE} \
    --eval ${EVALUATION_METRICS} \
    --show \
    --show-dir ${SHOW_DIRECTORY} \
    --opacity ${OPACITY}