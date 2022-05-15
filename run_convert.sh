CONFIG_FILE=configs/ocrnet/ocrnet_hr18_512x1024_20k_labelmefacade.py
WORK_DIR=work_dirs/ocrnet_hr18_512x1024_20k_labelmefacade

if [ ! -f ${WORK_DIR}/latest.pth ]; then
    printf "Unable to find pytorch checkpoint ${WORK_DIR}/latest.pth"
    exit 1
fi

if [ ! -f ${WORK_DIR}/latest.onnx ]; then
    printf "GENERATING ONNX \n\n\n"

    INPUT_IMG=data/labelmefacade/tests/IMG_4303.jpg
    INPUT_SHAPE="384 512"
    RESCALE_SHAPE="384 512"

    python tools/pytorch2onnx.py \
        ${CONFIG_FILE} \
        --checkpoint ${WORK_DIR}/latest.pth \
        --output-file ${WORK_DIR}/latest.onnx \
        --input-img ${INPUT_IMG} \
        --shape ${INPUT_SHAPE} \
        --show \
        --verify \
        --cfg-options \
        model.test_cfg.mode="whole"
fi

if false && [ ! -f ${WORK_DIR}/onnx_out.pkl ]; then
    printf "TESTING ONNX \n\n\n"

    MODEL_FILE=${WORK_DIR}/latest.onnx
    BACKEND=onnxruntime
    OUTPUT_FILE=${WORK_DIR}/onnx_out.pkl
    EVALUATION_METRICS=mIoU
    SHOW_DIRECTORY=${WORK_DIR}/onnx_out_dir
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
fi

if [ ! -d ${WORK_DIR}/latest.pb ]; then
    printf "GENERATING PB \n\n\n"
    python tools/onnx2tf.py --onnx_in ${WORK_DIR}/latest.onnx --tf_out ${WORK_DIR}/latest.pb
fi

if [ ! -f ${WORK_DIR}/latest.tflite ]; then
    printf "GENERATING TFLITE \n\n\n"
    python tools/tf2tflite.py --tf_in ${WORK_DIR}/latest.pb --tflite_out ${WORK_DIR}/latest.tflite
fi

if [ ! -f ${WORK_DIR}/tflite_out.pkl ]; then
    printf "TESTING TFLITE \n\n\n"

    MODEL_FILE=${WORK_DIR}/latest.tflite
    BACKEND=tflite
    OUTPUT_FILE=${WORK_DIR}/tflite_out.pkl
    EVALUATION_METRICS=mIoU
    SHOW_DIRECTORY=${WORK_DIR}/tflite_out_dir
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
fi