# From tensorflow/models/research/
export CODE_PATH=/home/ubuntu/projects/training/tensorflow/models/research

export PYTHONPATH=$PYTHONPATH:$CODE_PATH:$CODE_PATH/slim

export PIPELINE_CONFIG_PATH=object_detection/models/faces/faster_rcnn_resnet101_faces.config
export TRAIN_PATH=object_detection/models/faces/train/model.ckpt-5000

python object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix ${TRAIN_PATH} \
    --output_directory object_detection/models/faces/train/output_inference_graph

