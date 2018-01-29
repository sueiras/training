export CODE_PATH=/home/ubuntu/projects/training/tensorflow/models/research

export PYTHONPATH=$PYTHONPATH:$CODE_PATH:$CODE_PATH/slim
export CUDA_DEVICE_ORDER="PCI_BUS_ID" 
export CUDA_VISIBLE_DEVICES=2

python object_detection/eval.py \
    --logtostderr \
    --pipeline_config_path=$CODE_PATH/object_detection/models/faces/faster_rcnn_resnet101_faces.config \
    --checkpoint_dir=$CODE_PATH/object_detection/models/faces/train \
    --eval_dir=$CODE_PATH/object_detection/models/faces/eval