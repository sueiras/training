# Instructions

To run this use case:

- Install the object detection API of Tensorflow.
    - Follow the instructions [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)
    - Clone the repo
        ```
        git clone https://github.com/tensorflow/models.git
        ```
    - Download the pretrained modelfaster_rcnn_resnet101_coco 
        ```
        wget http://storage.googleapis.com/download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz
        tar -xvf faster_rcnn_resnet101_coco_11_06_2017.tar.gz
        ```
    
    
- Create in the object_detection dir the next structure and copy the files
```
<reseach>
    train_faces.sh
    eval_faces.sh
    export_model_faces.sh
    <object_detection>
        <data>
            faces_label_map.pbtxt
            <faces>
        <models>
            <faces>
                faster_rcnn_resnet101_faces.config
                <train>
                <eval>
```

- Convert the database to TFRecords whit the notebook Convert_annotated_images_into_TFRecords.ipynb


- Execute the train_faces.sh to train the model.
- Simultaneously execute eval_faces.sh to stat the evaluatoin proccess. 
- Start the tensorboard over the models dir to follow the 