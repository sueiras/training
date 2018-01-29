# Training materials for deep learning

## List of resources and use cases

### Caffe examples
    - Intro to caffe
    - 00_MNIST_linear
    - 01_MNIST_full_connect
    - 02_MNIST_Lenet

### sklearn
scikit learn examples (SKLEARN)
    - Basic classification model
    - Basic Regression model
    - Control overfit and hyperparameter optimization
    - Text classification example

### Tensorflow
    - 00-Intro_to_keras
    - 00-Intro_to_tensorflow
    - 01-tensorboard_example (based on https://gist.github.com/dandelionmane/4f02ab8f1451e276fea1f165a20336f1#file-mnist-py)
    - 02-text
        - 01-char_languaje_model
        - 02-sentiment_model
        - 03-word_tagging
        - 20newsgroups_keras_model
    - 03-image
        - mnist_data_augmentation
        - cifar10_basic_architecture
        - cifar10_resnet
        - transfer_learning examples
        - object_detection
    - 04-time_series
        - Basic example
        - CIF dataset example
    - 05-others


### Tensorflow_old (version 1.0 keras not integrated)
    - 00-Intro_to_tensorflow
    - 01-Template_session
    - 02-template_class
    - 03-text_use_cases
    - 04-image_use_cases
    - 05-others


### Torch examples (work in progress)
    - Torch basics 




## List of deep learning courses

Deep learning course of Google in Udacity
    https://www.udacity.com/course/deep-learning--ud730

Deep learning course of fast.ai
    http://course.fast.ai/

Stanford course Deep learning for Natural languaje processing
    http://web.stanford.edu/class/cs224n/

Stanford course Convolutional Neural Networks for Visual Recognition
    http://cs231n.stanford.edu/

Machine learning course of Stanford in Coursera 
    https://www.coursera.org/learn/machine-learning



# Anaconda environment

1.- Install anaconda3 version 5. All default options.

2.- Start an Anaconda terminal and execute...

```
# Install jupyter extensions 
conda install anaconda-nb-extensions -c nb-conda
```


```
# Create environment and install deep learning packages
conda create -n tf12 python=3.5
activate tf12

conda install graphviz
conda install pandas scikit-learn
conda install -c anaconda jupyter 
conda install matplotlib
conda install pillow 
pip install h5py
pip install pydot-ng
pip install --upgrade https://storage.googleapis.com/tensorflow/windows/cpu/tensorflow-1.2.0-cp35-cp35m-win_amd64.whl
pip install keras
```

