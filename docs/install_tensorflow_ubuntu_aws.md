# Instructions to install tensorflow with cuda in ubuntu in AWS


```
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install -y build-essential git libfreetype6-dev libxft-dev libncurses-dev gfortran linux-headers-generic linux-image-extra-virtual unzip swig unzip wget pkg-config zip g++ zlib1g-dev libcurl3-dev

sudo apt-get remove unattended-upgrades
```



## Anaconda
```
mkdir soft
cd soft
wget https://repo.anaconda.com/archive/Anaconda3-5.1.0-Linux-x86_64.sh
bash ./Anaconda3-5.1.0-Linux-x86_64.sh
# all defaults
```


## Nvidia driver
```
sudo apt-get purge nvidia*
sudo add-apt-repository ppa:graphics-drivers
sudo apt-get update
sudo apt-get install nvidia-390
```


## Cuda
```
wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64-deb
sudo apt-key add /var/cuda-repo-<version>/7fa2af80.pub
sudo dpkg -i cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64-deb
sudo apt-get update
sudo apt-get install cuda
```


## Change the ~/.bashrc file 
```
# add at the end:
export CUDA_HOME=/usr/local/cuda
export PATH=$PATH:$CUDA_/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64
```

## Reload or open a new terminal
```
source ./.bashrc
```


## CUDNN
```
# - download the runtime 6.0 for ubuntu 16.04 file manually from nvidia website
sudo dpkg -i libcudnn7_7.1.2.21-1+cuda9.0_amd64.deb
```




## Install tensorflow
```
conda update -n base conda

conda install nb_conda

conda create -n tf18 python=3.5
source activate tf18

conda install graphviz
conda install pandas scikit-learn
conda install jupyter
conda install matplotlib
conda install pillow 
conda install lxml
pip install Cython
pip install pydot-ng
pip install h5py

pip install tensorflow-gpu==1.8
```


## Start jupyter at the beggining
```
export EDITOR=nano
crontab -e

# Add at the end of the file
@reboot cd ~/training; source ~/.bashrc; ~/anaconda3/bin/jupyter notebook >>~/cronrun.log 2>&1
```


## Configure jupyter
```
# Create config
jupyter notebook --generate-config

# Add pasword
jupyter notebook password

# generate certs
openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout mykey.key -out mycert.pem
```



### Edit the config file
```
nano /home/ubuntu/.jupyter/jupyter_notebook_config.py

# Edit...
# Set options for certfile, ip, password, and toggle off
# browser auto-opening

c.NotebookApp.certfile = u'/home/ubuntu/mycert.pem'
c.NotebookApp.keyfile = u'/home/ubuntu/mykey.key'

# Set ip to '*' to bind on all interfaces (ips) for the public server
c.NotebookApp.ip = '*'
c.NotebookApp.open_browser = False
```



