## Class template

Standar usage of TensoFlow with model class. Tipically use 3 files:
 - data_utils.py: With the data access and batch generator functions
 - model_name.py: With the class model. A constructor with the graph definition and methods to manage the model needs
 - train.py: With the parameters, access to the data, instance the model and train it.
 

## Training

Print parameters:

```bash
./train.py --help
```

```
optional arguments:
  -h, --help            show this help message and exit
  --data_directory DATA_DIRECTORY
                        Data dir (default /tmp/MNIST_data)
  --dense_size DENSE_SIZE
                        dense_size (default 500)
  --learning_rate LEARNING_RATE
                        learning rate (default: 0.001)
  --batch_size BATCH_SIZE
                        Batch Size (default: 256)
  --num_epochs NUM_EPOCHS
                        Number of training epochs (default: 20)
  --log_device_placement [LOG_DEVICE_PLACEMENT]
                        Log placement of ops on devices
  --nolog_device_placement                      
```

Train:

```bash
./train.py
```                        