# *Readme*

# Environment

Pipfile and piplock don't work well off the microgrid PC.

This seems to work:

## GPU

```bash
# create python 3.8 env with tensorflow
conda create --name prophet-lstm python=3.8 -y
conda activate prophet-lstm
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 -y
pip install "tensorflow<2.11"

# check that you have tensorflow for the gpu
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# install the rest
pip install pandas daiquiri scikit-learn matplotlib
```

## CPU

Tested in linux but need to save model as .keras not .hd5

```bash
conda create --name lstm python=3.11 -y
conda activate lstm
pip install tensorflow pandas daiquiri scikit-learn matplotlib
```

Gives some warnings but seems to work
