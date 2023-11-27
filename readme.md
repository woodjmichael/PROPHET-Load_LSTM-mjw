# Environment

Pipfile and piplock don't work well off the microgrid PC.

This seems to work:

```bash
# create python 3.8 env with tensorflow
conda create --name prophet-lstm python=3.8 -y
conda activate prophet-lstm
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 -y
pip install "tensorflow<2.11"

# check that you have tensorflow for the gpu
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# install the rest
pip install pandas
pip install daiquiri
pip install scikit-learn
pip install matplotlib
```