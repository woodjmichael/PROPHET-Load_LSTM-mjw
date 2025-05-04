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
pip install pandas daiquiri scikit-learn matplotlib
```

Good HP search (1 epoch)
```
,units,n_back,n_dense,dropout,vloss,test_mae_pers,test_mae_pred,test_skill,frac_0_mae_pers

11,144,96,36,0.1,0.050905898213386536,0.04557496955470568,0.03828909344982988,0.15986573718124453,0.0
```