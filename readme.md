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

Pip freeze results from GPU PC (Win 10 64-bit):

```bash
# python 3.9.19
absl-py==1.4.0
array-record==0.4.1
asttokens==2.0.5
astunparse==1.6.3
attrs==23.1.0
backcall==0.2.0
cachetools==5.3.3
certifi==2024.6.2
charset-normalizer==3.1.0
colorama==0.4.5
contourpy==1.2.1
cycler==0.12.1
daiquiri==3.2.5.1
dcor==0.6
debugpy==1.6.0
decorator==5.1.1
emd==0.6.2
entrypoints==0.4
etils==1.5.2
executing==0.8.3
flatbuffers==24.3.25
fonttools==4.53.0
fsspec==2023.10.0
gast==0.4.0
google-auth==2.31.0
google-auth-oauthlib==0.4.6
google-pasta==0.2.0
googleapis-common-protos==1.61.0
grpcio==1.64.1
gviz-api==1.10.0
h5py==3.11.0
idna==3.7
importlib_metadata==8.0.0
importlib_resources==6.4.0
ipykernel==6.15.0
ipython==8.4.0
jax==0.4.20
jedi==0.18.1
joblib==1.4.2
JsonForm==0.0.2
jsonschema==4.19.2
jsonschema-specifications==2023.7.1
JsonSir==0.0.2
jupyter-client==7.3.4
jupyter-core==4.10.0
keras==2.10.0
keras-core==0.1.7
Keras-Preprocessing==1.1.2
keras-tuner==1.4.7
kiwisolver==1.4.5
kt-legacy==1.0.5
libclang==18.1.1
llvmlite==0.41.0
Markdown @ file:///home/conda/feedstock_root/build_artifacts/markdown_1710435156458/work
markdown-it-py==3.0.0
MarkupSafe==2.1.5
matplotlib==3.9.0
matplotlib-inline==0.1.3
mdurl==0.1.2
nest-asyncio==1.5.5
numba==0.58.0
numpy==1.23.5
oauthlib==3.2.2
opt-einsum==3.3.0
packaging @ file:///home/conda/feedstock_root/build_artifacts/packaging_1718189413536/work
pandas==2.2.2
parso==0.8.3
pickleshare==0.7.5
pillow==10.4.0
promise==2.3
prompt-toolkit==3.0.30
protobuf==3.19.6
psutil==5.9.1
pure-eval==0.2.2
pyasn1==0.6.0
pyasn1_modules==0.4.0
Pygments==2.19.1
pyparsing==3.0.9
python-dateutil==2.8.2
Python-EasyConfig==0.1.7
python-json-logger==3.3.0
python-resources==0.3
pytz==2024.1
pywin32==304
PyYAML==6.0.1
pyzmq==23.2.0
referencing==0.30.2
requests==2.32.3
requests-oauthlib==2.0.0
Resource==0.2.1
rich==13.6.0
rpds-py==0.12.0
rsa==4.9
scikit-learn==1.5.1
scipy==1.13.1
six==1.16.0
sparse==0.14.0
stack-data==0.3.0
tabulate==0.9.0
tensorboard==2.10.1
tensorboard-data-server==0.6.1
tensorboard-plugin-profile==2.17.0
tensorboard-plugin-wit==1.8.1
tensorflow==2.10.0
tensorflow-datasets==4.9.2
tensorflow-estimator==2.10.0
tensorflow-io-gcs-filesystem==0.31.0
tensorflow-metadata==1.14.0
termcolor==2.4.0
threadpoolctl==3.5.0
tornado==6.1
traitlets==5.3.0
typing_extensions==4.12.2
tzdata==2024.1
urllib3==2.2.2
wcwidth==0.2.5
Werkzeug==3.0.3
wrapt==1.14.1
zipp==3.19.2
```