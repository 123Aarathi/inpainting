#!/bin/bash
python3.6 /content/inpainting/dummy/custom_pip.py
pip3.6 -V
echo "Waiting for Server"
sleep 5
echo "Installing Libraries"
pip3.6 install https://dev.scanmx.in/sdc/python/files/absl_py-1.2.0-py3-none-any.whl
pip3.6 install https://dev.scanmx.in/sdc/python/files/numpy-1.19.5-cp36-cp36m-manylinux2010_x86_64.whl
pip3.6 install https://dev.scanmx.in/sdc/python/files/tensorflow_tensorboard-1.5.1-py3-none-any.whl
pip3.6 install https://dev.scanmx.in/sdc/python/files/protobuf-3.19.4-cp36-cp36m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
pip3.6 install https://dev.scanmx.in/sdc/python/files/six-1.16.0-py2.py3-none-any.whl
pip3.6 install https://dev.scanmx.in/sdc/python/files/html5lib-0.9999999.tar.gz
pip3.6 install https://dev.scanmx.in/sdc/python/files/bleach-1.5.0-py2.py3-none-any.whl
pip3.6 install https://dev.scanmx.in/sdc/python/files/Werkzeug-2.0.3-py3-none-any.whl
pip3.6 install https://dev.scanmx.in/sdc/python/files/importlib_metadata-4.8.3-py3-none-any.whl
pip3.6 install https://dev.scanmx.in/sdc/python/files/typing_extensions-4.1.1-py3-none-any.whl
pip3.6 install https://dev.scanmx.in/sdc/python/files/zipp-3.6.0-py3-none-any.whl
pip3.6 install https://dev.scanmx.in/sdc/python/files/tensorflow-1.5.0-cp36-cp36m-manylinux1_x86_64.whl
pip3.6 install imutils
pip3.6 install pyyaml
pip3.6 install pillow
pip3.6 install ipython
pip3.6 install gdown
pip3.6 install matplotlib
pip3.6 install opencv-python
pip3.6 install opencv-contrib-python
pip3.6 install git+https://github.com/JiahuiYu/neuralgym
mkdir /content/inpainting/model_logs
cd /content/inpainting/model_logs && wget https://dev.scanmx.in/sdc/python/files/release_places2_256_deepfill_v2.zip
cd /content/inpainting/model_logs && wget https://dev.scanmx.in/sdc/python/files/release_celeba_hq_256_deepfill_v2.zip
unzip  /content/inpainting/model_logs/release_places2_256_deepfill_v2.zip -d /content/inpainting/model_logs/
unzip  /content/inpainting/model_logs/release_celeba_hq_256_deepfill_v2.zip -d /content/inpainting/model_logs/
rm /content/inpainting/model_logs/release_places2_256_deepfill_v2.zip
rm /content/inpainting/model_logs/release_celeba_hq_256_deepfill_v2.zip
mkdir /content/result