# ESTADO DEL ARTE TENSORRT

## Instalacion

### Prerquisites

* Linux (based on Ubuntu 22.04 LTS)
* Python 3.10 or lower
* virtualenv
* CUDA 12.2

### Requirements to install

```
pip install -r requirements.txt --no-cache-dir --use-pep517
```

---

## Train:

Para entrenar usar este comando

```
python train.py --batch_size=128 --weights='weights/best.pth'
```

## Eval

### Eval test -outdated- need review!

Para evaluar en dataset de pruebas:

```
python eval.py
```

### Eval Experiment -updated!-

Para evaluar en imagenes reales:

```
python eval.py --experiment --case X
```
Donde X corresponde a una condición de llama. Por ejemplo, para la llama Yale-60:

```
python eval.py --weights weights/modelo_base_unet.pth --model unet --experiment --case C
```

Existen los casos:

* case A: emi ?¿
* case B: emi ?¿
* case C: mae para la llama Yale-60

Existe los modelos:

* unet
* attunet
* tensorrt


### Eval Compare two models -updated!-

Para comparar dos modelos se puede de la siguiente manera:

```
python eval.py --weights 'weights/attunet.pth weights/unet.pth' --model 'attunet unet' --case C --compare
```

---

## TensorRT Optimization

### ONNX 

Luego de entrenar la data y tener el archivo `.pth`, es necesario transformarlo a un `.onnx` con el siguiente codigo

```
python onnx_transform.py --weights weights/best_5.pth --input_shape 1 3 128 32
```

Note:

input shape es B C H W

### ENGINE

Luego de tener el `.onnx` se puede optimizar usando TensorRT con el siguiente script

```
python build_trt.py --weights weights/best.onnx --input_shape 1 3 128 32 --int8
```

---

# REFERENICAS A SOLUCIONES DE ERRORES

* Unet parts: `https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py`
* Unet:  `https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets/blob/master/Models.py`
* corrige error de tamaños en torch.cut: `https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py` linea 56


# INSTALL ON JETSON ORIN 

## git and add virtualenv

```
$ sudo apt-get install python3-pip
$ pip install virtualenv
```

You might encounter a message like WARNING: The script virtualenv is installed in '/home/your_user_name/.local/bin' which is not on PATH. Therefore, it is necessary to add it to the PATH.

```
$ sudo apt-get install nano
$ sudo nano ~/.bashrc
```

At the end of your ~/.bashrc file, add the following line:

`export PATH=/home/your_user_name/.local/bin:$PATH`
Remember to run source ~/.bashrc to apply the changes.

Create and activate a virtual environment in the ArtTRT repository folder, use the following commands:

```
$ git clone git@github.com:Juanx65/UNetTRT.git
$ cd UNetTRT/
$ virtualenv env --system-site-packages
$ source env/bin/activate
```

## Install PyTorch

Depending on the version of Jetpack used, and according to the official Nvidia page PyTorch for Jetson[https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048], download the appropriate wheel file.

For example, for PyTorch 2.3.0 for CUDA 12.2 with Jetpack 6.0 we download the following wheel:

```
wget https://nvidia.box.com/shared/static/mp164asf3sceb570wvjsrezk1p4ftj8t.whl
```

Chage its name to match with the one in Nvidia's page as torch-2.3.0-cp310-cp310-linux_aarch64.whl
and install the wheel as follows:

```
$ sudo apt-get install libopenblas-base libopenmpi-dev  
$ pip install torch-2.3.0-cp310-cp310-linux_aarch64.whl
```

To verify the installation:

```
$ python
>>import torch
>>torch.cuda.is_available()
True
>>exit()
```

## Install torchvision 

Depending on the version of Jetpack used, and according to the official Nvidia page PyTorch for Jetson[https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048], download the appropriate wheel file.

For example, for torchvision 0.18.0 for CUDA 12.2 with Jetpack 6.0 we download the following wheel:

wget https://nvidia.box.com/shared/static/xpr06qe6ql3l6rj22cu3c45tz1wzi36p.whl

Chage its name to match with the one in Nvidia's page as torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl
and install the wheel as follows:

```
$ sudo apt-get install libopenblas-base libopenmpi-dev  
$ pip install torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl
```

## pip dependecies

```
scipy==1.13.0
random2
numpy
matplotlib
pandas
scikit-learn
tqdm==4.66.1
torchsummary
opencv-python
polygraphy==0.49.0
onnx_opcounter
tabulate==0.9.0
```