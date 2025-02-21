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


### Eval Compare TRT vs Vanilla -outdated- need review!

Para comparar la red optimizada con TRT vs la red sin optimizacion (VANILLA) se puede de la siguiente manera:

```
python eval.py --compare
```

por ahora usa por default el path al engine `weights/best.engine` y otros detalles x parametrizar

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
