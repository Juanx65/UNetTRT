# ESTADO DEL ARTE TENSORRT

## Instalacion

instalar pytorch segun se explica en `https://pytorch.org/get-started/locally/`, ejemplo:

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

instalar requirements con

obs: para instalar el torch de los requirements, es necesario contar con cuda 12.2

```
pip install -r requirements.txt
```

---

## Train:

Para entrenar usar este comando

```
python train.py --batch_size=32 --epochs=100 --dropout=0.089735 --num_filters=29 --learning_rate=0.000410 --weights='weights/best.pth'
```

## Eval

### Eval test

Para evaluar en dataset de pruebas:

```
python eval.py
```

### Eval exp

Para evaluar en imagenes reales:

```
python eval.py --experiment --case X
```
Donde X corresponde a una condición de llama. Por ejemplo, para la llama Yale-60:
```
python eval.py --batch_size 128 --kernel_size 4 --dropout 0.119372 --num_filters 20 --learning_rate 0.001112 --weights 'weights/best_5.pth' --experiment --case D
```

### Eval TRT experimental

Por ahora solo se puede probar TRT con la data experimental, para hacerlo, debes usar este codigo

```
python eval.py --trt --weights='weights/best.engine'
```

### Eval Compare TRT vs Vanilla

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
python onnx_transform.py --weights weights/best.pth --input_shape 1 3 128 32
```

Note:

input shape es B C H W

... comprobar si es BCHW o BCWH ...?

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
