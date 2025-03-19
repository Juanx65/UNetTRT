# PIROMETRÍA DEL HOLLÍN USANDO TENSORRT

## Instalación

### Prerrequisitos

* Linux (basado en Ubuntu 22.04 LTS)
* Python 3.10 o inferior
* CUDA 12.2

## Dataset

Los datasets utilizados en este trabajo pueden descargarse desde el siguiente 
[enlace](https://drive.google.com/drive/folders/1lYqGvNBEsVsJ2xiGPNvcZCEPdD_nZ9Aa?usp=sharing).

### Instalación de requisitos

```
pip install -r requirements.txt --no-cache-dir --use-pep517
```

## Flujo de trabajo

Puedes encontrar un ejemplo con todos los códigos ejecutables en `workflow.ipynb` en el directorio raíz de este repositorio.

## Optimización con TensorRT

Antes de las evaluaciones, es necesario generar los modelos optimizados con TensorRT.

### ONNX

Con los modelos previamente entrenados en formato `.pth`, es necesario transformarlos a `.onnx` con el siguiente código:

```
python onnx_transform.py --weights weights/attunet.pth --input_shape 1 3 128 32
```

Donde:

* `--weights weights/attunet.pth` corresponde al modelo base que se desea transformar a ONNX.
* `--input_shape 1 3 128 32` corresponde a la forma de entrada en formato B C H W.

### ENGINE

Una vez generado el archivo `.onnx`, se puede optimizar con TensorRT utilizando el siguiente script:

```
python build_trt.py --weights weights/attunet.onnx --input_shape 1 3 128 32 --int8 --engine_name attunet_int8.engine
```

---

## Evaluación de un modelo preentrenado

Para evaluar la pirometría del hollín con un modelo previamente entrenado, se usa el siguiente script:

```
python eval.py
```

Este script cuenta con los siguientes parámetros:

| Parámetro       | Tipo   | Descripción                                                                                      | Valor por defecto       |
|----------------|--------|--------------------------------------------------------------------------------------------------|-------------------------|
| `--batch_size`  | Int    | Tamaño del lote (batch).                                                                        | `1`                     |
| `--dataset`     | String | Carpeta del dataset para evaluar latencia o throughput; imágenes en formato TIFF.              | `"datasets/img_preprocess"` |
| `--model`       | String | Modelo a evaluar (`tensorrt`, `unet` o `attunet`); si se compara, usar un espacio entre modelos. | `"attunet"`             |
| `--weights`     | String | Ruta a los pesos del modelo; si se compara, usar un espacio para indicar dos rutas.           | `"weights/attunet.pth"` |
| `--experiment`  | Flag   | Indica si se está ejecutando un experimento.                                                   | `False`                 |
| `--case`        | String | Condición de llama (`A`, `B` o `C`).                                                           | `"A"`                   |
| `--compare`     | Flag   | Habilita la comparación entre la red optimizada con TensorRT y la versión base.               | `False`                 |
| `--compare_all` | Flag   | Compara todos los modelos para un solo caso específico.                                       | `False`                 |
| `--latency`     | Flag   | Evalúa la latencia si el batch size es 1; evalúa throughput si el batch size es mayor a 1.    | `False`                 |
| `--closeness`   | Flag   | Evalúa la métrica de closeness sobre el dataset para todos los modelos posibles.             | `False`                 |


Las condiciones de llama (`--case`) corresponden a los siguientes casos:

* Caso A: Emisión para la llama Yale-32
* Caso B: Emisión para la llama Yale-40
* Caso C: MAE para la llama Yale-60

## Ejemplos de uso

* Para evaluar en imágenes reales, por ejemplo, la llama Yale-60 con el modelo base Attention U-Net:

    ```
    python eval.py --weights weights/attunet.pth --model attunet --experiment --case C
    ```

* Para evaluar en imágenes reales, por ejemplo, la llama Yale-60 con el modelo optimizado TensorRT FP16 Attention U-Net:

    ```
    python eval.py --weights weights/attunet_fp16.engine --model tensorrt --experiment --case C
    ```

* Para comparar dos modelos específicos entre sí:

    ```
    python eval.py --weights 'weights/attunet.pth weights/unet.pth' --model 'attunet unet' --case C --compare
    ```

* Para comparar todos los modelos posibles:

    Para comparar todos los modelos previamente entrenados y/o optimizados:

    * Modelo base UNet
    * Modelo base Attention UNet
    * TRT FP32 UNet
    * TRT FP16 UNet
    * TRT INT8 UNet
    * TRT FP32 Attention UNet
    * TRT FP16 Attention UNet
    * TRT INT8 Attention UNet

    ```
    python eval.py --experiment --compare_all --case X
    ```

    Donde X corresponde a una condición de llama (`A`, `B` o `C`).


* Para evaluar la latencia de un modelo especifico en un dataset experimental

    ```
    python eval.py --weights weights/attunet.pth --model attunet --dataset datasets/img_preprocess --latency --batch_size 1
    ```
    
* Para evaluar el throughput de un modelo especifico en un dataset experimental

    ```
    python eval.py --weights weights/attunet.pth --model attunet --dataset datasets/img_preprocess --latency --batch_size 4
    ```

    `--latency` inmediatamente calculara el thr. para cualquier batch_size mayor a uno.

* Para evaluar el closeness sobre todos los modelos posibles en un dataset experimenta

    ```
    python eval.py --closeness
    ```

---

## Entrenamiento (desactualizado):

Para entrenar, usa este comando:

```
python train.py --batch_size=128 --weights='weights/best.pth'
```

# Referencias a soluciones de errores

* Partes de UNet: [`https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py`](https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py)
* UNet:  [`https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets/blob/master/Models.py`](https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets/blob/master/Models.py)
* Corrección de error de tamaños en `torch.cut`: [`https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py`](https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py) (línea 56)

# Instalación en Jetson Orin

## Instalación de `git` y creación de un entorno virtual

```
$ sudo apt-get install python3-pip
$ pip install virtualenv
```

Si aparece un mensaje de advertencia indicando que el script `virtualenv` no está en el `PATH`, es necesario agregarlo:

```
$ sudo apt-get install nano
$ sudo nano ~/.bashrc
```

Al final del archivo `~/.bashrc`, agrega la siguiente línea:

```
export PATH=/home/your_user_name/.local/bin:$PATH
```

Luego, ejecuta:

```
source ~/.bashrc
```

Para crear y activar un entorno virtual en el repositorio `UNetTRT`, usa los siguientes comandos:

```
$ git clone git@github.com:Juanx65/UNetTRT.git
$ cd UNetTRT/
$ virtualenv env --system-site-packages
$ source env/bin/activate
```

## Instalación de PyTorch

Según la versión de JetPack utilizada y la página oficial de Nvidia sobre PyTorch para Jetson ([Enlace](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)), descarga el archivo wheel correspondiente.

Ejemplo para PyTorch 2.3.0 con CUDA 12.2 y JetPack 6.0:

```
wget https://nvidia.box.com/shared/static/mp164asf3sceb570wvjsrezk1p4ftj8t.whl
```

Renómbralo para coincidir con la versión en la página de Nvidia:

```
torch-2.3.0-cp310-cp310-linux_aarch64.whl
```

Instálalo con:

```
$ sudo apt-get install libopenblas-base libopenmpi-dev  
$ pip install torch-2.3.0-cp310-cp310-linux_aarch64.whl
```

## Dependencias de `pip`

```
tqdm==4.67.1
scikit-learn
