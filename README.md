# ESTADO DEL ARTE TENSORRT

## Instalacion

instalar pytorch segun se explica en `https://pytorch.org/get-started/locally/`, ejemplo:

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

instalar requirements con

obs: quizas faltan otros requierements que se me olvido añadir... revisar

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
python eval.py --experiment
```

---

# REFERENICAS A SOLUCIONES DE ERRORES

* Unet parts: `https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py`
* Unet:  `https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets/blob/master/Models.py`
* corrige error de tamaños en torch.cut: `https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py` linea 56
