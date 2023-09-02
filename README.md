# ESTADO DEL ARTE TENSORRT

## Instalacion

instalar pytorch segun se explica en `https://pytorch.org/get-started/locally/`, ejemplo:

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

instalar requirements con

```
pip install -r requirements.txt
```

---

# REFERENICAS A SOLUCIONES DE ERRORES

* Unet parts: `https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py`
* Unet:  `https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets/blob/master/Models.py`
* corrige error de tamaños en torch.cut: `https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py` linea 56
