import torch
import onnx
import os
from io import BytesIO
from models.unet import U_Net

BATCH_SIZE = 1


current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
weights_path = 'weights/best.pth'
weights_path = os.path.join(parent_directory,weights_path)


model = U_Net() # por alguna razon debe importarse un modelo o si no se pone triste esta cosa
model = torch.load(weights_path)
model.to('cuda:0')
model.eval()
fake_input = torch.randn([BATCH_SIZE,3, 128, 32]).to('cuda:0')
for _ in range(2):
    model(fake_input)
save_path = weights_path.replace('.pth', '.onnx')

with BytesIO() as f:
    torch.onnx.export(
        model,
        fake_input,
        f,
        opset_version=11,
        input_names=['images'],
        output_names=['outputs'])
    f.seek(0)
    onnx_model = onnx.load(f)

# Guardar el modelo ONNX en un archivo .onnx
onnx.save(onnx_model, save_path)

print("La conversión a ONNX se ha completado exitosamente. El modelo se ha guardado en:", save_path)