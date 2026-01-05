# Chatterbox Turbo Fine-tuning para Español (RunPod)

Fine-tuning de Chatterbox Turbo con vocabulario extendido para español/guatemalteco.

## Requisitos RunPod
- GPU: RTX 3090/4090 o A100 (mínimo 24GB VRAM)
- Template: PyTorch 2.0+ con CUDA

## Setup Rápido en RunPod

```bash
# 1. Clonar repo
git clone https://github.com/TU_USUARIO/chatterbox-spanish-finetune.git
cd chatterbox-spanish-finetune

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Descargar modelos Turbo
cd finetuning_utils
python setup.py

# 4. Subir dataset (metadata.csv + wavs/)
# Opción A: scp desde local
# Opción B: wget desde Google Drive/S3

# 5. Entrenar
python train.py
```

## Configuración (finetuning_utils/src/config.py)

```python
is_turbo: bool = True          # Modo Turbo para español
new_vocab_size: int = 52260    # Vocab extendido
batch_size: int = 4            # Con 24GB VRAM
grad_accum: int = 2            # Effective batch = 8
```

## Dataset Format (LJSpeech)
```
chatterbox_dataset_2/
├── metadata.csv    # filename|raw_text|normalized_text
└── wavs/
    ├── 0001.wav
    ├── 0002.wav
    └── ...
```

## Monitoreo
```bash
# TensorBoard
tensorboard --logdir models/turbo/runs --port 6006

# Logs en tiempo real
tail -f train.log
```

## Checkpoints
Se guardan cada 500 steps en `models/turbo/checkpoint-*/`

## Inferencia después de entrenar
```bash
python inference.py
```
