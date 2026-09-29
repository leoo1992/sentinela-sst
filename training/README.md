# Modelos de EPI do Sentinela SST

## Produção

A produção usa dois detectores SH17 leves e independentes, exportados para ONNX:

- YOLOv8n SH17;
- YOLOv10n SH17.

As classes consumidas pelo Sentinela são apenas `glasses`, `gloves` e `helmet`.

O workflow `PPE model assets` baixa os pesos públicos SH17, exporta ONNX e publica os arquivos no release `ppe-models-v1`. O runtime usa OpenCV DNN e baixa os ONNX para `/tmp`, sem carregar PyTorch na Vercel.

## Próxima geração: YOLO26 + RT-DETR

O script `train_next_gen.py` prepara treino de YOLO26 e RT-DETR com o mesmo dataset. RT-DETR genérico COCO não entra em produção porque não possui as classes PPE necessárias.

Dataset alvo:

```yaml
names:
  0: glasses
  1: helmet
  2: gloves
```

```bash
pip install -r training/requirements.txt
python training/train_next_gen.py --data /caminho/data.yaml --epochs 120 --imgsz 832
```
