# Modelos de EPI do Sentinela SST

## Produção

A produção usa dois detectores SH17 leves e independentes, exportados para ONNX:

- YOLOv8n SH17;
- YOLOv10n SH17.

As classes consumidas são apenas `glasses`, `gloves`, `helmet` e `ear-mufs`. **Headphones, headset, earbuds e earphones não são classes aceitas como EPI.** Para reduzir falsos positivos, `ear-mufs` também precisa aparecer junto à região das orelhas quando há pose disponível.

O workflow `PPE model assets` baixa os pesos públicos SH17, exporta ONNX e publica os arquivos no release `ppe-models-v1`. O runtime usa OpenCV DNN e baixa os ONNX para `/tmp`, sem carregar PyTorch na Vercel.

## Próxima geração: YOLO26 + RT-DETR

O script `train_next_gen.py` prepara treino de YOLO26 e RT-DETR com o mesmo dataset. RT-DETR genérico COCO não entra em produção porque não possui as classes PPE necessárias.

Dataset alvo:

```yaml
names:
  0: glasses
  1: helmet
  2: gloves
  3: hearing_protection
```

Para `hearing_protection`, mantenha abafadores/conchas e plugs como positivos. Imagens de fones de música/headsets/earbuds devem entrar como **negativos de fundo**, sem anotação de EPI.

```bash
pip install -r training/requirements.txt
python training/train_next_gen.py --data /caminho/data.yaml --epochs 120 --imgsz 832
```
