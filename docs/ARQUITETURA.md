# Arquitetura — Sentinela SST

## Objetivo

O Sentinela SST está focado exclusivamente em quatro EPIs:

- óculos;
- capacete;
- luvas;
- protetor auricular de segurança (plug ou abafador/concha).

Fones de áudio comuns não são uma classe válida de EPI.

## Foto enviada

```text
Imagem
  ├─ MoveNet/TensorFlow.js → regiões de cabeça/olhos/orelhas/mãos
  ├─ Heurística local → evidência auxiliar
  └─ FastAPI
       ├─ OpenCV → evidência visual por região
       ├─ YOLOv8n SH17 ONNX ─┐
       ├─ YOLOv10n SH17 ONNX ├─ inferência inteira + fatiada
       └─────────────────────┘
                    ↓
             Weighted Boxes Fusion
                    ↓
       média ponderada por classe
                    ↓
       Detectado / Inconclusivo /
       Não detectado / Não avaliável
```

## Por que dois modelos

Os dois modelos SH17 produzem caixas e probabilidades independentes. O Sentinela combina detecções espacialmente compatíveis por Weighted Boxes Fusion em vez de simplesmente escolher o maior score.

A probabilidade final considera:

- ensemble treinado: peso principal;
- OpenCV: validação visual/regional;
- heurística local orientada pela pose: apoio.

Os pesos são normalizados quando alguma fonte está indisponível.

## Objetos pequenos

Óculos e protetores auditivos podem ocupar poucos pixels. Imagens grandes são analisadas tanto por inteiro quanto em recortes sobrepostos de até 640 px. As detecções são remapeadas para a imagem original antes da fusão.

Essa abordagem segue o mesmo princípio de sliced inference popularizado pelo SAHI, mas o runtime implementa a divisão diretamente para não adicionar dependências pesadas à função Vercel.

## Proteção auditiva x fone de áudio

O modelo SH17 só mapeia a classe industrial `ear-mufs` para `protetorAuricular`.

Além disso:

- a caixa precisa estar próxima das orelhas quando MoveNet fornece esses pontos;
- caixas excessivamente grandes são rejeitadas;
- OpenCV/heurística não podem confirmar sozinhos proteção auditiva: sem confirmação do detector treinado o resultado fica inconclusivo/não avaliável;
- o pipeline de treino futuro exige headphones/headsets/earbuds como negativos de fundo.

## Modelos

Os pesos PT benchmark do SH17 não são empacotados no deploy. Um workflow separado:

1. baixa YOLOv8n e YOLOv10n do release oficial SH17;
2. exporta ambos para ONNX;
3. publica os ONNX no release `ppe-models-v1` deste repositório;
4. não cria commit adicional.

Em runtime, a função baixa os ONNX sob demanda para `/tmp` e os mantém em cache enquanto a instância permanece quente.

## Próxima geração

`training/train_next_gen.py` prepara fine-tuning de:

- YOLO26;
- RT-DETR.

Eles só devem entrar no ensemble depois de treinamento e validação nas quatro classes PPE. Um checkpoint COCO genérico não substitui um modelo PPE treinado.
