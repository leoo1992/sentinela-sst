# Sentinela SST

Aplicação full-stack de visão computacional para **inspeção visual de EPI**, com Next.js/TypeScript no frontend e FastAPI/Python no backend, mantendo deploy unificado na Vercel.

## EPIs avaliados

- óculos, independentemente do tipo ou especificação;
- capacete, independentemente do tipo, cor ou especificação;
- luvas, independentemente do tipo ou material.

## Ensemble de visão computacional

Fotos enviadas usam várias fontes independentes de evidência:

1. MoveNet/TensorFlow.js localiza cabeça, olhos e mãos;
2. OpenCV executa análise geométrica/visual das regiões corporais;
3. YOLOv8n treinado no SH17 roda em ONNX;
4. YOLOv10n treinado no SH17 roda em ONNX;
5. imagens grandes passam também por inferência fatiada para objetos pequenos;
6. caixas dos dois detectores são combinadas por **Weighted Boxes Fusion**;
7. a API calcula uma probabilidade final ponderada por EPI combinando ensemble treinado, OpenCV e análise local.

O runtime usa **OpenCV DNN para ONNX**, evitando PyTorch dentro da função Vercel. Se os ONNX ainda não estiverem disponíveis, o sistema continua operando com MoveNet + OpenCV em fallback.

## Modelos SH17

O workflow `.github/workflows/models.yml` baixa os pesos benchmark públicos SH17 para YOLOv8n e YOLOv10n, exporta ambos para ONNX e publica os artefatos no release `ppe-models-v1` sem criar commit adicional.

O SH17 possui classes como `glasses`, `gloves` e `helmet`, entre outras. Somente essas três classes são consumidas pelo Sentinela.

## YOLO26 + RT-DETR

A pasta `training/` inclui pipeline para treinar **YOLO26** e **RT-DETR** no mesmo dataset PPE e exportar ONNX. Eles só devem substituir os modelos atuais depois de fine-tuning e validação por classe; RT-DETR genérico COCO não possui as classes PPE necessárias.

## Privacidade

A aplicação não mantém histórico de imagens. A câmera ao vivo usa MoveNet no navegador. Fotos enviadas são processadas durante a requisição e não são persistidas pela aplicação.

## Stack

Next.js 16, React 19, TypeScript, TensorFlow.js, MoveNet SinglePose, Python 3.12, FastAPI, OpenCV DNN, ONNX, Weighted Boxes Fusion, Vitest, Pytest, GitHub Actions e Vercel.

## Executar

```bash
npm install
npm run dev
```

API local:

```bash
python -m venv .venv
pip install -e ".[dev]"
uvicorn api.index:app --reload --port 8000
```

## Qualidade

```bash
npm run lint
npm run typecheck
npm run test
npm run build
PYTHONPATH=. pytest -q
```

## Limitações

Este é um protótipo técnico/portfólio, não um sistema certificado de SST. Imagens desfocadas, oclusões, objetos muito pequenos e EPIs visualmente semelhantes a itens não-EPI podem exigir confirmação presencial.

O SH17 é disponibilizado pelos autores sob CC BY-NC-SA 4.0. Antes de uso comercial do modelo treinado nesses dados, revise as obrigações de licença ou treine modelos equivalentes com dataset compatível com o uso pretendido.

## Licença

MIT para o código deste repositório; pesos/datasets externos mantêm suas próprias licenças.
