# Arquitetura — Sentinela SST

## Fluxo

Câmera do navegador → TensorFlow.js / MoveNet → landmarks → métricas geométricas → FastAPI /api/evaluate → regras → interface.

## Decisão principal

Os frames permanecem no dispositivo. Enviar vídeo continuamente a uma função serverless aumentaria latência, custo e exposição de dados. O backend Python recebe somente dados derivados.

## Responsabilidades

**Next.js:** câmera, seletor de módulo, MoveNet, desenho do esqueleto, métricas e interface responsiva.

**FastAPI:** contrato de avaliação, regras independentes de UI, findings e endpoints de saúde. Sem persistência.

## Evolução

Uma versão futura pode substituir as heurísticas de EPI por um modelo ONNX/YOLO treinado especificamente para capacete, óculos, luvas, colete, cinturão e talabarte, mantendo o mesmo contrato de interface.
