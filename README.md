# Sentinela SST

[English version](README.en.md)

Aplicação full-stack de visão computacional em tempo real voltada a Segurança e Saúde no Trabalho. O frontend usa Next.js e TypeScript; a API usa FastAPI e Python.

## Módulos

| Módulo | Objetivo |
| --- | --- |
| Inspeção de EPI | Checklist visual experimental de capacete, óculos, colete, luvas e calçado. |
| Segurança em Altura | Pose, capacete e zona virtual configurável de borda/risco. |
| Análise Ergonômica | Esqueleto e métricas de tronco, pescoço, joelhos e assimetrias. |
| Levantamento de Cargas | Fase aproximada do movimento e indicadores posturais. |

## Privacidade por arquitetura

1. A câmera é aberta no navegador com `getUserMedia`.
2. MoveNet/TensorFlow.js executa a estimativa de pose no dispositivo.
3. O frontend extrai métricas derivadas.
4. FastAPI recebe apenas essas métricas e aplica regras.
5. Frames de vídeo não são enviados ao backend.

## Stack

Next.js 16, React 19, TypeScript, TensorFlow.js, MoveNet MultiPose, Python 3.12, FastAPI, Vitest, Pytest, GitHub Actions e Vercel.

## Executar

```bash
npm install
npm run dev
```

API isolada:

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
pytest -q
```

## Limitações

Protótipo técnico e de portfólio, não certificado para tomada de decisão de SST. Heurísticas visuais podem produzir resultados inconclusivos; os indicadores não substituem inspeção presencial, APR, análise ergonômica formal, laudos, normas ou profissional habilitado.
