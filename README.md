# Sentinela SST

Aplicação full-stack de visão computacional em tempo real voltada a Segurança e Saúde no Trabalho. O projeto usa **Next.js + TypeScript** no frontend e **FastAPI + Python** no backend, com deploy unificado na Vercel.

## Módulos

| Módulo | Objetivo |
| --- | --- |
| Inspeção de EPI | Checklist visual experimental de capacete, óculos, colete, luvas e calçado. |
| Segurança em Altura | Pose, capacete e uma zona virtual configurável de borda/risco. |
| Análise Ergonômica | Esqueleto e métricas de tronco, pescoço, joelhos e assimetrias. |
| Levantamento de Cargas | Fase aproximada do movimento e indicadores posturais durante o levantamento. |

## Privacidade por arquitetura

A aplicação foi desenhada para **não armazenar vídeo, imagem ou histórico**.

1. A câmera é aberta pelo navegador com getUserMedia.
2. MoveNet/TensorFlow.js executa a estimativa de pose no dispositivo.
3. O frontend extrai métricas derivadas como ângulos e estados visuais.
4. FastAPI recebe apenas essas métricas e aplica regras.
5. Nenhum frame de vídeo é enviado ao backend.

## Stack

Next.js 16, React 19, TypeScript, TensorFlow.js, MoveNet MultiPose, Python 3.12, FastAPI, Vitest, Pytest, GitHub Actions e Vercel.

## Executar

    npm install
    npm run dev

Para testar somente a API:

    python -m venv .venv
    pip install -e ".[dev]"
    uvicorn api.index:app --reload --port 8000

Na Vercel, o arquivo api/index.py é empacotado como função Python e o Next.js permanece na raiz do mesmo projeto.

## Qualidade

    npm run lint
    npm run typecheck
    npm run test
    npm run build
    pytest -q

O workflow de CI executa essas verificações na branch **master**.

## Limitações

Este é um protótipo técnico e de portfólio, não um sistema certificado para tomada de decisão de SST. A inspeção de EPI usa pose e heurísticas visuais leves para preservar execução no celular e privacidade. EPIs transparentes, pequenos, de cores não previstas ou parcialmente ocultos podem ficar inconclusivos.

Os indicadores não substituem inspeção presencial, APR, análise ergonômica formal, laudos, normas aplicáveis ou profissional legalmente habilitado.

## Licença

MIT.
