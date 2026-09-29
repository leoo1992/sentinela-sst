# Sentinela SST

Aplicação full-stack de visão computacional em tempo real voltada a Segurança e Saúde no Trabalho. O projeto usa **Next.js + TypeScript** no frontend e **FastAPI + Python** no backend, com deploy unificado na Vercel.

## Escopo

O projeto está focado exclusivamente em **Inspeção de EPI**.

Itens avaliados:
- óculos, independentemente do tipo ou especificação;
- capacete, independentemente do tipo, cor ou especificação;
- luvas, independentemente do tipo ou material;
- protetor auricular, incluindo plug e abafador/concha.

Fotos enviadas passam por análise combinada entre TensorFlow/MoveNet, heurísticas locais e OpenCV no backend.

## Privacidade por arquitetura

A aplicação foi desenhada para **não armazenar vídeo, imagem ou histórico**.

1. A câmera é aberta pelo navegador com getUserMedia.
2. MoveNet/TensorFlow.js executa a estimativa de pose no dispositivo.
3. O frontend extrai métricas derivadas como ângulos e estados visuais.
4. FastAPI recebe as métricas derivadas e aplica regras.
5. A câmera ao vivo continua processada localmente, sem envio de frames.
6. Em fotos enviadas manualmente, uma versão reduzida é analisada pelo backend com OpenCV e o resultado é combinado com o TensorFlow/MoveNet do navegador. A aplicação não mantém histórico nem armazena a foto.

## Stack

Next.js 16, React 19, TypeScript, TensorFlow.js, MoveNet SinglePose, Python 3.12, FastAPI, OpenCV, Vitest, Pytest, GitHub Actions e Vercel.

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

Este é um protótipo técnico e de portfólio, não um sistema certificado para tomada de decisão de SST. Itens pequenos, ocultos, desfocados ou fora do enquadramento devem permanecer inconclusivos ou não avaliáveis.

Os indicadores não substituem inspeção presencial, procedimentos aplicáveis ou profissional responsável pela atividade.

## Licença

MIT.
