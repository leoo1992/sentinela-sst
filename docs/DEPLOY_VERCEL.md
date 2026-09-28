# Deploy de produção — Vercel

O Sentinela SST foi preparado para um único projeto Vercel contendo Next.js e FastAPI.

## Fonte correta

- Repositório: `leoo1992/sentinela-sst`
- Branch de produção: `master`
- Diretório raiz: raiz do repositório
- Framework: Next.js (detecção automática)
- Variáveis de ambiente: nenhuma obrigatória
- Backend Python: `api/index.py`
- Dependências Python: `pyproject.toml`

> O repositório ainda possui `main` como branch padrão do GitHub. Na criação do projeto Vercel, a branch de produção deve ser explicitamente configurada como `master`.

## Validação após publicar

1. A página inicial deve abrir com os quatro módulos.
2. `/api/health` deve responder com `status: ok`.
3. Ao clicar em **Abrir câmera**, o navegador deve solicitar permissão.
4. A câmera deve permanecer local; o backend recebe apenas métricas derivadas.
5. No celular, validar câmera frontal e traseira.
