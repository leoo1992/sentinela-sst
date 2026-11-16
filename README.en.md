# Sentinela SST

[Versão em português](README.md)

A full-stack, real-time computer vision application focused on Occupational Health and Safety. The frontend uses Next.js and TypeScript; the API uses FastAPI and Python.

## Modules

| Module | Purpose |
| --- | --- |
| PPE inspection | Experimental visual checklist for helmets, glasses, vests, gloves, and footwear. |
| Working at height | Pose, helmet, and configurable virtual edge/risk zone. |
| Ergonomic analysis | Skeleton tracking and indicators for torso, neck, knees, and asymmetry. |
| Load handling | Approximate movement phase and posture indicators. |

## Privacy by design

1. The browser accesses the camera through `getUserMedia`.
2. MoveNet/TensorFlow.js performs pose estimation on-device.
3. The frontend derives visual metrics.
4. FastAPI receives only those metrics and applies rules.
5. Video frames are not sent to the backend.

## Stack

Next.js 16, React 19, TypeScript, TensorFlow.js, MoveNet MultiPose, Python 3.12, FastAPI, Vitest, Pytest, GitHub Actions, and Vercel.

## Run locally

```bash
npm install
npm run dev
```

Run the API separately:

```bash
python -m venv .venv
pip install -e ".[dev]"
uvicorn api.index:app --reload --port 8000
```

## Quality checks

```bash
npm run lint
npm run typecheck
npm run test
npm run build
pytest -q
```

## Limitations

This is a portfolio prototype, not a certified Occupational Health and Safety decision system. Visual heuristics can be inconclusive. Its indicators do not replace on-site inspection, formal risk assessments, ergonomic analysis, technical reports, applicable standards, or qualified professionals.
