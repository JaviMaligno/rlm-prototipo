# RLM Prototipo (Out-of-Core)

Mini-proyecto para demostrar un **Recursive Language Model (RLM)** que trabaja con un documento enorme fuera de la ventana de contexto. El LLM inspecciona el contenido usando un entorno Python persistente, con paginación/búsqueda y sub-llamadas `llm_query()`.

## Requisitos
- Python 3.12+
- `uv` instalado
- Credenciales de Azure OpenAI

## Setup rápido
1. Crea un `.env` basado en `.env.example`.
2. Instala dependencias:
```bash
uv venv
uv pip install -e .
```
3. Ejecuta el CLI:
```bash
rlm run --input "data/*.txt" --question "¿Cuál es la contribución principal?"
```

> Nota: no subas claves al repo. Si has pegado una clave en chats o logs, rótala.

## Flujo de demo
1. Junta varios artículos/papers en `data/` (se ignora por git).
2. El CLI concatenará los archivos y cargará todo en `context` dentro del entorno Python.
3. El LLM leerá solo lo necesario vía slicing/búsqueda y sub-llamadas.

## Tamaño objetivo del documento
Para que el demo tenga sentido, apunta a **~1M tokens** totales (aprox `~4M caracteres`).  
Puedes lograrlo concatenando varios papers/artículos largos sobre LLMs.

## CLI
```bash
rlm run \
  --input "data/*.txt" \
  --question "Resume los hallazgos clave" \
  --max-turns 12 \
  --max-subcalls 20 \
  --max-obs-chars 8000
```

## Variables de entorno
Ver `.env.example`.

`AZURE_OPENAI_ENDPOINT` debe ser el **endpoint base** del recurso (sin `/openai/...`).
Si pegas una URL completa, el CLI intentara normalizarla automaticamente.

## Estructura
- `src/rlm/` Orquestador, entorno Python y cliente LLM
- `data/` Documentos de demo (gitignored)

## Licencia
MIT
