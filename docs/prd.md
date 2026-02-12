# PRD Tecnico — Prototipo RLM Out-of-Core

## Resumen
Construir un prototipo funcional de **Recursive Language Model (RLM)** que procese documentos masivos fuera de la ventana de contexto. El sistema carga el documento completo en un entorno Python persistente y el LLM lo inspecciona con slicing/busqueda y sub-llamadas `llm_query()` para responder preguntas.

## Objetivo del MVP
- Responder preguntas sobre un documento de **~1M tokens** sin inyectarlo completo en el prompt.
- Mostrar de forma visible el enfoque out-of-core: el LLM trae solo fragmentos necesarios.
- Permitir grabar un demo (logs claros en CLI).

## Alcance
Incluye:
- CLI con flujo completo RLM (orquestador + REPL persistente).
- Carga de multiples archivos locales.
- Herramientas `python_exec`, `final` y `llm_query`.
- Limites de seguridad (max turnos y max sub-llamadas).

No incluye:
- UI web.
- Persistencia en base de datos.
- Indexado vectorial (RAG).

## Arquitectura
Componentes:
1. **Orquestador CLI**: bucle conversacional, estado, limites, logs.
2. **Entorno Python persistente**: variable `context` y helpers (`get_slice`, `search`).
3. **LLM API (Azure OpenAI)**: genera codigo + decide que leer.

## Requisitos Funcionales
1. **Carga Out-of-Core**
   - El prompt largo del usuario NO se envia como mensaje.
   - Se inyecta en `context` dentro del entorno Python.
2. **Herramientas**
   - `python_exec(code)`: ejecuta codigo sobre `context`.
   - `llm_query(prompt_text)`: sub-llamada LLM para fragmentos.
   - `final(answer)`: finaliza la ejecucion.
3. **Persistencia**
   - Variables en el entorno Python persisten entre turnos.
4. **Limites**
   - `max_turns` y `max_subcalls` para evitar bucles y costes.
5. **Observabilidad**
   - Logs en CLI con turnos, sub-llamadas y outputs truncados.

## Requisitos No Funcionales
- Tiempo de respuesta aceptable para demo (latencia secuencial OK).
- Reproducible localmente con `.env`.
- No exponer secretos en repo.

## Flujo de Interaccion (MVP)
1. Usuario ejecuta `rlm run --input ... --question "..."`
2. CLI concatena archivos -> `context` en PythonEnv.
3. LLM recibe system prompt con instruccion out-of-core.
4. LLM llama `python_exec` para slicing/busqueda.
5. Si necesita, usa `llm_query` para resumir fragmentos.
6. LLM responde con `final(answer)`.

## Configuracion
Variables en `.env`:
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_DEPLOYMENT_NAME`
- `AZURE_OPENAI_API_VERSION`
- `AZURE_OPENAI_MODEL_NAME`
- `AZURE_OPENAI_SUBMODEL_NAME` (opcional)

## Dataset / Documentos
- Objetivo: ~1M tokens (aprox 4M caracteres).
- Entrada: varios .txt locales concatenados.
- Directorio sugerido: `data/` (gitignored).

## Criterios de Exito
- Responde correctamente preguntas con evidencia en fragmentos.
- No intenta cargar el documento completo en contexto.
- Logs muestran llamadas `python_exec` y slicing.

## Riesgos
- Coste elevado si el LLM entra en loops.
- Latencia por sub-llamadas secuenciales.
- Dependencia del comportamiento del modelo (tool calling).

## Roadmap (opcional)
- V1: UI web simple + export de reporte.
- V1.1: Soporte de indexado ligero (sin embeddings).
