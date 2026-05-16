# SAGE Software Architecture

This document records the main runtime structure of SAGE and the engineering
contract used by the built-in coding agent.

## Runtime Layers

1. Web UI
   - Main pages: Chat, Knowledge, Skills, LLM Settings, Scientific Analysis Agent,
     and Parameter Optimization Agent.
   - Chat requests are routed by an LLM router into QA, RAG, code draft, or
     executable CodeEngine jobs.

2. Agent Orchestration
   - Flask routes under `web_app/routes/` manage background jobs, streaming,
     uploaded documents, project context, and cancellation.
   - Long-running CodeEngine and workflow jobs are stored in process-level job
     registries and polled by the web UI.

3. Retrieval and Skills
   - `web_app/rag_engine.py` provides persistent project knowledge retrieval.
   - `seismo_skill/skill_loader.py` retrieves built-in, user, and generated
     OpenAI-style SKILLs.
   - Chat, Scientific Analysis, Parameter Optimization, and CodeEngine share
     SKILL/RAG context so generated answers and code use the same domain APIs.

4. CodeEngine
   - `seismo_code/code_engine.py` is the built-in execution and debugging core.
   - It supports standalone scientific scripts and repository-editing tasks.
   - It generates Python/Bash/GMT-oriented scripts, runs them in a kept
     execution directory, checks outputs, and performs multi-round debugging.

5. Domain Tooling
   - `seismo_stats/` contains local statistical APIs such as b-value and
     Gutenberg-Richter plotting.
   - `seismo_code/toolkit.py` exposes waveform, plotting, travel-time, and
     source-parameter helper functions.
   - Skill-local packages such as `seismo_skill/skills/pnsn_phase_detection/`
     provide specialized workflows and APIs.

## Built-in Coding Agent Flow

For executable coding requests, CodeEngine follows this sequence:

1. Reconnaissance
   - Profile explicit data files.
   - For repository tasks, build a compact repo map from `rg --files`, targeted
     `rg` hits, symbols, routes, ranked files, and snippets.
   - Build local API references by introspecting SAGE modules when the task
     likely uses local APIs.

2. Engineering plan
   - Ask the LLM for a broad-to-detailed `## Engineering Plan`.
   - The plan must include route, files, API details, unit tests, and validation.
   - The plan is injected into code generation as a design contract.

3. Plan persistence
   - The initial plan is written to `engineering_plan.md` inside the execution
     directory.
   - Debug rounds load this persisted file before generating a fix.
   - When a debug round changes the design, CodeEngine writes
     `engineering_plan_debug_round_<N>.md`.

4. Implementation
   - For repository edits, the generated script is an edit-and-test driver.
   - It prints `[SAGE_AGENT] located <path>: <reason>` for selected files.
   - It prints `[SAGE_CHANGED] <path>` for every changed file.
   - It uses exact old-block to new-block replacements or small structured edits.

5. Unit tests and validation
   - Python behavior changes must add or update focused tests, or locate and run
     existing focused tests related to the changed API.
   - CodeEngine runs `py_compile` for changed Python files.
   - CodeEngine runs targeted `pytest` for changed or related tests.
   - If no focused tests are changed or found for a Python behavior change,
     validation fails and the debug loop must add or locate tests.

6. Debug loop
   - Runtime errors, failed mini-tests, py_compile failures, pytest failures, and
     missing-test validation failures are all treated as real bugs.
   - The debugger receives the same SKILL docs, local API reference, RAG snippets,
     and persisted engineering plan used during generation.
   - The fixed script is rerun and revalidated.

7. Artifacts
   - `analysis.py` or `analysis.sh`: final generated driver script.
   - `engineering_plan.md`: initial design/API/test plan.
   - `engineering_plan_debug_round_<N>.md`: debug-time plan revisions.
   - Generated figures, tables, logs, and output files.

8. Artifact lifecycle
   - Serialized CodeEngine results include `artifact_paths` and `exec_dir`.
   - Conversation records keep these paths so cleanup can happen later.
   - Deleting a conversation deletes safe CodeEngine temporary execution
     directories and persisted `engineering_plan*.md` files for that
     conversation.
   - Deleting a project deletes each project conversation first, so the same
     cleanup path removes its related coding-plan files.
   - Cleanup is conservative: files written into normal project/user data paths
     are not blindly deleted unless they are inside a recognized CodeEngine temp
     run directory or the managed plan cache.

## Important Source Files

- `web_app/routes/code.py`: CodeEngine job API and polling.
- `web_app/routes/chat.py`: chat/RAG/plain/code-draft builders and sources.
- `web_app/templates/chat.html`: chat UI, code blocks, copy buttons, job polling.
- `seismo_code/code_engine.py`: planning, generation, execution, debug, validation.
- `seismo_code/ce_prompts.py`: codegen, debug, verify, and engineering-plan prompts.
- `seismo_code/repo_intelligence.py`: repository map and editing discipline.
- `seismo_code/safe_executor.py`: subprocess execution and artifact collection.
- `seismo_skill/skill_loader.py`: SKILL discovery, matching, references, and RAG.
- `seismo_stats/`: local statistics APIs used by generated code.

## Runtime Directories

- `.seismicx/`: project-local application state and backend/runtime settings.
- `.sage_runtime/`: web service logs, PID files, and local runtime diagnostics.
- CodeEngine execution directories: temporary kept directories containing scripts,
  plan files, debug revisions, and generated artifacts.
