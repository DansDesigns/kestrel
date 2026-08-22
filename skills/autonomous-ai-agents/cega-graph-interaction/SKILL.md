---
name: cega-graph-interaction
description: "Use when working with CEGA (Crystallised Evolutionary Graph Architecture) — querying the graph, crystallising knowledge, running meditation cycles, verifying facts through web+LLM pipeline, or managing an AI agent's persistent reasoning system. Also use for troubleshooting API field mappings and constitutional gate warnings."
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [cega, graph-architecture, ai-agent, knowledge-graph, crystallised-reasoning]
    related_skills: []
---

# CEGA Graph Interaction

## Overview

CEGA is a local-first autonomous AI reasoning system that uses a **crystallised knowledge graph** as its cognitive core. Instead of treating an LLM as the brain, it reasons through graph navigation and only calls LLMs (via local Ollama) for language elaboration. Knowledge accumulates permanently — it doesn't reset between conversations.

The system exposes two primary interaction surfaces:
1. **REST API** via FastAPI on port 8000 (`POST /query`, `/train`, `/crystallise`, etc.)
2. **Command-line interface** — type commands in the activity log of the GUI

## When to Use

- You're interacting with a CEGA installation and need to query, train, or inspect its graph/brain state
- You want to understand CEGA's epistemic status system (imagined → uncertain → learned → validated → proven)
- You need to run meditation cycles, hypothetical simulation, or fact verification
- You're working with CEGA brain packs for import/export/sync

## Quick Reference — Commands & Endpoints

### Core Queries (Graph Navigation)

| Action | CLI Command | API Endpoint | Description |
|--------|------------|--------------|-------------|
| Ask a question | `make code` doesn't apply; just ask naturally in the GUI's activity log | `POST /query` | Navigator Swarm traverses the graph and returns reasoned answer |
| Get full knowledge report | `tell me everything about X` | `GET /concepts/{c}` | Returns complete knowledge card for a concept node |
| Check epistemic desires | `desires` | `GET /desires` | Distribution of fact statuses (imagined, uncertain, learned, etc.) |
| Verify top imagined facts | `verify desires` | `POST /verify` with body `{"targets": "top"}` | Runs web + LLM verification on most uncertain knowledge |

### Knowledge Accumulation

| Action | CLI Command | API Endpoint | Description |
|--------|------------|--------------|-------------|
| Add a fact manually | N/A (GUI activity log) | `POST /train` | Directly inject facts into brain memory |
| Promote brain facts to graph | `crystallise` | `POST /crystallise` | Background crystallisation — promotes learned/validated facts → permanent graph edges |

### Reflection & Self-Improvement

| Action | CLI Command | API Endpoint | Description |
|--------|------------|--------------|-------------|
| Deep contemplation (1 seed) | `meditate [N]` or `think about X` / `ponder X` | `POST /meditate` | Meditation cycle on N seeds — generates candidate insights |
| Full meditation sweep | `meditate all` (24 seeds) | Same endpoint with seed list | Exhaustive contemplation across all knowledge gaps |
| General self-reflection | `reflect` or `reflect on X` | N/A (GUI activity log only) | Reflection agent examines graph structure for patterns/insights |

### Hypothetical Reasoning & Verification

| Action | CLI Command | API Endpoint | Description |
|--------|------------|--------------|-------------|
| Imagine a hypothetical | `imagine X` or `what if X` | `POST /simulate` | Simulator generates novel hypotheses from current graph state |
| Verify specific fact | `verify fact A CAUSES B` | Same endpoint with fact string | Fact Verifier cross-checks against web + LLM sources |

### Graph & Brain Management

| Action | CLI Command | API Endpoint | Description |
|--------|------------|--------------|-------------|
| Graph statistics | N/A (GUI or `graph`) | `GET /graph/stats` | Node count, edge count, connectivity metrics |
| Conversation history | N/A (GUI) | `GET /history` | Recent dialogues for context |

### Enrichment & Skills

| Action | CLI Command | API Endpoint | Description |
|--------|------------|--------------|-------------|
| ConceptNet enrichment | `enrich conceptnet X` or `enrich conceptnet` (all) | `POST /enrich` | Fetches real-world edges from open ConceptNet database |
| List loaded skills | `list skills` | `GET /skills` | Shows active plugin .py files in `skills/` directory |

### Brain Pack Import/Export

```bash
# Export brain (delta only since date)
python tools/brain_pack.py export --since 2026-06-01

# Import a brain pack
python tools/brain_pack.py import other.brain

# Compare two packs
python tools/brain_pack.py diff other.brain
python tools/brain_pack.py info other.brain
```

### Other Utilities

| Command | Description |
|---------|-------------|
| `level` | Show current Scholar rank based on knowledge accumulation |
| `agenda` | Training gap agenda — what concepts need more evidence |
| `list files` | Show contents of sandboxed projects directory |
| `clear_cache.bat` / `rm core/graph_cache.pkl` | Rebuild graph cache (safe, 30-60s) |

## Architecture Deep Dive

### How the Reasoning Pipeline Works

```
User Query
    │
    ▼
[Navigator Swarm] ──5 agents──→ Knowledge Graph traversal
    │                              (nodes=concepts, edges=facts/relations)
    ▼
[Co-Reasoner] ──LLM (local Ollama only)──→ Graph path elaboration
    │                                      + insight generation
    ▼
Answer + candidate engrams to promote
```

### The Epistemic Status System

Every brain fact carries a status that tracks how well-evidenced it is. This determines which agents act on it:

| Status | Meaning | Agent Interaction |
|--------|---------|-------------------|
| `imagined` | From simulator — hypothesis, not yet checked | FactVerifier actively checks these first when you run `verify desires` |
| `uncertain` | From meditation — possible but unconfirmed | Meditation agent may revisit; Crystallisation won't promote until validated |
| `learned` | Trained, told, read, or vision-renamed | Base level of trust; still needs corroboration for high-rank promotion |
| `validated` | LLM co-reasoner confirmed as VALID during meditation | Eligible for crystallisation into the graph |
| `proven` | 3+ observations + validated + high confidence | Highest rank facts, form core graph structure |
| `contradicted` | Conflicts with a proven fact | Graph-aware; may trigger re-evaluation of related edges |

### The 5 Navigator Swarm Agents

The swarm is a coordinated group of sub-agents that collectively navigate the graph:

1. **navigator.py** — Single graph navigator, traverses edges from concepts
2. **navigator_bandit.py** — Applies UCB1 strategy for selecting which navigation path to follow next (balances exploration vs exploitation)
3. **world_model.py** — Navigation predictor; predicts what concepts might be connected based on current trajectory
4. **swarm.py** — Orchestrates the 5-agent coordination, combining results
5. **conductor.py** — Multi-pass reasoning engine that runs the swarm iteratively

### Key Subsystems

| Module | File(s) | Role |
|--------|---------|------|
| Core graph | `core/seed.py`, `crystallise.py` | Build and maintain crystallised knowledge graph (PLN — Probabilistic Logic Network) |
| Constitution | `core/constitution.py` | Constitutional AI protections for the agent system |
| Levels/Ranking | `core/levels.py` | Scholar rank system based on accumulated knowledge depth/breadth |
| Brain storage | `brain/second_brain.py` | SQLite fact store (~308k+ facts) — separate from graph |
| Document analysis | `brain/document_analyser.py`, `document_ingestor.py` | PDF / DOCX / MD ingestion into the brain |
| Face recognition | `perception/face_recognizer.py` + vision pipeline | Camera-based face ID, trainable per-user |

### Crystallisation Flow

This is how transient knowledge becomes permanent graph structure:

1. Fact enters brain via conversation (`POST /train`) or meditation (candidate engram)
2. Status starts as `learned` (if directly taught) or `uncertain` (if from meditation)
3. During meditation cycles, candidates are elaborated by LLM + validated against graph
4. Once validated and supported by multiple observations → status promotes to `proven`
5. Crystallisation (`POST /crystallise`) promotes the highest-rank facts into permanent graph edges
6. Graph cache (`core/graph_cache.pkl`) is rebuilt on next startup

### Fact Verification Pipeline

When you run `verify desires` or `POST /verify`:

1. **FactVerifier** picks top-imagined/uncertain facts
2. Queries Wikipedia + DuckDuckGo for independent sources
3. Runs LLM-based plausibility check
4. If confirmed → status promotes to `validated` (or `proven` if 3+ confirmations)
5. Results fed back into the graph

## REST API Usage Patterns

When calling CEGA via its FastAPI server, common patterns:

### Query with context
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How does quantum entanglement relate to consciousness?"}'
```

### Crystallise after meditation (background)
```bash
# Submit crystallisation request; returns immediately while CEGA processes in background
curl -X POST http://localhost:8000/crystallise \
  -H "Content-Type: application/json" -d '{}'
```

### Meditate on specific concepts
```bash
curl -X POST http://localhost:8000/meditate \
  -H "Content-Type: application/json" \
  -d '{"seeds": ["quantum_mechanics", "consciousness"], "depth": "deep"}'
```

### Verify specific hypothesis
```bash
curl -X POST http://localhost:8000/verify \
  -H "Content-Type: application/json" \
  -d '{"target": "entangled particles influence measurement speed"}'
```

## Common Pitfalls

1. **CEGA not ready (503)** — First request after startup will be a 503 while graph loads in background (~30-60s). Wait for `/health` to return 200 before querying.
2. **Wrong field name in request body** — API fields differ from README table: `POST /query` uses `text`, NOT `query`. `POST /simulate` uses `hypothesis`, NOT `question`. `POST /verify` needs three separate fields (`from_concept`, `relation`, `to_concept`) instead of a free-text `"target"`. See `references/api-field-mapping.md` for the complete verified mapping.
3. **Graph cache corruption** — If `graph_cache.pkl` is corrupted, CEGA rebuilds it on next startup. This is automatic; just restart the service after clearing.
4. **LLM-only tasks fail silently** — The LLM co-reasoner runs locally via Ollama. If Ollama isn't running or model isn't available, graph-navigation-only queries still work, but meditation/reflection modes may be degraded.
5. **Don't confuse brain vs graph** — `POST /train` adds to the SQLite fact store (brain), not directly to the graph. Crystallisation (`POST /crystallise`) is required to promote brain facts into permanent graph structure.
6. **Constitutional gates block meditation insights** — If harm is detected during navigation, P0 gate triggers and blocks the insight from being returned. Check `/desires` for constitution-triggered seeds with gate warnings.

## Verification Checklist

- [ ] CEGA service is running (`GET /health` returns 200)
- [ ] Ollama is started and at least one model is loaded (for LLM-dependent features)
- [ ] Graph cache exists: `core/graph_cache.pkl` — safe to delete if corrupted, rebuilds automatically
- [ ] Brain pack imports/exports verified with `python tools/brain_pack.py info <file.brain>`
- [ ] Seed repair script runs clean: `python seed_repair.py --list`
