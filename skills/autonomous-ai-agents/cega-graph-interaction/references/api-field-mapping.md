# CEGA REST API Field Mapping & Quirks

## Endpoint → Field Name Mapping (Live-Verified)

| Endpoint | Request Body Field | Common Mistake |
|----------|-------------------|---------------|
| `POST /query` | **`text`** (NOT `query`) | Sending `"query": "..."` fails with Field required |
| `POST /simulate` | **`hypothesis`** (NOT `question`) | Sending `"question": "..."` fails with Field required |
| `GET /health` | Response: `status`, `nodes`, `edges`, `facts`, `facts_by_status` | — |
| `GET /desires` | Response: `summary`, `desires[]` (each has fact/status/score/probe) | — |
| `POST /verify` | **`from_concept`, `relation`, `to_concept`** as separate fields | Sending free-text `"target": "..."` fails; must parse into triples |

## Response Structure Patterns

### `/query` response
```json
{
  "query": "...",
  "answer": "Navigator Swarm output with paths found...",
  "raw_path": "...",
  "confidence": 0.92,
  "validation": ""
}
```

The `answer` field contains the Navigator Swarm traversal results: path listings with relationship types (`[CAUSES]`, `[IS-A]`, `[ANALOGOUS-TO]`), confidence scores per edge, and walk statistics (total walks vs succeeded + duration).

### `/meditate` response
```json
{
  "insights": [{
    "seed": "cega_graph_limits",
    "held_as": "WISDOM_INSIGHT",
    "insight": "...",
    "mode": "reason | action",
    "keywords": [...]
  }],
  "count": N
}
```

Insights are tagged with `Held_As` types (typically `WISDOM_INSIGHT`). The meditation cycle can trigger **constitutional gates** — if harm is detected on navigation paths, a P0 gate blocks and the insight is held back. This is checked via `/desires` under constitution-triggered seeds.

### `/verify` response
```json
{
  "fact": "...",
  "verdict": "INCONCLUSIVE | VALIDATED | CONTRADICTED",
  "confidence": 0.XX,
  "reasoning": "...",
  "new_status": "uncertain | validated | contradicted",
  "sources": [{
    "source": "wikipedia | duckduckgo | llm",
    "found": true | false,
    "snippet": "..."
  }],
  "elapsed_ms": NNN
}
```

Verification is slow (5-8s typically). The `new_status` field indicates what the brain fact's epistemic status will become if accepted.

### `/simulate` response
```json
{
  "hypothesis": "...",
  "results": [{
    "answer": "Hypothetical reasoning result...",
    "confidence": 0.XX,
    "resolved": false,
    "candidates": [],
    "notes": "real_conf → hyp_conf comparison"
  }]
}
```

The `resolved` field indicates whether the hypothesis was resolved by existing graph knowledge. `candidates[]` is typically empty — hypothetical insights go to meditation instead of immediate crystallisation.

## Observed Behavior Notes

1. **Graph cache rebuild**: After deleting `core/graph_cache.pkl`, CEGA rebuilds in ~30-60s on next request
2. **Background loading**: On first startup, `/health` returns 503 until graph loads (~30-60s); subsequent calls share the same instance
3. **Fact statuses are sticky**: Facts stay `learned` unless explicitly re-train or verify changes them; meditation only creates new uncertain facts
4. **Constitutional gates**: Meditation may flag harm on paths and block those insights — check via `/desires` for constitution-triggered seeds with P0 gate warnings
