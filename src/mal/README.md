# Memory Adaptation Loop (MAL)

MAL tunes the gosh.memory pipeline for one `(key, agent_id)` binding
using that binding's own production failures as signal.

Disabled by default. When off, zero runtime cost and zero behavior change.

## Quick start

```
memory_mal_configure(key="proj", enabled=True)
memory_mal_feedback(key="proj", verdict="bad_answer", query="...",
                    runtime_trace_ref="ask_...", runtime_trace={...})
memory_mal_trigger(key="proj")
memory_mal_status(key="proj")
```

## What MAL can tune

| Atom type | What it changes |
|---|---|
| lexical_signal_bundle | word/number/entity phrase bonuses |
| locality_bundle | currentness bonus, generic penalty |
| window_bundle | supporting facts, budget |
| fusion_bundle | late fusion, rrf_k |
| inference_leaf_toggle | enable/disable leaf plugins |
| grouping_bundle | grouping mode, size cap |
| extraction_example_append | extraction prompts (all 3 planes) |

All consumed by production runtime. Zero dead state.

## CODE_REQUIRED

When failure is beyond tunable surface, MAL emits courier task to
`agent_id="coding"`. If no coding agent exists, task stays pending.

## Safety gates

- min_signals (default 10): minimum failures before any change
- Family support: max(2, ceil(sqrt(N))) on same-family
- A/B eval: failure slice + control group regression check
- Overfitting: optional 70/30 holdout split
- Convergence: 5 rejections → converged

## MCP tools (7)

memory_mal_configure, memory_mal_feedback, memory_mal_trigger,
memory_mal_status, memory_mal_list_feedback, memory_mal_get_artifact,
memory_mal_rollback
