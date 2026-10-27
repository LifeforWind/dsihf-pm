# DSIHF-PM development

- Start with README.md and docs/PROVENANCE.md; inspect the actual Git state before editing.
- This is a code-only repository. Do not commit real .env files, API keys, experiment results, requests/responses, caches, model weights, or upstream datasets.
- Public entry: scripts/run.py. Preserve every-step cap of 3 queries (clock and repeats count); no episode-wide observation cap. Keep bounded fault/call/cost protection separate.
- Non-Jev remote calls use a-mem/.env OPENAI_BASE_URL and OPENAI_API_KEY. OpenRouter is reserved for Jev; never use it as a non-Jev fallback. Local profiles require explicit selection.
- Keep upstreams pinned and pristine. Simulator/private groundtruth must not be passed to policies. Preserve native correction behavior; do not silently filter invalid model content or resample answers.
- Do not modify sibling original experiment worktrees or running services while working here.
- Run the offline tests for method changes; report skipped/incomplete checks honestly. Update source/version documentation when changing a frozen dependency or method boundary.
