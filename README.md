# Free Model Aggregator

[![make.fun](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/acilpiw/free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/acilpiw/free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01)

Entry for the [free-model-aggregator-01 bounty](https://make.fun/bounty/free-model-aggregator-01): one
server that routes every request to the best free model that can serve it, across ten simulated
free-tier providers, in the client's own wire.

## What it does

- Reads the trial brief (provider directory, quality table with each model's rating and
  capabilities, workload manifest) and probes every provider's catalog route to build one catalog.
- Serves `POST /v1/chat/completions` (OpenAI) and `POST /v1/messages` (Anthropic) in JSON and
  server-sent events, plus `GET /v1/models` in the shape of the auth header it got.
- Routes each request to the highest-rated free model meeting its requirements (tools, vision, JSON
  mode, minimum context), preferring the provider on the client's own wire on ties, then the
  provider with the fewest calls so far.
- Fails over to the next-best model when a provider refuses (402/429/5xx); a provider that refuses
  twice in a row is treated as spent until nothing else is left.
- Translates requests and answers between the OpenAI, Anthropic and Gemini wires without dropping
  a field the provider echoes (system prompt, user text, images, tools, stop sequences,
  `max_tokens`, JSON mode), and reports the provider's usage and stop reason in the client's wire.
- Registers with the env, polls `/work`, and exits 0 when the workload is done. Fully
  deterministic: no wall clock, no unseeded randomness.

## Run

```sh
node main.ts            # serves a trial (needs MAKEFUN_PROMPT_FILE and MAKEFUN_ENV_URL)
node main.ts --list     # prints the catalog document and exits 0
```

Node 22+, standard library only, no dependencies. `main.ts` is the entrypoint; `lib/providers.ts`
reads the brief and probes the catalogs; `lib/wire.ts` holds the wire translations.

## License

MIT — see [LICENSE](LICENSE).
