# Agent skills for external users

Skills here are packaged for organizations outside SkyTruth. Agents working in this repository do not load them; repo-local skills live in `.claude/skills`.

## cerulean-investigation

The skill uses two routes only: `curl` against the public TiPG API, or computer use on the Cerulean map.

| Path | Contents | Shipped |
|---|---|---|
| `cerulean-investigation/` | The skill: `SKILL.md` and `references/` (API, map, interpretation) | yes |
| `cerulean-investigation-maintenance/check_contract.py`, `expected_contract.json` | Live checks of the API behavior the skill documents, plus manual map checks | no |

### Build the zip

```bash
cd docs/agent-skills && zip -r -X /tmp/cerulean-investigation.zip cerulean-investigation -x '*.DS_Store'
```

### Check the live contract

```bash
python3 docs/agent-skills/cerulean-investigation-maintenance/check_contract.py
```

It makes about 20 anonymous requests (under 30 s), prints PASS or DIFF for each of 12 API checks, then lists 3 map checks to do by hand. Run it before shipping a zip and after any deploy that changes the TiPG allowlist (`stack/cloud_run_tipg.py`) or the `slick_plus` and `source_plus` views. When a check fails, update the skill text and `expected_contract.json` together.
