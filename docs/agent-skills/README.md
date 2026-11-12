# Agent skills for external users

Skills here are packaged for organizations outside SkyTruth. Agents working in this repository do not load them; repo-local skills live in `.claude/skills`.

## cerulean-investigation

The skill uses two routes only: `curl` against the public TiPG API, or computer use on the Cerulean map.

### Hosted publication

TiPG renders the references from this directory as a browser-readable API guide
at `https://api.cerulean.skytruth.org/guide`, for humans and automated clients.
It publishes the original Markdown at `/skills/cerulean-investigation/`. Agents
can start at `https://api.cerulean.skytruth.org/llms.txt` for the short entry guide and task links.
The service derives its entry guide's evidence rules from `SKILL.md`, serves the
Markdown references individually, and generates an installable `.tar.gz` archive
and SHA-256 discovery index at `/.well-known/agent-skills/index.json`. Artifacts
are cached per process and support ETags; no separately maintained documentation
copy or manual packaging step is needed.

The TiPG Docker build includes only this distributable skill directory from
`docs/agent-skills`. HTTP `Link` headers advertise the guide across endpoints.
The landing page's Get started section, shared HTML navigation and callout,
OGC collection metadata and OpenAPI also link to the browser guide. Feature,
CSV and tile response bodies are not rewritten. The maintenance checks below
are not published.

The [API notebook](../../notebooks/Cerulean%20API%20Guide.ipynb) links to the browser
guide and hosted skill. The README opens that checked-in notebook through GitHub-backed Colab,
so updates follow `main` rather than requiring a separate Drive copy. Existing
CI builds and deploys staging after a merge to `main`; production still requires
a release tag.

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

It makes 22 anonymous requests (about 30 s), each tagged `utm_medium=agentic` and `utm_source=skytruth-contract-check`, prints PASS or DIFF for each of 15 API checks, then lists 3 map checks to do by hand. Run it before shipping a zip and after any deploy that changes the TiPG allowlist (`stack/cloud_run_tipg.py`) or the `slick_plus` and `source_plus` views. When a check fails, update the skill text and `expected_contract.json` together.
