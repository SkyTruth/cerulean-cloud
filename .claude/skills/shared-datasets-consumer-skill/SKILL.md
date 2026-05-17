---
name: shared-datasets-consumer
description: >-
  Use when integrating shared-datasets-1 into the current SkyTruth consumer repo
  or app, including catalog.json-driven layer/search/config/citation/release
  metadata, replacing direct GCS or hardcoded PMTiles URLs, using access_tier
  and tiered tiles.skytruth.org PMTiles URLs, adding signed-cookie private
  PMTiles access, installing or using the Python SDK for backend GCS fetch or
  resolve flows, preserving DatasetRef.resolved_id lineage, or preparing small
  focused consumer adoption PRs.
---

# Shared Datasets Consumer Integration

Use this skill when the current SkyTruth consumer repo or app needs to adopt
shared-datasets-1 with the smallest safe change. The main job is adoption, not
redesign: find hardcoded shared dataset access paths, replace them with the
stable shared access contract, and package each consumer change as a small
focused PR.

Before changing this consumer repo or app, re-check the current shared catalog
and the upstream shared-datasets-1 Terraform/docs if the change depends on a
specific asset list, access tier, browser origin, service account, or signing
grant. Treat the values below as the verified upstream repo/live state at the
time this skill was last reviewed, not as a replacement for catalog-driven
behavior.

Terminology in this skill:

- "this repo" or "this app" means the current workspace being changed.
- "upstream shared-datasets-1 repo" means the separate shared catalog,
  SDK/docs, and Terraform control-plane repository.

## Shared-Datasets Sources

Use these upstream shared-datasets-1 sources from this consumer repo or app:

```text
Canonical repo: https://github.com/SkyTruth/shared-datasets-1
Machine catalog: https://tiles.skytruth.org/_catalog/web/catalog.json
Catalog CSV: https://tiles.skytruth.org/_catalog/shared-datasets-catalog.csv
Consumer guide: docs/consumer-guide.md
PMTiles CDN details: docs/pmtiles-cdn.md
Python SDK README: api/python/README.md
Terraform config: terraform/envs/prod/variables.tf and terraform/envs/prod/production.auto.tfvars
```

If a local checkout of the upstream shared-datasets-1 repo is available, prefer
its current files.
Otherwise use the GitHub paths above plus the public catalog URLs. Do not guess
current asset access tiers, browser CORS origins, reader service accounts, or
signer grants from memory.

## Adoption Workflow

Use this workflow when moving this repo or app off direct shared bucket URLs or
adding shared-datasets catalog access.

1. Scan this repo or app for hardcoded shared dataset URLs, PMTiles paths,
   dataset slugs, and ad hoc GCS access.
2. Classify each hit by runtime surface:
   - Browser map PMTiles should use tiered `tiles.skytruth.org` URLs.
   - Private browser PMTiles need a backend signed-cookie session endpoint.
   - Browser or service catalog discovery should treat
     `https://tiles.skytruth.org/_catalog/web/catalog.json` as the critical
     machine-readable source of truth, either fetched directly from the public
     catalog route or served through an app-owned backend/config API.
   - Backend/server downloads or authenticated catalog resolution should use
     the Python SDK.
   - Unrelated references, docs, tests, and comments should only change when
     they would otherwise keep the old integration pattern alive.
3. Make the smallest coherent replacement. Prefer a tiny helper such as
   `sharedPmtilesUrl("<slug>", "<access-tier>")` over broad map-layer
   refactors.
4. Keep PRs focused by surface and dataset. Do not combine unrelated UI,
   infrastructure, dependency, or formatting changes with adoption work.
5. In the PR description, call out the old hardcoded access path, the new shared
   access path, and the focused validation that was run.

## Recommended Path

Start here. Do not add more machinery than the consumer actually needs.

1. For browser map layers that already know the PMTiles slug and only use public
   assets, use the public tiered URL template directly. Do not install the
   Python SDK.
2. For backend/server code that needs to download or resolve shared dataset
   files, install the Python SDK with the `gcs` extra and use
   `fetch_dataset(...)` or `resolve_dataset(...)`. A successful
   `fetch_dataset(...)` returns a `DatasetRef`; use `ref.cache_path` as the
   local file path and `ref.resolved_id` as the durable resolved identity to
   record when callers request `version="latest"`. When both bytes and lineage
   are needed, call `fetch_dataset(...)` once; do not call
   `resolve_dataset(...)` and `fetch_dataset(...)` separately.
3. Public catalog reads should use
   `https://tiles.skytruth.org/_catalog/shared-datasets-catalog.csv` or
   `https://tiles.skytruth.org/_catalog/web/catalog.json`, not anonymous direct
   `storage.googleapis.com` bucket URLs. Prefer `catalog.json` for browser and
   service layer configuration because it is the machine-readable source of
   truth for PMTiles URLs, access tiers, docs URLs, release metadata, bounds,
   and geometry metadata.
4. For credentials, prefer runtime identity: Cloud Run, jobs, or CI should run
   as an established runtime or reader service account that already has bucket
   read access. Do not create service account JSON keys.
5. For browser map layers that may use private shared-dataset PMTiles, resolve
   the PMTiles URL from the shared catalog `access_tier`, add a signed-cookie
   session endpoint in this app's backend, and make PMTiles range requests
   with browser credentials included.

## Catalog Discovery

Use this when this repo or app needs search, layer config, citations, schema
notes, or release metadata.

Public catalog reads use:

```text
https://tiles.skytruth.org/_catalog/shared-datasets-catalog.csv
https://tiles.skytruth.org/_catalog/web/catalog.json
https://tiles.skytruth.org/_catalog/web/index.html
https://tiles.skytruth.org/_catalog/web/docs/assets/{asset-slug}.md
https://tiles.skytruth.org/_catalog/releases/{asset-slug}.json
```

`https://tiles.skytruth.org/_catalog/web/catalog.json` is the critical
machine-readable source of truth for this repo's browser and service config. Use
it when generating layer lists, PMTiles URLs, access decisions, docs links,
release metadata, bounds, or geometry metadata. Use the CSV when a backend or
SDK workflow specifically needs the tabular catalog contract.

Minimum `catalog.json` contract for consumers:

- Read the top-level `assets` array; do not infer assets from bucket paths.
- For layer/config rows, use `slug`, `title`, `status`, `access_tier`,
  `available_formats`, `canonical_format`, `has_pmtiles`, `pmtiles_url`,
  `docs_url`, `release_index_url`, `latest_release`, `last_updated`,
  `citation`, `license`, `bounds`, and `geometry_type` when present.
- Use `pmtiles_url` for browser PMTiles. It is already the tiered
  `tiles.skytruth.org` URL. Only reconstruct a URL when `pmtiles_url` is absent
  and the app has explicitly validated `access_tier` and `slug`.
- Resolve relative `docs_url` and `release_index_url` values against
  `https://tiles.skytruth.org/_catalog/web/`.
- Use only `status="active"` for default production layer lists. If a UI
  intentionally shows `deprecated`, `superseded`, or `retired` assets, display
  `consumer_guidance`.
- Reject missing or unknown `access_tier` values. Do not silently default an
  asset to public.
- Treat `latest_release.date` as the freshest release date when present;
  otherwise use `last_updated`.

Do not build new behavior in this repo or app on anonymous direct
`https://storage.googleapis.com/skytruth-shared-datasets-1/_catalog/...` reads.
Those direct bucket URLs are diagnostic or transitional only. Browser apps
should use `tiles.skytruth.org/_catalog/`, a docs site, or an app-owned config
API. Backend code that needs authenticated catalog resolution should use the SDK
with Application Default Credentials.

When generating app config from the catalog, preserve `access_tier`,
`citation`, `license`, and `last_updated` where the UI or API displays
provenance. Reject missing or unknown `access_tier` values instead of silently
defaulting private assets to public.

## Frontend PMTiles

Use this when this repo or app needs browser PMTiles and already knows the
asset slug, or can resolve the slug from the shared catalog.

Use these URL shapes:

```text
https://tiles.skytruth.org/pmtiles/public/{slug}.pmtiles
https://tiles.skytruth.org/pmtiles/private/{slug}.pmtiles
```

Example:

```text
https://tiles.skytruth.org/pmtiles/public/wdpa-marine.pmtiles
https://tiles.skytruth.org/pmtiles/private/iucn-mammal-ranges.pmtiles
```

If the repo only uses known public assets, a tiny helper is enough:

```ts
const SHARED_PMTILES_BASE_URL =
  process.env.NEXT_PUBLIC_SHARED_PMTILES_BASE_URL ??
  "https://tiles.skytruth.org/pmtiles";

export function sharedPmtilesUrl(
  assetSlug: string,
  accessTier: "public" | "private" = "public"
) {
  return `${SHARED_PMTILES_BASE_URL}/${accessTier}/${assetSlug}.pmtiles`;
}
```

Replace direct GCS PMTiles URLs like:

```text
https://storage.googleapis.com/skytruth-shared-datasets-1/.../latest/wdpa-marine.pmtiles
```

with:

```ts
sharedPmtilesUrl("wdpa-marine", "public")
```

The access tier is part of the stable URL contract. Use the same
`tiles.skytruth.org` URL in redirect mode and CDN mode; do not switch consumers
back to `storage.googleapis.com` for CDN rollout or fallback behavior.

Do not assume every PMTiles asset is public. The live catalog should be the
source of truth. Verified against the checked-in and live catalog on
2026-05-17, the private PMTiles fixtures are
`acled-europe-central-asia-aggregated-weekly-admin1`,
`acled-middle-east-aggregated-weekly-admin1`, `global-coral-reefs`,
`iucn-mammal-ranges`, and `iucn-reptile-ranges`. Use them as test fixtures, not
as a hardcoded production allowlist. If a consumer derives layers from the
catalog, preserve `citation` in service-facing metadata and parse `access_tier`
to emit:

```ts
`${SHARED_PMTILES_BASE_URL}/${accessTier}/${assetSlug}.pmtiles`
```

When a config API caches shared-dataset layer config, bump the cache key after
changing URL generation so stale direct GCS PMTiles URLs are flushed.

The current production mode is CDN mode. Successful PMTiles reads should stay
on `tiles.skytruth.org` and return normal object or range responses such as
`200` or `206`. Public-tier CDN access does not require signed cookies. Direct
`storage.googleapis.com` PMTiles URLs are not a production fallback for shared
dataset browser layers.

Redirect mode is retained as historical rollback machinery only. In that mode,
public PMTiles may be served by a temporary `307` redirect to public GCS, but
consumers should still keep the stable tiered `tiles.skytruth.org` URL.

## Private PMTiles Signed-Cookie Pattern

Use this pattern when this repo or app may display private shared
PMTiles. Before implementing or reviewing private PMTiles, read
[Private PMTiles Signed Cookies](references/private-pmtiles-signed-cookies.md).

Minimum consumer contract:

- Public PMTiles use `pmtiles_url` directly and do not need a cookie.
- Private PMTiles use the same `pmtiles_url` shape only after the consumer
  backend grants a browser session.
- This app's backend exposes a route such as
  `GET /api/pmtiles/session?tier=private`, authenticates and authorizes the
  user, signs the private PMTiles prefix, sets `Cloud-CDN-Cookie`, and returns
  `Cache-Control: no-store`.
- The signer is the service account running the cookie endpoint, not
  necessarily the reader service account used for backend data downloads.
- Every PMTiles byte-range request for private layers must include
  `credentials: "include"` so the browser sends the cross-site cookie to
  `tiles.skytruth.org`.
- Browser code must never receive GCS credentials, service account keys, raw
  signing key bytes, or full signed cookie values.

Upstream shared-datasets-1 infrastructure changes are separate from this repo's
PR. For a new private-PMTiles environment, open an upstream
shared-datasets-1 PR that adds the exact browser origin to
`pmtiles_cdn_allowed_origins` and grants this app's endpoint runtime service
account access to the signing-key secret. Let the protected shared-datasets-1
production workflow apply after merge; do not run a local
production Terraform apply.

Roll out in this order: replace direct PMTiles URLs with catalog-derived
`pmtiles_url`, bump any layer-config cache key, add the session endpoint, add
credentialed PMTiles range fetches, call the session endpoint before mounting
private layers, verify CSP/CORS, then run this repo's lint, type, test,
and build checks.

## WDPA MPA Identity

For WDPA MPA features and new shared-dataset joins, use `site_id` / source
`SITE_ID` as the durable feature identity. Do not build new behavior around
`WDPAID`. Do not rewrite legacy backfills or tables that only contain `WDPAID`
unless an explicit alias or backfill plan exists.

## Backend Data

Use this when backend/server code needs the actual data file, a canonical
`gs://` URI, or catalog-driven format resolution.

If this project has backend/server code and a runtime service account,
install the SDK with the GCS extra. The package is installed from GitHub for
now, not PyPI. `SkyTruth/shared-datasets-1` is public, so unauthenticated public
HTTPS installs are acceptable:

```bash
pip install "skytruth-shared-datasets[gcs] @ git+https://github.com/SkyTruth/shared-datasets-1.git@main#subdirectory=api/python"
```

For production, pin a tag or commit SHA instead of `main`. Before committing a
GitHub dependency URL, verify the exact install surface can reach it. Docker
builds and GitHub Actions may use different requirement files and may lack
`git`, SSH, or cross-repo token access; do not treat a local checkout or local
editable install as proof the committed requirement works.

Use `git+https` when the install surface has `git`. If the runtime installer
lacks `git`, a pinned public archive URL is acceptable:

```bash
pip install "skytruth-shared-datasets[gcs] @ https://github.com/SkyTruth/shared-datasets-1/archive/<tag-or-sha>.zip#subdirectory=api/python"
```

Do not use unauthenticated `archive/<sha>.zip` URLs for private repos or future
private forks unless runtime authentication is explicitly wired.

Then fetch a dataset with one call:

```python
from skytruth_shared_datasets import fetch_dataset

ref = fetch_dataset("wdpa-marine", "fgb")
path = ref.cache_path
resolved_id = ref.resolved_id
```

For AOI joins, job records, lineage tables, or other durable references, record
`resolved_id` values such as `wdpa-marine@2026-05-02`, not
`wdpa-marine@latest` and not a value inferred from the cache path.

When both bytes and lineage are needed, call `fetch_dataset(...)` once and use
`ref.cache_path` plus `ref.resolved_id`; do not call `resolve_dataset(...)` and
`fetch_dataset(...)` separately.

Or resolve without downloading:

```python
from skytruth_shared_datasets import resolve_dataset

ref = resolve_dataset("wdpa-marine", "pmtiles")
print(ref.gs_uri)
print(ref.url)
print(ref.resolved_id)
```

This path uses Application Default Credentials. In Cloud Run, scheduled jobs, or
CI with Workload Identity Federation, the established runtime service account
should be enough.

The only required IAM setup is:

```text
Grant this repo/app runtime service account `roles/storage.objectViewer` on
`gs://skytruth-shared-datasets-1`.
```

Prefer a repo-provisioned reader service account when that is this repo's
established deployment model:

```text
Cerulean:     shared-datasets-reader@cerulean-338116.iam.gserviceaccount.com
30x30:        shared-datasets-reader@x30-399415.iam.gserviceaccount.com
SkyTruthTech: shared-datasets-reader@skytruth-tech.iam.gserviceaccount.com
```

Those three accounts are the current production reader accounts verified in
Terraform and live IAM on 2026-05-17. Do not assume a Monitor reader account
exists; if Monitor or another project needs a repo-provisioned reader, add it
through the upstream shared-datasets-1 Terraform review path or grant the
existing runtime identity instead.

Otherwise grant the existing runtime service account `roles/storage.objectViewer`
on the shared bucket. Run backend jobs/services as the chosen runtime identity,
then use `fetch_dataset(...)` or `resolve_dataset(...)`. Do not treat
`fetch_dataset(...)` as a path-only helper; it returns the resolved reference
that also carries the populated cache path.

For Cloud Run, that means the service configuration names the existing runtime
service account or the repo-provisioned reader service account. The Python code
stays credential-free:

```bash
gcloud run services update SERVICE_NAME \
  --project=PROJECT_ID \
  --region=REGION \
  --service-account=RUNTIME_SERVICE_ACCOUNT_EMAIL
```

```python
from skytruth_shared_datasets import fetch_dataset

ref = fetch_dataset("wdpa-marine", "fgb")
path = ref.cache_path
resolved_id = ref.resolved_id
```

Only pass a custom `google.cloud.storage.Client` if the repo already has a
deliberate ADC or service-account impersonation pattern. Do not add service
account key files.

## Search Targets

Look for:

```text
storage.googleapis.com/skytruth-shared-datasets-1
gs://skytruth-shared-datasets-1
tiles.skytruth.org/pmtiles/
tiles.skytruth.org/_catalog/
catalog.json
shared-datasets-catalog.csv
.pmtiles
access_tier
Cloud-CDN-Cookie
pmtiles/session
skytruth_shared_datasets
skytruth-shared-datasets
wdpa
WDPAID
```

Prefer replacing only the PMTiles access path first. Do not refactor unrelated
map-layer behavior.

## Minimum Patch Checklist

- Apply only the checklist items for surfaces that exist in this repo or app.
  Backend-only adopters should not add PMTiles helpers, frontend env vars, or
  browser URL changes.
- Add a configurable PMTiles base URL with default
  `https://tiles.skytruth.org/pmtiles`.
- Replace direct `storage.googleapis.com` PMTiles sources with
  `sharedPmtilesUrl("<slug>", "<access-tier>")` or an equivalent catalog-driven
  tiered URL.
- If the repo reads the shared catalog, parse `access_tier` and do not preserve
  old direct GCS `pmtiles_url` overrides for shared-dataset layers. Use
  `catalog.json` as the machine-readable source for browser/service config.
- If the repo reads public catalog files, use `https://tiles.skytruth.org/_catalog/`
  or an app-owned backend/config API rather than direct
  `storage.googleapis.com` bucket URLs.
- Bump any config cache key that could retain old PMTiles URLs.
- Keep PMTiles asset slugs lowercase kebab-case.
- If private PMTiles are possible, add a backend session endpoint that issues
  `Cloud-CDN-Cookie` for allowed users only.
- If private PMTiles are possible, ensure PMTiles range requests include
  browser credentials and private layers wait for the session endpoint.
- If backend code downloads shared data, add the SDK dependency and use
  `fetch_dataset("<slug>", "<format>")`; read the local path from
  `ref.cache_path` and record resolved dataset identity from `ref.resolved_id`
  when lineage matters.
- If backend code runs in GCP, confirm the runtime identity has
  `roles/storage.objectViewer` on the shared bucket.
- If backend code signs private PMTiles cookies, confirm the runtime identity
  has `roles/secretmanager.secretAccessor` on the shared signing-key secret.
- Do not expose GCS credentials to browser code.

## Private PMTiles Rules

Private PMTiles are active catalog concepts, not a future placeholder. Browser
clients still must not receive GCS credentials. This app's backend issues a
Cloud CDN signed cookie only for users allowed to access the private tier.

Signed-cookie support belongs in this app's backend, not in static frontend
configuration. The backend signs the allowed tier prefix,
`https://tiles.skytruth.org/pmtiles/private/`, and sets `Cloud-CDN-Cookie` with
a short TTL. Browser PMTiles fetches must send credentials so the cross-site
cookie is included.

## Tests

Add focused tests that prove:

- PMTiles URLs use `https://tiles.skytruth.org/pmtiles/public/...`.
- Private PMTiles URLs use
  `https://tiles.skytruth.org/pmtiles/private/...`.
- Catalog-derived PMTiles URLs use `access_tier`; tests should include at least
  one public and one private fixture row.
- Browser/service config reads use
  `https://tiles.skytruth.org/_catalog/web/catalog.json` or an app-owned API,
  not direct public GCS URLs.
- PMTiles layer source configuration does not hardcode
  `storage.googleapis.com`.
- The PMTiles base URL is configurable by environment.
- Public-tier PMTiles do not add signed-cookie or service-account auth.
- Private signed-cookie PMTiles reject anonymous users, set `Cache-Control:
  no-store`, set `Cloud-CDN-Cookie` for authorized users, send browser fetch
  credentials, and do not expose GCS credentials.
- WDPA MPA selection/join logic uses `site_id`, not `WDPAID`.
- Backend code that needs data files uses `fetch_dataset(...)` or
  `resolve_dataset(...)` rather than service account keys.
- Backend code that requests `version="latest"` and records lineage persists
  `ref.resolved_id`, not `<slug>@latest` and not a cache-path-derived version.

## Non-Goals

Do not implement Cloud CDN infrastructure in this project.
Do not store Cloud CDN signing keys in this repo or Terraform state.
Do not create service account keys.
Do not expose GCS credentials to the browser.
Do not mirror PMTiles into this project.
Do not make broad UI rewrites while doing the minimal shared-datasets adoption.
