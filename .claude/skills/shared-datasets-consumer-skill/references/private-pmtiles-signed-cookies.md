# Private PMTiles Signed Cookies

Read this reference before implementing, reviewing, or debugging private
shared-datasets PMTiles in this consumer repo or app.

## Shared-Datasets Prerequisites

Before deploying this consumer environment, identify the backend or UI runtime
service account that serves this app's PMTiles session endpoint. Grant that
service account access to the shared signing-key secret from the upstream
`shared-datasets-1` project; do not create or distribute a service account key
file.

Current shared signing material:

```text
Secret: projects/shared-datasets-1/secrets/pmtiles-cdn-signed-request-key/versions/latest
Cloud CDN key name: shared-datasets-pmtiles-v1
Signed prefix: https://tiles.skytruth.org/pmtiles/private/
```

In the upstream shared-datasets-1 Terraform, add this app's signer principal to
the signer allowlist used by
`google_secret_manager_secret_iam_member.pmtiles_cdn_cookie_signers` in
`terraform/envs/prod/pmtiles_cdn.tf`. The signer members come from the
historically named `cerulean_pmtiles_cookie_signer_service_accounts` variable in
`terraform/envs/prod/variables.tf` and
`terraform/envs/prod/production.auto.tfvars`.

The Cerulean rollout used:

```hcl
cerulean_pmtiles_cookie_signer_service_accounts = [
  "serviceAccount:734798842681-compute@developer.gserviceaccount.com",
]
```

For this repo or app, use its real runtime service account. The 30x30 reader
service account is
`shared-datasets-reader@x30-399415.iam.gserviceaccount.com`, but the signer
should be the service account actually running the endpoint that issues the
cookie.

CDN backend-bucket mode requires exact browser origins in
`pmtiles_cdn_allowed_origins`; credentialed CORS cannot use `*`,
`https://*.skytruth.org`, or the redirector-only
`pmtiles_cdn_allowed_origin_regexes` regex list. Add each browser origin before
deploying private PMTiles in that environment.

Verified on 2026-05-17, the exact CDN allowlist was:

```text
http://localhost:3000
https://localhost:3000
https://feature-three.cerulean.skytruth.org
https://test.cerulean.skytruth.org
https://develop.cerulean.skytruth.org
https://cerulean.skytruth.org
https://30x30.skytruth.org
https://monitor.skytruth.org
```

Re-check Terraform before relying on that list.

Any upstream shared-datasets-1 Terraform change must land through a reviewed PR
and the protected production workflow after merge, not a local apply.

## App Backend Endpoint

Add an endpoint like:

```text
GET /api/pmtiles/session?tier=public
GET /api/pmtiles/session?tier=private
```

Required behavior:

- `tier=public` returns `204` without a cookie.
- `tier=private` requires an authenticated user session.
- For the first rollout, allow only SkyTruth or admin users unless this app has
  a more specific entitlement model.
- Unknown `tier` returns `400`.
- Unauthorized private users return `401` or `403`.
- Authorized private users read
  `projects/shared-datasets-1/secrets/pmtiles-cdn-signed-request-key/versions/latest`.
- Decode the secret value from base64url to the raw 16-byte Cloud CDN key.
- Sign `https://tiles.skytruth.org/pmtiles/private/`, not individual PMTiles
  URLs.
- Use HMAC-SHA1 and key name `shared-datasets-pmtiles-v1`.
- Set `Cache-Control: no-store`.
- Never log or return the key bytes, HMAC input, HMAC output, unsigned policy,
  full cookie value, or signed URL.

The cookie must be named `Cloud-CDN-Cookie` and set with:

```text
Domain=.skytruth.org
Path=/pmtiles
Secure
HttpOnly
SameSite=None
Max-Age=3600
Expires=<1 hour from now>
```

Cloud CDN signed-cookie policy fields must be in this order:

```text
URLPrefix=<base64url-prefix>:Expires=<unix-seconds>:KeyName=shared-datasets-pmtiles-v1:Signature=<base64url-hmac>
```

Keep base64url padding characters if the runtime's encoder emits them. The key
used for HMAC is the decoded raw 16-byte key, not the encoded secret text.

## App Frontend Loading

Before enabling private shared layers, call:

```ts
await fetch("/api/pmtiles/session?tier=private", {
  credentials: "include"
});
```

Only add the private layer if the response is successful. Public layers may skip
the call or call `tier=public`, which should return `204`.

PMTiles byte-range requests must include credentials so the browser sends the
cross-site cookie to `tiles.skytruth.org`. If the PMTiles library hides
byte-range fetches, wrap or subclass its fetch source so every header,
directory, and tile range request includes:

```ts
fetch(url, {
  headers: { range: "bytes=start-end" },
  credentials: "include"
});
```

Keep CSP `connect-src` allowing `https://tiles.skytruth.org`. Remove direct
`storage.googleapis.com` dependencies only for shared-dataset PMTiles access; do
not remove `storage.googleapis.com` if the app still uses it for unrelated
features.

## Validation

Run this repo's lint, type, test, and build checks. Also verify:

- Public PMTiles load without a cookie.
- Anonymous private users are denied.
- Authorized private users receive `Cloud-CDN-Cookie` and `Cache-Control:
  no-store`.
- Private PMTiles range requests carry the `Cloud-CDN-Cookie`.
- Expired or malformed cookies fail with `403`.
- The deployed browser origin is exactly allowlisted for credentialed CORS.
- No logs contain secret bytes, HMAC material, signed cookie values, or GCS
  credentials.
