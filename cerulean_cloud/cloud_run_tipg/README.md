
# M1 MacBooks
set `DOCKER_DEFAULT_PLATFORM=linux/amd64` in M1 macbooks

# Build image

```shell
docker build -f Dockerfiles/Dockerfile.cloud_run_tipg -t gcr.io/cerulean-338116/cr-tipg .
```

# Debug image

```shell
PORT=8080 && docker run --rm -p 8080:${PORT} -e UVICORN_PORT=${PORT} -e DATABASE_URL=${DB_URL} --name cloud_run_tipg gcr.io/cerulean-338116/cr-tipg
```

# Extract the SBOM

Start the container as described in [Debug image](#debug-image) and in another shell execute

```shell
docker cp cloud_run_tipg:/app/sbom.xml .
```

# Public collections

`TIPG_DB_TABLES` and `TIPG_DB_FUNCTIONS` are JSON arrays of schema-qualified
collection names. The deployed allowlists live in `stack/cloud_run_tipg.py`.
Only listed tables, views, and functions appear in `/collections` or have API
endpoints; new database objects remain private until explicitly added.
Missing or empty allowlists expose no collections of that type.

There is no API-key override for omitted collections. For local development, set
both allowlists explicitly, for example:

```shell
export TIPG_DB_TABLES='["public.slick_plus", "public.source_plus", "public.sentinel1_grd"]'
export TIPG_DB_FUNCTIONS='[]'
```
