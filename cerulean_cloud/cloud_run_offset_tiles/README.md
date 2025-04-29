# M1 MacBooks
set `DOCKER_DEFAULT_PLATFORM=linux/amd64` in M1 macbooks

# Build image

```shell
docker build -f Dockerfiles/Dockerfile.cloud_run_offset -t gcr.io/cerulean-338116/cr-infer-image .
```

# Debug image

```shell
PORT=8080 && docker run --rm -p 8080:${PORT} -e UVICORN_PORT=${PORT} --name cloud_run_offset_tiles gcr.io/cerulean-338116/cr-infer-image
```

# Extract the SBOM

Start the container as described in [Debug image](#debug-image) and in another shell execute

```shell
docker cp cloud_run_offset_tiles:/app/sbom.xml .
```
