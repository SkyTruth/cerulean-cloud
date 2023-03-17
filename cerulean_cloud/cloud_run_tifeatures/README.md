# M1 MacBooks
set `DOCKER_DEFAULT_PLATFORM=linux/amd64` in M1 macbooks

# Debug image

```
cd stack/
docker build -f ../Dockerfiles/Dockerfile.cloud_run_tifeatures -t gcr.io/cerulean-338116/cloud-run-tifeatures ../ --no-cache
PORT=8080 && docker run -p 8080:${PORT} -e PORT=${PORT} -e DATABASE_URL=${DB_URL} gcr.io/cerulean-338116/cloud-run-tifeatures
```
