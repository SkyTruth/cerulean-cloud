# Debug image

```
cd stack/
docker build -f ../Dockerfiles/Dockerfile.cloud_run_offset -t gcr.io/cerulean-338116/cloud-run-offset-tile-image ../
PORT=8080 && docker run -p 8080:${PORT} -e PORT=${PORT} gcr.io/cerulean-338116/cloud-run-offset-tile-image
```