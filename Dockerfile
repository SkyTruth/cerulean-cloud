FROM postgres:17-bookworm


RUN apt-get update && apt-get install -y postgresql-17-postgis-3


CMD ["/usr/local/bin/docker-entrypoint.sh","postgres", "-c", "log_statement=all"]
