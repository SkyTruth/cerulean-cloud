FROM postgres:18-bookworm


RUN apt-get update \
    && apt-get install -y --no-install-recommends "postgresql-${PG_MAJOR}-postgis-3" \
    && rm -rf /var/lib/apt/lists/*


CMD ["/usr/local/bin/docker-entrypoint.sh","postgres", "-c", "log_statement=all"]
