# Generic single-database configuration.

Set your connection to the DB by using the `DB_URL` env var.
```
export DB_URL=DB_URL=postgresql://cerulean-cloud-test-database:*****somepassword*****@34.79.58.65/cerulean-cloud-test-database
```

## Create a revision

```
alembic revision -m "create account table"
```

## Apply revision

```
alembic upgrade head
```