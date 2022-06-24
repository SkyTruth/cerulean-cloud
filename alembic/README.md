# Generic single-database configuration.

## Create a revision

```
alembic revision -m "create account table"
```

## Apply revision

```
alembic upgrade head
```