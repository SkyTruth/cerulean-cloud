"""Shared SQLAlchemy base & metadata for the Cerulean Cloud data‑model package."""

from sqlalchemy.orm import declarative_base
from sqlalchemy.orm.decl_api import DeclarativeMeta

#: Declarative base used by **all** model classes
Base: DeclarativeMeta = declarative_base()  # noqa: N816  (keep legacy camel‑case)
#: Shared metadata, exposed for Alembic autogenerate and engine bindings
metadata = Base.metadata
