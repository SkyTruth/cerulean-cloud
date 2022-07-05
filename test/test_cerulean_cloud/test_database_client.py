"""Test database client"""
import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session, scoped_session, sessionmaker

import cerulean_cloud.database_schema as database_schema
from cerulean_cloud.database_client import get_engine


def test_get_engine(postgresql):
    connection = f"postgresql://{postgresql.info.user}:@{postgresql.info.host}:{postgresql.info.port}/{postgresql.info.dbname}"
    engine = get_engine(connection)
    assert Session(engine)


@pytest.fixture
def connection(postgresql):
    connection = f"postgresql://{postgresql.info.user}:@{postgresql.info.host}:{postgresql.info.port}/{postgresql.info.dbname}"
    engine = get_engine(connection)
    return engine.connect()


@pytest.fixture
def setup_database(connection):
    database_schema.Base.metadata.bind = connection
    connection.execute(sa.text("CREATE EXTENSION postgis"))
    database_schema.Base.metadata.create_all()

    yield

    database_schema.Base.metadata.drop_all()


@pytest.fixture
def db_session(setup_database, connection):
    transaction = connection.begin()
    yield scoped_session(
        sessionmaker(autocommit=False, autoflush=False, bind=connection)
    )
    transaction.rollback()


def test_create_model(db_session):
    db_session.add(database_schema.Model(name="Jane Doe", file_path="true"))
    db_session.commit()

    assert (
        db_session.query(database_schema.Model).filter_by(name="Jane Doe").one_or_none()
    )
