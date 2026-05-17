"""Regression tests for Alembic migration portability."""

from pathlib import Path

import httpx
import pytest
import sqlalchemy as sa

from alembic import command
from alembic.config import Config

REPO_ROOT = Path(__file__).resolve().parents[2]
MIGRATION_DIR = REPO_ROOT / "alembic" / "versions"
PG_CTL = REPO_ROOT / ".conda" / "ceru-ci" / "bin" / "pg_ctl"
EEZ_REVISION = "5e03ce584f3c"


def _migration_paths() -> list[Path]:
    return sorted(MIGRATION_DIR.glob("*.py"))


def test_alembic_versions_do_not_import_generated_database_schema():
    offenders = []
    for path in _migration_paths():
        text = path.read_text(encoding="utf-8")
        if (
            "cerulean_cloud.database_schema" in text
            or "from cerulean_cloud import database_schema" in text
        ):
            offenders.append(path.name)

    assert not offenders


def test_alembic_versions_do_not_use_orm_sessions():
    forbidden_fragments = [
        "from sqlalchemy import orm",
        "from sqlalchemy.orm import",
        "sqlalchemy.orm",
        "orm.Session",
    ]
    offenders = []
    for path in _migration_paths():
        text = path.read_text(encoding="utf-8")
        matches = [fragment for fragment in forbidden_fragments if fragment in text]
        if matches:
            offenders.append(f"{path.name}: {', '.join(matches)}")

    assert not offenders


@pytest.mark.skipif(
    not PG_CTL.exists(), reason="canonical ceru-ci PostgreSQL pg_ctl is unavailable"
)
def test_blank_postgres_upgrade_and_downgrade_through_eez_canary(
    postgresql, monkeypatch
):
    db_url = (
        f"postgresql://{postgresql.info.user}:@"
        f"{postgresql.info.host}:{postgresql.info.port}/{postgresql.info.dbname}"
    )
    monkeypatch.setenv("DB_URL", db_url)
    monkeypatch.setattr(httpx, "get", _fake_httpx_get)

    config = Config(str(REPO_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(REPO_ROOT / "alembic"))

    command.upgrade(config, EEZ_REVISION)

    engine = sa.create_engine(db_url)
    try:
        with engine.connect() as conn:
            assert (
                conn.execute(
                    sa.text("SELECT version_num FROM alembic_version")
                ).scalar_one()
                == EEZ_REVISION
            )
            counts = (
                conn.execute(
                    sa.text("""
                    SELECT
                        (SELECT count(*) FROM cls) AS cls_count,
                        (SELECT count(*) FROM model) AS model_count,
                        (SELECT count(*) FROM aoi_type) AS aoi_type_count,
                        (SELECT count(*) FROM tag) AS tag_count,
                        (SELECT count(*) FROM aoi_eez) AS eez_count
                    """)
                )
                .mappings()
                .one()
            )
            assert dict(counts) == {
                "cls_count": 9,
                "model_count": 2,
                "aoi_type_count": 4,
                "tag_count": 4,
                "eez_count": 1,
            }

            eez_row = (
                conn.execute(
                    sa.text("""
                    SELECT aoi.name, eez.mrgid, eez.sovereigns
                    FROM aoi
                    JOIN aoi_eez AS eez ON eez.aoi_id = aoi.id
                    JOIN aoi_type AS type ON type.id = aoi.type
                    WHERE type.short_name = 'EEZ'
                    """)
                )
                .mappings()
                .one()
            )
            assert eez_row["name"] == "Test EEZ"
            assert eez_row["mrgid"] == 123
            assert eez_row["sovereigns"] == ["Testland"]

        command.downgrade(config, "base")
    finally:
        engine.dispose()


class _GeoJSONResponse:
    def json(self):
        return {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "GEONAME": "Test EEZ",
                        "MRGID": 123,
                        "SOVEREIGN1": "Testland",
                        "SOVEREIGN2": None,
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [-1.0, 0.0],
                                [-1.0, 1.0],
                                [0.0, 1.0],
                                [0.0, 0.0],
                                [-1.0, 0.0],
                            ]
                        ],
                    },
                }
            ],
        }


def _fake_httpx_get(*_args, **_kwargs):
    return _GeoJSONResponse()
