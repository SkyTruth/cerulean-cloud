"""Add AOI access type and dataset version metadata

Revision ID: 1f70e7d0c5b1
Revises: d6c7b48d9f11
Create Date: 2026-04-23 15:00:00.000000

"""

import json

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision = "1f70e7d0c5b1"
down_revision = "d6c7b48d9f11"
branch_labels = None
depends_on = None

SLICK_PLUS_2_SQL = """
CREATE OR REPLACE VIEW public.slick_plus_2 AS
WITH not_oil_clses AS (
    SELECT id
    FROM public.get_slick_subclses(1)
),
base AS (
    SELECT
        id,
        slick_timestamp,
        geometry::geometry,
        machine_confidence,
        geometric_slick_potential AS slick_confidence,
        length,
        area,
        perimeter,
        centroid,
        polsby_popper,
        fill_factor,
        centerlines,
        aspect_ratio_factor,
        cls,
        orchestrator_run,
        length^2 / area / polsby_popper AS linearity
    FROM public.slick
    WHERE active
      AND cls NOT IN (SELECT id FROM not_oil_clses)
)
SELECT
    base.*,
    sentinel1_grd.scene_id AS s1_scene_id,
    sentinel1_grd.geometry AS s1_geometry,
    hs.cls AS hitl_cls,
    cls.long_name AS hitl_cls_name,
    aois.aoi_type_1_ids,
    aois.aoi_type_2_ids,
    aois.aoi_type_3_ids,
    srcs.source_type_1_ids,
    srcs.source_type_2_ids,
    srcs.source_type_3_ids,
    srcs.max_source_collated_score,
    'https://cerulean.skytruth.org/slicks/' || base.id || '?ref=api&slick_id=' || base.id
        AS slick_url,
    aois.aoi_ids
FROM base
JOIN public.orchestrator_run ON orchestrator_run.id = base.orchestrator_run
JOIN public.sentinel1_grd ON sentinel1_grd.id = orchestrator_run.sentinel1_grd
LEFT JOIN LATERAL (
    SELECT hs.cls
    FROM public.hitl_slick hs
    WHERE hs.slick = base.id
    ORDER BY hs.update_time DESC
    LIMIT 1
) AS hs ON TRUE
LEFT JOIN public.cls ON cls.id = hs.cls
LEFT JOIN LATERAL (
    SELECT
        array_agg(aoi.id) FILTER (WHERE aoi_type_for_ids.short_name = 'EEZ') AS aoi_type_1_ids,
        array_agg(aoi.id) FILTER (WHERE aoi_type_for_ids.short_name = 'IHO') AS aoi_type_2_ids,
        array_agg(aoi.id) FILTER (WHERE aoi_type_for_ids.short_name = 'MPA') AS aoi_type_3_ids,
        (
            SELECT COALESCE(json_object_agg(aoi_ids.short_name, aoi_ids.ext_ids), '{}'::json)
            FROM (
                SELECT
                    aoi_type.short_name,
                    json_agg(aoi_by_type.ext_id ORDER BY aoi_by_type.ext_id) AS ext_ids
                FROM public.slick_to_aoi sta_by_type
                JOIN public.aoi aoi_by_type ON aoi_by_type.id = sta_by_type.aoi
                JOIN public.aoi_type ON aoi_type.id = aoi_by_type.type
                WHERE sta_by_type.slick = base.id
                  AND aoi_type.short_name IN ('EEZ', 'IHO', 'MPA')
                  AND aoi_by_type.ext_id IS NOT NULL
                GROUP BY aoi_type.short_name
            ) AS aoi_ids
        ) AS aoi_ids
    FROM public.slick_to_aoi sta
    JOIN public.aoi ON aoi.id = sta.aoi
    JOIN public.aoi_type AS aoi_type_for_ids ON aoi_type_for_ids.id = aoi.type
    WHERE sta.slick = base.id
) AS aois ON TRUE
LEFT JOIN LATERAL (
    SELECT
        array_agg(src.ext_id) FILTER (WHERE source_type.short_name = 'VESSEL') AS source_type_1_ids,
        array_agg(src.ext_id) FILTER (WHERE source_type.short_name = 'INFRA') AS source_type_2_ids,
        array_agg(src.ext_id) FILTER (WHERE source_type.short_name = 'DARK') AS source_type_3_ids,
        MAX(sts.collated_score) AS max_source_collated_score
    FROM public.slick_to_source sts
    JOIN public.source src ON src.id = sts.source
    JOIN public.source_type ON source_type.id = src.type
    WHERE sts.slick = base.id
      AND sts.active = TRUE
) AS srcs ON TRUE
WHERE hs.cls IS NULL OR hs.cls NOT IN (SELECT id FROM not_oil_clses);
"""


def _get_seed_ids():
    bind = op.get_bind()

    owner_id = bind.execute(
        sa.text(
            """
            SELECT id
            FROM public.users
            WHERE email = :email
            ORDER BY id
            LIMIT 1
            """
        ),
        {"email": "dummy@dummy.dummy"},
    ).scalar()
    if owner_id is None:
        op.get_context().impl.static_output(
            "No bootstrap user dummy@dummy.dummy found; leaving aoi_type.owner NULL."
        )

    read_perm_id = bind.execute(
        sa.text(
            """
            SELECT id
            FROM public.permission
            WHERE short_name = :short_name
            ORDER BY id
            LIMIT 1
            """
        ),
        {"short_name": "any"},
    ).scalar()
    if read_perm_id is None:
        raise RuntimeError(
            "Expected seeded permission short_name='any' before AOI access migration."
        )

    return owner_id, read_perm_id


def _update_aoi_type(
    *,
    short_name: str,
    filter_toggle: bool,
    access_type: str,
    properties: dict,
    owner_id: int,
    read_perm_id: int,
) -> None:
    op.get_bind().execute(
        sa.text(
            """
            UPDATE public.aoi_type
            SET
                filter_toggle = :filter_toggle,
                owner = :owner_id,
                read_perm = :read_perm_id,
                access_type = :access_type,
                properties = CAST(:properties AS jsonb)
            WHERE short_name = :short_name
            """
        ),
        {
            "short_name": short_name,
            "filter_toggle": filter_toggle,
            "owner_id": owner_id,
            "read_perm_id": read_perm_id,
            "access_type": access_type,
            "properties": json.dumps(properties),
        },
    )


def upgrade():
    owner_id, read_perm_id = _get_seed_ids()

    op.add_column("aoi", sa.Column("ext_id", sa.Text()))

    op.execute(
        """
        UPDATE public.aoi a
        SET ext_id = eez.mrgid::text
        FROM public.aoi_eez eez
        WHERE eez.aoi_id = a.id
        """
    )
    op.execute(
        """
        UPDATE public.aoi a
        SET ext_id = iho.mrgid::text
        FROM public.aoi_iho iho
        WHERE iho.aoi_id = a.id
        """
    )
    op.execute(
        """
        UPDATE public.aoi a
        SET ext_id = mpa.wdpaid::text
        FROM public.aoi_mpa mpa
        WHERE mpa.aoi_id = a.id
        """
    )

    op.create_index("idx_aoi_type_ext_id", "aoi", ["type", "ext_id"])

    op.alter_column(
        "aoi_type",
        "short_name",
        existing_type=sa.Text(),
        nullable=False,
    )
    op.create_unique_constraint(
        "uq_aoi_type_short_name",
        "aoi_type",
        ["short_name"],
    )

    op.create_table(
        "aoi_access_type",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("short_name", sa.Text(), nullable=False, unique=True),
        sa.Column("prop_keys", sa.ARRAY(sa.Text()), nullable=False),
    )

    op.execute(
        """
        INSERT INTO public.aoi_access_type (id, short_name, prop_keys)
        VALUES
            (1, 'GCS', ARRAY['fgb_uri', 'pmt_uri', 'dataset_version', 'ext_id_field', 'display_name_field']),
            (2, 'DB_LOCAL', ARRAY['table_name', 'geog_col', 'ext_id_col']),
            (3, 'DB_REMOTE', ARRAY['db_conn_str', 'table_name', 'geog_col', 'ext_id_col'])
        """
    )

    op.add_column("aoi_type", sa.Column("filter_toggle", sa.Boolean()))
    op.add_column("aoi_type", sa.Column("owner", sa.BigInteger()))
    op.add_column("aoi_type", sa.Column("read_perm", sa.BigInteger()))
    op.add_column("aoi_type", sa.Column("access_type", sa.Text()))
    op.add_column("aoi_type", sa.Column("properties", postgresql.JSONB()))

    op.create_foreign_key(
        "fk_aoi_type_owner_users",
        "aoi_type",
        "users",
        ["owner"],
        ["id"],
    )
    op.create_foreign_key(
        "fk_aoi_type_read_perm_permission",
        "aoi_type",
        "permission",
        ["read_perm"],
        ["id"],
    )
    op.create_foreign_key(
        "fk_aoi_type_access_type_aoi_access_type",
        "aoi_type",
        "aoi_access_type",
        ["access_type"],
        ["short_name"],
    )

    _update_aoi_type(
        short_name="EEZ",
        filter_toggle=True,
        access_type="GCS",
        properties={
            "fgb_uri": "gs://cerulean-cloud-aoi/eez-mr/eez_v12.fgb",
            "pmt_uri": "gs://cerulean-cloud-aoi/eez-mr/eez_v12.pmt",
            "dataset_version": "eez_v12",
            "ext_id_field": "MRGID",
            "display_name_field": "GEONAME",
        },
        owner_id=owner_id,
        read_perm_id=read_perm_id,
    )
    _update_aoi_type(
        short_name="IHO",
        filter_toggle=False,
        access_type="GCS",
        properties={
            "fgb_uri": "gs://cerulean-cloud-aoi/iho-mr/World_Seas_IHO_v3.fgb",
            "pmt_uri": "gs://cerulean-cloud-aoi/iho-mr/World_Seas_IHO_v3.pmt",
            "dataset_version": "World_Seas_IHO_v3",
            "ext_id_field": "MRGID",
            "display_name_field": "NAME",
        },
        owner_id=owner_id,
        read_perm_id=read_perm_id,
    )
    _update_aoi_type(
        short_name="MPA",
        filter_toggle=True,
        access_type="GCS",
        properties={
            "fgb_uri": "gs://cerulean-cloud-aoi/mpa-wdpa/marine_wdpa_0.001.fgb",
            "pmt_uri": "gs://cerulean-cloud-aoi/mpa-wdpa/marine_wdpa_0.001.pmt",
            "dataset_version": "marine_wdpa_0.001",
            "ext_id_field": "WDPAID",
            "display_name_field": "NAME",
        },
        owner_id=owner_id,
        read_perm_id=read_perm_id,
    )
    _update_aoi_type(
        short_name="USER",
        filter_toggle=False,
        access_type="DB_LOCAL",
        properties={
            "table_name": "aoi_user",
            "geog_col": "geometry",
            "ext_id_col": "aoi_id",
        },
        owner_id=owner_id,
        read_perm_id=read_perm_id,
    )

    op.execute("ALTER TABLE public.aoi_user ADD COLUMN geometry geography")
    op.create_index(
        "idx_aoi_user_geometry",
        "aoi_user",
        ["geometry"],
        postgresql_using="gist",
    )

    op.add_column(
        "orchestrator_run",
        sa.Column("dataset_versions", postgresql.JSONB()),
    )
    op.execute(
        """
        UPDATE public.orchestrator_run
        SET dataset_versions = jsonb_build_object('sea_ice_date', sea_ice_date)
        """
    )

    op.execute(
        """
        CREATE OR REPLACE FUNCTION public.slick_before_trigger_func()
        RETURNS trigger
        AS $$
        DECLARE
            timer timestamptz := clock_timestamp();
            _geog geography := NEW.geometry;
            _geom geometry;
            oriented_envelope geometry;
            oe_ring geometry;
            rec record;
        BEGIN
            RAISE NOTICE '---------------------------------------------------------';
            RAISE NOTICE 'In slick_before_trigger_func. %', (clock_timestamp() - timer)::interval;
            _geom := _geog::geometry;
            oriented_envelope := st_orientedenvelope(_geom);
            oe_ring := st_exteriorring(oriented_envelope);
            NEW.geometry_count := st_numgeometries(_geom);
            NEW.largest_area := (
                SELECT MAX(st_area((poly.geom)::geography))
                FROM st_dump(_geom) AS poly
            );
            NEW.median_area := (
                SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY st_area((poly.geom)::geography))
                FROM st_dump(_geom) AS poly
            );
            NEW.area := st_area(_geog);
            NEW.centroid := st_centroid(_geog);
            NEW.perimeter = st_perimeter(_geog);
            NEW.polsby_popper := 4.0 * pi() * NEW.area / (NEW.perimeter ^ 2.0);
            NEW.fill_factor := NEW.area / st_area(oriented_envelope::geography);
            NEW.length := GREATEST(
                st_distance(
                    st_pointn(oe_ring,1)::geography,
                    st_pointn(oe_ring,2)::geography
                ),
                st_distance(
                    st_pointn(oe_ring,2)::geography,
                    st_pointn(oe_ring,3)::geography
                )
            );
            RAISE NOTICE 'Calculated all generated fields. %', (clock_timestamp() - timer)::interval;
            NEW.cls := COALESCE(
                NEW.cls,
                (
                    SELECT cls.id
                    FROM cls
                    JOIN orchestrator_run ON NEW.orchestrator_run = orchestrator_run.id
                    JOIN LATERAL json_each_text((SELECT cls_map FROM model WHERE id = orchestrator_run.model))
                        m(key, value)
                        ON key::integer = NEW.inference_idx
                    WHERE cls.short_name = CASE
                        WHEN value = 'BACKGROUND' THEN 'NOT_OIL'
                        ELSE value
                    END
                    LIMIT 1
                )
            );
            RAISE NOTICE 'Calculated NEW.cls. %', (clock_timestamp() - timer)::interval;

            RETURN NEW;
        END;
        $$ LANGUAGE PLPGSQL;
        """
    )
    op.execute(SLICK_PLUS_2_SQL)


def downgrade():
    op.execute("DROP VIEW IF EXISTS public.slick_plus_2")
    op.execute("DROP RULE IF EXISTS bypass_slick_to_aoi_insert ON public.slick_to_aoi")
    op.execute(
        """
        CREATE OR REPLACE FUNCTION public.slick_before_trigger_func()
        RETURNS trigger
        AS $$
        DECLARE
            timer timestamptz := clock_timestamp();
            _geog geography := NEW.geometry;
            _geom geometry;
            oriented_envelope geometry;
            oe_ring geometry;
            rec record;
        BEGIN
            RAISE NOTICE '---------------------------------------------------------';
            RAISE NOTICE 'In slick_before_trigger_func. %', (clock_timestamp() - timer)::interval;
            _geom := _geog::geometry;
            oriented_envelope := st_orientedenvelope(_geom);
            oe_ring := st_exteriorring(oriented_envelope);
            NEW.geometry_count := st_numgeometries(_geom);
            NEW.largest_area := (
                SELECT MAX(st_area((poly.geom)::geography))
                FROM st_dump(_geom) AS poly
            );
            NEW.median_area := (
                SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY st_area((poly.geom)::geography))
                FROM st_dump(_geom) AS poly
            );
            NEW.area := st_area(_geog);
            NEW.centroid := st_centroid(_geog);
            NEW.perimeter = st_perimeter(_geog);
            NEW.polsby_popper := 4.0 * pi() * NEW.area / (NEW.perimeter ^ 2.0);
            NEW.fill_factor := NEW.area / st_area(oriented_envelope::geography);
            NEW.length := GREATEST(
                st_distance(
                    st_pointn(oe_ring,1)::geography,
                    st_pointn(oe_ring,2)::geography
                ),
                st_distance(
                    st_pointn(oe_ring,2)::geography,
                    st_pointn(oe_ring,3)::geography
                )
            );
            RAISE NOTICE 'Calculated all generated fields. %', (clock_timestamp() - timer)::interval;
            NEW.cls := COALESCE(
                NEW.cls,
                (
                    SELECT cls.id
                    FROM cls
                    JOIN orchestrator_run ON NEW.orchestrator_run = orchestrator_run.id
                    JOIN LATERAL json_each_text((SELECT cls_map FROM model WHERE id = orchestrator_run.model))
                        m(key, value)
                        ON key::integer = NEW.inference_idx
                    WHERE cls.short_name = CASE
                        WHEN value = 'BACKGROUND' THEN 'NOT_OIL'
                        ELSE value
                    END
                    LIMIT 1
                )
            );
            RAISE NOTICE 'Calculated NEW.cls. %', (clock_timestamp() - timer)::interval;

            INSERT INTO slick_to_aoi(slick, aoi)
            SELECT DISTINCT NEW.id, aoi_chunks.id
            FROM aoi_chunks
            WHERE st_intersects(_geom, aoi_chunks.geometry);

            RAISE NOTICE 'Insert done to slick_to_aoi. %', (clock_timestamp() - timer)::interval;

            RETURN NEW;
        END;
        $$ LANGUAGE PLPGSQL;
        """
    )

    op.drop_column("orchestrator_run", "dataset_versions")

    op.drop_index("idx_aoi_user_geometry", table_name="aoi_user")
    op.drop_column("aoi_user", "geometry")

    op.drop_constraint(
        "fk_aoi_type_access_type_aoi_access_type",
        "aoi_type",
        type_="foreignkey",
    )
    op.drop_constraint(
        "fk_aoi_type_read_perm_permission",
        "aoi_type",
        type_="foreignkey",
    )
    op.drop_constraint("fk_aoi_type_owner_users", "aoi_type", type_="foreignkey")
    op.drop_column("aoi_type", "properties")
    op.drop_column("aoi_type", "access_type")
    op.drop_column("aoi_type", "read_perm")
    op.drop_column("aoi_type", "owner")
    op.drop_column("aoi_type", "filter_toggle")

    op.drop_table("aoi_access_type")

    op.drop_constraint(
        "uq_aoi_type_short_name",
        "aoi_type",
        type_="unique",
    )
    op.alter_column(
        "aoi_type",
        "short_name",
        existing_type=sa.Text(),
        nullable=True,
    )

    op.drop_index("idx_aoi_type_ext_id", table_name="aoi")
    op.drop_column("aoi", "ext_id")
