"""Add cache tables for heavy views

Revision ID: 4a87d52cb4ea
Revises: 15b23d4d9aa1
Create Date: 2025-06-02 00:00:00.000000
"""

from alembic import op

revision = "4a87d52cb4ea"
down_revision = "15b23d4d9aa1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Drop old views
    op.execute("DROP VIEW IF EXISTS public.slick_plus")
    op.execute("DROP VIEW IF EXISTS public.source_plus")
    op.execute("DROP VIEW IF EXISTS public.repeat_source")

    # Drop existing cache tables if they exist to avoid schema mismatches
    op.execute("DROP TABLE IF EXISTS public.slick_plus")
    op.execute("DROP TABLE IF EXISTS public.source_plus")

    # Create cache tables
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS public.slick_plus (
            id BIGINT PRIMARY KEY,
            slick_timestamp TIMESTAMP,
            geometry GEOGRAPHY(MULTIPOLYGON,4326),
            active BOOLEAN,
            orchestrator_run BIGINT,
            create_time TIMESTAMP,
            inference_idx INTEGER,
            cls INTEGER,
            hitl_cls INTEGER,
            machine_confidence DOUBLE PRECISION,
            precursor_slicks BIGINT[],
            notes TEXT,
            centerlines JSON,
            aspect_ratio_factor DOUBLE PRECISION,
            length DOUBLE PRECISION,
            area DOUBLE PRECISION,
            perimeter DOUBLE PRECISION,
            centroid GEOGRAPHY(POINT,4326),
            polsby_popper DOUBLE PRECISION,
            fill_factor DOUBLE PRECISION,
            linearity DOUBLE PRECISION,
            s1_scene_id TEXT,
            s1_geometry GEOGRAPHY(POLYGON,4326),
            cls_short_name TEXT,
            cls_long_name TEXT,
            aoi_type_1_ids BIGINT[],
            aoi_type_2_ids BIGINT[],
            aoi_type_3_ids BIGINT[],
            source_type_1_ids BIGINT[],
            source_type_2_ids BIGINT[],
            source_type_3_ids BIGINT[],
            slick_url TEXT
        )
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS public.source_plus (
            geometry GEOGRAPHY(MULTIPOLYGON,4326),
            slick_id BIGINT,
            slick_confidence DOUBLE PRECISION,
            source_id BIGINT,
            mmsi_or_structure_id TEXT,
            source_type TEXT,
            source_collated_score DOUBLE PRECISION,
            source_rank BIGINT,
            create_time TIMESTAMP,
            git_tag TEXT,
            slick_url TEXT,
            source_url TEXT
        )
        """
    )

    op.execute(
        """
        CREATE MATERIALIZED VIEW public.repeat_source AS
            WITH agg AS (
                SELECT
                    s.id AS source_id,
                    count(DISTINCT sl.orchestrator_run) AS occurrence_count,
                    sum(sl.area) / 1000000 AS total_area
                FROM slick_to_source sts
                JOIN source s ON s.id = sts.source
                JOIN slick sl ON sl.id = sts.slick
                LEFT JOIN source_to_tag stt ON stt.source = sts.source
                LEFT JOIN hitl_slick hs ON hs.slick = sl.id
                WHERE sl.active
                    AND sl.cls <> 1
                    AND (hs.cls IS NULL OR hs.cls <> 1)
                    AND sts.active
                    AND sts.hitl_verification IS NOT FALSE
                    AND (
                        (s.type = 2 AND sts.rank = 1)
                        OR
                        (
                            s.type = 1
                            AND sts.collated_score > 0
                            AND (
                                stt.tag IS NULL
                                OR (stt.tag <> ALL (ARRAY[5,6,7]))
                            )
                        )
                    )
                GROUP BY s.id, s.ext_id, s.type
            )
            SELECT
                agg.source_id,
                agg.occurrence_count,
                agg.total_area,
                row_number() OVER (
                    ORDER BY agg.occurrence_count DESC, agg.total_area DESC
                ) AS global_rank
            FROM agg
            ORDER BY agg.occurrence_count DESC, agg.total_area DESC
        """
    )
    # Create helper functions to refresh cache tables
    op.execute(
        """
        CREATE OR REPLACE FUNCTION refresh_slick_plus_cache(ids bigint[])
        RETURNS void LANGUAGE SQL AS $$
            DELETE FROM slick_plus WHERE id = ANY(ids);
            INSERT INTO slick_plus (
                id,
                slick_timestamp,
                geometry,
                active,
                orchestrator_run,
                create_time,
                inference_idx,
                cls,
                hitl_cls,
                machine_confidence,
                precursor_slicks,
                notes,
                centerlines,
                aspect_ratio_factor,
                length,
                area,
                perimeter,
                centroid,
                polsby_popper,
                fill_factor,
                linearity,
                s1_scene_id,
                s1_geometry,
                cls_short_name,
                cls_long_name,
                aoi_type_1_ids,
                aoi_type_2_ids,
                aoi_type_3_ids,
                source_type_1_ids,
                source_type_2_ids,
                source_type_3_ids,
                slick_url
            )
            SELECT
                slick.id,
                slick.slick_timestamp,
                slick.geometry,
                slick.active,
                slick.orchestrator_run,
                slick.create_time,
                slick.inference_idx,
                slick.cls,
                slick.hitl_cls,
                slick.machine_confidence,
                slick.precursor_slicks,
                slick.notes,
                slick.centerlines,
                slick.aspect_ratio_factor,
                slick.length,
                slick.area,
                slick.perimeter,
                slick.centroid,
                slick.polsby_popper,
                slick.fill_factor,
                slick.length^2 / slick.area / slick.polsby_popper AS linearity,
                sentinel1_grd.scene_id AS s1_scene_id,
                sentinel1_grd.geometry AS s1_geometry,
                cls.short_name AS cls_short_name,
                cls.long_name AS cls_long_name,
                aoi_agg.aoi_type_1_ids,
                aoi_agg.aoi_type_2_ids,
                aoi_agg.aoi_type_3_ids,
                source_agg.source_type_1_ids,
                source_agg.source_type_2_ids,
                source_agg.source_type_3_ids,
                'https://cerulean.skytruth.org/slicks/' || slick.id::text || '?ref=api&slick_id=' || slick.id AS slick_url
            FROM slick
            JOIN orchestrator_run ON orchestrator_run.id = slick.orchestrator_run
            JOIN sentinel1_grd ON sentinel1_grd.id = orchestrator_run.sentinel1_grd
            JOIN cls ON cls.id = slick.cls
            LEFT JOIN (
                SELECT slick_to_aoi.slick,
                    array_agg(aoi.id) FILTER (WHERE aoi.type = 1) AS aoi_type_1_ids,
                    array_agg(aoi.id) FILTER (WHERE aoi.type = 2) AS aoi_type_2_ids,
                    array_agg(aoi.id) FILTER (WHERE aoi.type = 3) AS aoi_type_3_ids
                FROM slick_to_aoi
                JOIN aoi ON slick_to_aoi.aoi = aoi.id
                GROUP BY slick_to_aoi.slick
            ) aoi_agg ON aoi_agg.slick = slick.id
            LEFT JOIN (
                SELECT slick_to_source.slick,
                    array_agg(source.id) FILTER (WHERE source.type = 1) AS source_type_1_ids,
                    array_agg(source.id) FILTER (WHERE source.type = 2) AS source_type_2_ids,
                    array_agg(source.id) FILTER (WHERE source.type = 3) AS source_type_3_ids
                FROM slick_to_source
                JOIN source ON slick_to_source.source = source.id
                WHERE slick_to_source.active = true
                GROUP BY slick_to_source.slick
            ) source_agg ON source_agg.slick = slick.id
            WHERE slick.active = true AND slick.id = ANY(ids);
        $$
        """
    )

    op.execute(
        """
        CREATE OR REPLACE FUNCTION refresh_source_caches(ids bigint[])
        RETURNS void LANGUAGE SQL AS $$
            DELETE FROM source_plus WHERE slick_id = ANY(ids);
            INSERT INTO source_plus (
                geometry,
                slick_id,
                slick_confidence,
                source_id,
                mmsi_or_structure_id,
                source_type,
                source_collated_score,
                source_rank,
                create_time,
                git_tag,
                slick_url,
                source_url
            )
            SELECT
                sk.geometry AS geometry,
                sts.slick AS slick_id,
                sk.machine_confidence AS slick_confidence,
                s.id AS source_id,
                s.ext_id AS mmsi_or_structure_id,
                st.short_name AS source_type,
                sts.collated_score AS source_collated_score,
                sts.rank AS source_rank,
                sts.create_time AS create_time,
                sts.git_hash AS git_tag,
                'https://cerulean.skytruth.org/slicks/' || sk.id::text || '?ref=api&slick_id=' || sk.id AS slick_url,
                'https://cerulean.skytruth.org/?ref=api&' || st.ext_id_name || '=' || s.ext_id AS source_url
            FROM slick_to_source sts
            INNER JOIN source s ON sts.source = s.id
            INNER JOIN slick sk ON sts.slick = sk.id AND sk.active = TRUE
            INNER JOIN source_type st ON st.id = s.type
            WHERE sts.active = TRUE AND sts.slick = ANY(ids)
            ORDER BY sts.slick DESC, sts.rank;
            REFRESH MATERIALIZED VIEW CONCURRENTLY repeat_source;
        $$
        """
    )

    # Populate caches for existing slicks
    op.execute(
        "SELECT refresh_slick_plus_cache(ARRAY(SELECT id FROM slick WHERE active))"
    )
    op.execute("SELECT refresh_source_caches(ARRAY(SELECT id FROM slick WHERE active))")


def downgrade() -> None:
    op.execute("DROP MATERIALIZED VIEW IF EXISTS public.repeat_source")
    op.execute("DROP FUNCTION IF EXISTS refresh_source_caches(bigint[])")
    op.execute("DROP FUNCTION IF EXISTS refresh_slick_plus_cache(bigint[])")
    op.execute("DROP TABLE IF EXISTS public.source_plus")
    op.execute("DROP TABLE IF EXISTS public.slick_plus")

    from alembic_utils.pg_view import PGView

    slick_plus = PGView(
        schema="public",
        signature="slick_plus",
        definition="""
    SELECT
        slick.*,
        slick.length^2 / slick.area / slick.polsby_popper as linearity,
        sentinel1_grd.scene_id AS s1_scene_id,
        sentinel1_grd.geometry AS s1_geometry,
        cls.short_name AS cls_short_name,
        cls.long_name AS cls_long_name,
        aoi_agg.aoi_type_1_ids,
        aoi_agg.aoi_type_2_ids,
        aoi_agg.aoi_type_3_ids,
        source_agg.source_type_1_ids,
        source_agg.source_type_2_ids,
        source_agg.source_type_3_ids,
        'https://cerulean.skytruth.org/slicks/' || slick.id::text ||'?ref=api&slick_id=' || slick.id AS slick_url
    FROM slick
    JOIN orchestrator_run ON orchestrator_run.id = slick.orchestrator_run
    JOIN sentinel1_grd ON sentinel1_grd.id = orchestrator_run.sentinel1_grd
    JOIN cls ON cls.id = slick.cls
    LEFT JOIN (
        SELECT slick_to_aoi.slick,
            array_agg(aoi.id) FILTER (WHERE aoi.type = 1) AS aoi_type_1_ids,
            array_agg(aoi.id) FILTER (WHERE aoi.type = 2) AS aoi_type_2_ids,
            array_agg(aoi.id) FILTER (WHERE aoi.type = 3) AS aoi_type_3_ids
        FROM slick_to_aoi
        JOIN aoi ON slick_to_aoi.aoi = aoi.id
        GROUP BY slick_to_aoi.slick
        ) aoi_agg ON aoi_agg.slick = slick.id
     LEFT JOIN ( SELECT slick_to_source.slick,
            array_agg(source.id) FILTER (WHERE source.type = 1) AS source_type_1_ids,
            array_agg(source.id) FILTER (WHERE source.type = 2) AS source_type_2_ids,
            array_agg(source.id) FILTER (WHERE source.type = 3) AS source_type_3_ids
           FROM slick_to_source
             JOIN source ON slick_to_source.source = source.id
            WHERE slick_to_source.active = true
          GROUP BY slick_to_source.slick) source_agg ON source_agg.slick = slick.id
    WHERE slick.active = true;
    """,
    )
    op.create_entity(slick_plus)

    source_plus = PGView(
        schema="public",
        signature="source_plus",
        definition="""
            SELECT
                sk.geometry as geometry,
                sts.slick as slick_id,
                sk.machine_confidence as slick_confidence,
                s.id as source_id,
                s.ext_id as mmsi_or_structure_id,
                st.short_name as source_type,
                sts.collated_score as source_collated_score,
                sts.rank as source_rank,
                sts.create_time as create_time,
                sts.git_hash as git_tag,
                'https://cerulean.skytruth.org/slicks/' || sk.id::text ||'?ref=api&slick_id=' || sk.id AS slick_url,
                'https://cerulean.skytruth.org/?ref=api&' || st.ext_id_name || '=' || s.ext_id AS source_url
            FROM
                slick_to_source sts
            INNER JOIN
                source s ON sts.source = s.id
            INNER JOIN
                slick sk ON sts.slick = sk.id AND sk.active = TRUE
            INNER JOIN
                source_type st ON st.id = s.type
            WHERE
                sts.active = TRUE
            ORDER BY sts.slick DESC, sts.rank
    """,
    )
    op.create_entity(source_plus)

    repeat_source = PGView(
        schema="public",
        signature="repeat_source",
        definition="""
            WITH agg AS (
                SELECT
                    s.id AS source_id,
                    count(DISTINCT sl.orchestrator_run) AS occurrence_count,
                    sum(sl.area) / 1000000 AS total_area
                FROM slick_to_source sts
                JOIN source s ON s.id = sts.source
                JOIN slick sl ON sl.id = sts.slick
                LEFT JOIN source_to_tag stt ON stt.source = sts.source
                LEFT JOIN hitl_slick hs ON hs.slick = sl.id
                WHERE true
                    AND sl.active
                    AND sl.cls <> 1
                    AND (hs.cls IS NULL OR hs.cls <> 1)
                    AND sts.active
                    AND sts.hitl_verification IS NOT FALSE
                    AND (
                        (s.type = 2 AND sts.rank = 1)
                        OR
                        (s.type = 1 AND sts.collated_score > 0 AND (stt.tag IS NULL OR (stt.tag <> ALL (ARRAY[5, 6, 7]))))
                    )
                GROUP BY s.id, s.ext_id, s.type
            )
            SELECT agg.source_id,
            agg.occurrence_count,
            agg.total_area,
            row_number() OVER (ORDER BY agg.occurrence_count DESC, agg.total_area DESC) AS global_rank
            FROM agg
            ORDER BY agg.occurrence_count DESC, agg.total_area DESC;
        """,
    )
    op.create_entity(repeat_source)
