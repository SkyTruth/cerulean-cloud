"""Add slick_to_mpa

Revision ID: 9f7f6a4d4c1b
Revises: 8f0c0f3f1f6d
Create Date: 2026-04-06 12:00:00.000000

"""

from sqlalchemy import text

from alembic import op

# revision identifiers, used by Alembic.
revision = "9f7f6a4d4c1b"
down_revision = "8f0c0f3f1f6d"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add slick_to_mpa and shift aoi_type_3_ids to wdpaid."""
    op.execute(
        """
        CREATE TABLE slick_to_mpa (
            slick bigint NOT NULL
                REFERENCES slick(id) ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED,
            wdpaid integer NOT NULL,
            PRIMARY KEY (slick, wdpaid)
        );

        CREATE INDEX idx_slick_to_mpa_slick ON slick_to_mpa (slick);
        CREATE INDEX idx_slick_to_mpa_wdpaid ON slick_to_mpa (wdpaid);

        INSERT INTO slick_to_mpa (slick, wdpaid)
        SELECT DISTINCT sta.slick, am.wdpaid
        FROM slick_to_aoi sta
        JOIN aoi a ON a.id = sta.aoi
        JOIN aoi_mpa am ON am.aoi_id = a.id
        WHERE a.type = 3
        AND am.wdpaid IS NOT NULL
        ON CONFLICT DO NOTHING;
        """
    )

    op.execute(
        text(
            """
        CREATE OR REPLACE FUNCTION sync_slick_to_mpa_from_aoi_func()
        RETURNS trigger
        AS $$
        BEGIN
            IF TG_OP = 'INSERT' THEN
                INSERT INTO slick_to_mpa (slick, wdpaid)
                SELECT NEW.slick, am.wdpaid
                FROM aoi a
                JOIN aoi_mpa am ON am.aoi_id = a.id
                WHERE a.id = NEW.aoi
                AND a.type = 3
                AND am.wdpaid IS NOT NULL
                ON CONFLICT DO NOTHING;
                RETURN NEW;
            ELSIF TG_OP = 'DELETE' THEN
                DELETE FROM slick_to_mpa
                WHERE slick = OLD.slick
                AND wdpaid IN (
                    SELECT am.wdpaid
                    FROM aoi a
                    JOIN aoi_mpa am ON am.aoi_id = a.id
                    WHERE a.id = OLD.aoi
                    AND a.type = 3
                    AND am.wdpaid IS NOT NULL
                );
                RETURN OLD;
            END IF;
            RETURN NULL;
        END;
        $$ LANGUAGE plpgsql;

        DROP TRIGGER IF EXISTS sync_slick_to_mpa_from_aoi_insert ON slick_to_aoi;
        CREATE TRIGGER sync_slick_to_mpa_from_aoi_insert
        AFTER INSERT ON slick_to_aoi
        FOR EACH ROW
        EXECUTE FUNCTION sync_slick_to_mpa_from_aoi_func();

        DROP TRIGGER IF EXISTS sync_slick_to_mpa_from_aoi_delete ON slick_to_aoi;
        CREATE TRIGGER sync_slick_to_mpa_from_aoi_delete
        AFTER DELETE ON slick_to_aoi
        FOR EACH ROW
        EXECUTE FUNCTION sync_slick_to_mpa_from_aoi_func();
        """
        )
    )

    op.execute(
        text(
            """
        CREATE OR REPLACE VIEW public.slick_plus AS
            WITH base AS (
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
                FROM slick
                WHERE active
                AND cls != 1
            )
            SELECT
                base.*,
                sentinel1_grd.scene_id AS s1_scene_id,
                sentinel1_grd.geometry AS s1_geometry,
                hs.cls AS hitl_cls,
                cls.long_name AS hitl_cls_name,
                aois.aoi_type_1_ids,
                aois.aoi_type_2_ids,
                mpas.aoi_type_3_ids,
                srcs.source_type_1_ids,
                srcs.source_type_2_ids,
                srcs.source_type_3_ids,
                srcs.max_source_collated_score,
                'https://cerulean.skytruth.org/slicks/' || base.id || '?ref=api&slick_id=' || base.id
                                    AS slick_url
            FROM base
            JOIN orchestrator_run ON orchestrator_run.id = base.orchestrator_run
            JOIN sentinel1_grd ON sentinel1_grd.id = orchestrator_run.sentinel1_grd

            LEFT JOIN LATERAL (
                SELECT hs.cls
                FROM   hitl_slick hs
                WHERE  hs.slick = base.id
                ORDER  BY hs.update_time DESC
                LIMIT  1
            ) AS hs ON TRUE
            LEFT JOIN cls ON cls.id = hs.cls

            LEFT JOIN LATERAL (
                SELECT
                    array_agg(aoi.id) FILTER (WHERE aoi.type = 1) AS aoi_type_1_ids,
                    array_agg(aoi.id) FILTER (WHERE aoi.type = 2) AS aoi_type_2_ids
                FROM slick_to_aoi sta
                JOIN aoi ON aoi.id = sta.aoi
                WHERE sta.slick = base.id
            ) AS aois ON TRUE

            LEFT JOIN LATERAL (
                SELECT array_agg(stm.wdpaid::bigint) AS aoi_type_3_ids
                FROM slick_to_mpa stm
                WHERE stm.slick = base.id
            ) AS mpas ON TRUE

            LEFT JOIN LATERAL (
                SELECT
                    array_agg(src.ext_id) FILTER (WHERE src.type = 1) AS source_type_1_ids,
                    array_agg(src.ext_id) FILTER (WHERE src.type = 2) AS source_type_2_ids,
                    array_agg(src.ext_id) FILTER (WHERE src.type = 3) AS source_type_3_ids,
                    MAX(sts.collated_score) AS max_source_collated_score
                FROM slick_to_source sts
                JOIN source src ON src.id = sts.source
                WHERE sts.slick = base.id
                AND sts.active = TRUE
            ) AS srcs ON TRUE

            WHERE (hs.cls IS NULL OR hs.cls != 1);
        """
        )
    )

    op.execute(
        text(
            """
        CREATE OR REPLACE FUNCTION public.get_slicks_by_aoi_or_source(
            aoi_id text DEFAULT 'NULL',
            source_id text DEFAULT 'NULL',
            source_rank integer DEFAULT 1,
            collation_threshold double precision DEFAULT NULL,
            OUT id integer,
            OUT linearity double precision,
            OUT slick_timestamp timestamp without time zone,
            OUT geometry geography,
            OUT machine_confidence double precision,
            OUT length double precision,
            OUT area double precision,
            OUT perimeter double precision,
            OUT centroid geography,
            OUT polsby_popper double precision,
            OUT fill_factor double precision,
            OUT centerlines json,
            OUT aspect_ratio_factor double precision,
            OUT hitl_cls integer,
            OUT hitl_cls_name text,
            OUT s1_scene_id character varying,
            OUT s1_geometry geography,
            OUT aoi_type_1_ids bigint[],
            OUT aoi_type_2_ids bigint[],
            OUT aoi_type_3_ids bigint[],
            OUT source_type_1_ids text[],
            OUT source_type_2_ids text[],
            OUT source_type_3_ids text[],
            OUT max_source_collated_score double precision,
            OUT slick_url text
        )
            RETURNS SETOF record
            LANGUAGE 'sql'
            COST 100
            IMMUTABLE PARALLEL SAFE
            ROWS 1000
        AS $BODY$
            select distinct on (sp.id)
                sp.id,
                sp.linearity,
                sp.slick_timestamp,
                sp.geometry,
                sp.machine_confidence,
                sp.length,
                sp.area,
                sp.perimeter,
                sp.centroid,
                sp.polsby_popper,
                sp.fill_factor,
                sp.centerlines,
                sp.aspect_ratio_factor,
                sp.hitl_cls,
                sp.hitl_cls_name,
                sp.s1_scene_id,
                sp.s1_geometry,
                sp.aoi_type_1_ids,
                sp.aoi_type_2_ids,
                sp.aoi_type_3_ids,
                sp.source_type_1_ids,
                sp.source_type_2_ids,
                sp.source_type_3_ids,
                sp.max_source_collated_score,
                sp.slick_url
            FROM public.slick_plus sp
            LEFT JOIN slick_to_source sts
                   ON sts.slick  = sp.id
                  AND source_id  != 'NULL'
                  AND sts.active
            LEFT JOIN slick_to_aoi sta
                   ON sta.slick  = sp.id
                  AND aoi_id     != 'NULL'
            LEFT JOIN slick_to_mpa stm
                   ON stm.slick  = sp.id
                  AND aoi_id     != 'NULL'
            WHERE  (source_id = 'NULL'
                    OR sts.source = ANY (string_to_array(source_id, ',')::int[])
                    AND sts.rank  <= source_rank)
              AND  (
                    aoi_id = 'NULL'
                    OR sta.aoi = ANY (string_to_array(aoi_id, ',')::int[])
                    OR stm.wdpaid = ANY (string_to_array(aoi_id, ',')::int[])
              )
              AND  (collation_threshold IS NULL
                    OR sp.max_source_collated_score >= collation_threshold);
        $BODY$;
        """
        )
    )

    op.execute(
        text(
            """
        CREATE OR REPLACE FUNCTION public.get_slicks_by_source(
            source_id text,
            source_rank integer DEFAULT 1,
            collation_threshold double precision DEFAULT NULL,
            OUT id integer,
            OUT linearity double precision,
            OUT slick_timestamp timestamp without time zone,
            OUT geometry geography,
            OUT machine_confidence double precision,
            OUT length double precision,
            OUT area double precision,
            OUT perimeter double precision,
            OUT centroid geography,
            OUT polsby_popper double precision,
            OUT fill_factor double precision,
            OUT centerlines json,
            OUT aspect_ratio_factor double precision,
            OUT hitl_cls integer,
            OUT hitl_cls_name text,
            OUT s1_scene_id character varying,
            OUT s1_geometry geography,
            OUT aoi_type_1_ids bigint[],
            OUT aoi_type_2_ids bigint[],
            OUT aoi_type_3_ids bigint[],
            OUT source_type_1_ids text[],
            OUT source_type_2_ids text[],
            OUT source_type_3_ids text[],
            OUT max_source_collated_score double precision,
            OUT slick_url text
        )
            RETURNS SETOF record
            LANGUAGE 'sql'
            COST 100
            IMMUTABLE PARALLEL SAFE
            ROWS 1000
        AS $BODY$
            select distinct on (sp.id)
                sp.id,
                sp.linearity,
                sp.slick_timestamp,
                sp.geometry,
                sp.machine_confidence,
                sp.length,
                sp.area,
                sp.perimeter,
                sp.centroid,
                sp.polsby_popper,
                sp.fill_factor,
                sp.centerlines,
                sp.aspect_ratio_factor,
                sp.hitl_cls,
                sp.hitl_cls_name,
                sp.s1_scene_id,
                sp.s1_geometry,
                sp.aoi_type_1_ids,
                sp.aoi_type_2_ids,
                sp.aoi_type_3_ids,
                sp.source_type_1_ids,
                sp.source_type_2_ids,
                sp.source_type_3_ids,
                sp.max_source_collated_score,
                sp.slick_url
            FROM public.slick_plus sp
            JOIN slick_to_source sts ON sts.slick = sp.id AND sts.active
            WHERE sts.source = ANY(string_to_array(source_id, ',')::int[])
            AND (sts.rank <= source_rank)
            AND (collation_threshold IS NULL OR sp.max_source_collated_score >= collation_threshold);
        $BODY$;
        """
        )
    )

    op.execute(
        text(
            """
        CREATE OR REPLACE FUNCTION public.get_slicks_by_aoi(
            aoi_id text,
            collation_threshold double precision DEFAULT NULL,
            OUT id integer,
            OUT linearity double precision,
            OUT slick_timestamp timestamp without time zone,
            OUT geometry geography,
            OUT machine_confidence double precision,
            OUT length double precision,
            OUT area double precision,
            OUT perimeter double precision,
            OUT centroid geography,
            OUT polsby_popper double precision,
            OUT fill_factor double precision,
            OUT centerlines json,
            OUT aspect_ratio_factor double precision,
            OUT hitl_cls integer,
            OUT hitl_cls_name text,
            OUT s1_scene_id character varying,
            OUT s1_geometry geography,
            OUT aoi_type_1_ids bigint[],
            OUT aoi_type_2_ids bigint[],
            OUT aoi_type_3_ids bigint[],
            OUT source_type_1_ids text[],
            OUT source_type_2_ids text[],
            OUT source_type_3_ids text[],
            OUT max_source_collated_score double precision,
            OUT slick_url text
        )
            RETURNS SETOF record
            LANGUAGE 'sql'
            COST 100
            IMMUTABLE PARALLEL SAFE
            ROWS 1000
        AS $BODY$
            select distinct on (sp.id)
                sp.id,
                sp.linearity,
                sp.slick_timestamp,
                sp.geometry,
                sp.machine_confidence,
                sp.length,
                sp.area,
                sp.perimeter,
                sp.centroid,
                sp.polsby_popper,
                sp.fill_factor,
                sp.centerlines,
                sp.aspect_ratio_factor,
                sp.hitl_cls,
                sp.hitl_cls_name,
                sp.s1_scene_id,
                sp.s1_geometry,
                sp.aoi_type_1_ids,
                sp.aoi_type_2_ids,
                sp.aoi_type_3_ids,
                sp.source_type_1_ids,
                sp.source_type_2_ids,
                sp.source_type_3_ids,
                sp.max_source_collated_score,
                sp.slick_url
            FROM public.slick_plus sp
            LEFT JOIN slick_to_aoi sta ON sta.slick = sp.id
            LEFT JOIN slick_to_mpa stm ON stm.slick = sp.id
            WHERE (
                sta.aoi = ANY(string_to_array(aoi_id, ',')::int[])
                OR stm.wdpaid = ANY(string_to_array(aoi_id, ',')::int[])
            )
            AND (collation_threshold IS NULL OR sp.max_source_collated_score >= collation_threshold);
        $BODY$;
        """
        )
    )


def downgrade() -> None:
    """Remove slick_to_mpa and restore aoi_type_3_ids from slick_to_aoi."""
    op.execute(
        text(
            """
        CREATE OR REPLACE VIEW public.slick_plus AS
            WITH base AS (
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
                FROM slick
                WHERE active
                AND cls != 1
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
                                    AS slick_url
            FROM base
            JOIN orchestrator_run ON orchestrator_run.id = base.orchestrator_run
            JOIN sentinel1_grd ON sentinel1_grd.id = orchestrator_run.sentinel1_grd

            LEFT JOIN LATERAL (
                SELECT hs.cls
                FROM   hitl_slick hs
                WHERE  hs.slick = base.id
                ORDER  BY hs.update_time DESC
                LIMIT  1
            ) AS hs ON TRUE
            LEFT JOIN cls ON cls.id = hs.cls

            LEFT JOIN LATERAL (
                SELECT
                    array_agg(aoi.id) FILTER (WHERE aoi.type = 1) AS aoi_type_1_ids,
                    array_agg(aoi.id) FILTER (WHERE aoi.type = 2) AS aoi_type_2_ids,
                    array_agg(aoi.id) FILTER (WHERE aoi.type = 3) AS aoi_type_3_ids
                FROM   slick_to_aoi sta
                JOIN   aoi ON aoi.id = sta.aoi
                WHERE  sta.slick = base.id
            ) AS aois      ON TRUE

            LEFT JOIN LATERAL (
                SELECT
                    array_agg(src.ext_id) FILTER (WHERE src.type = 1) AS source_type_1_ids,
                    array_agg(src.ext_id) FILTER (WHERE src.type = 2) AS source_type_2_ids,
                    array_agg(src.ext_id) FILTER (WHERE src.type = 3) AS source_type_3_ids,
                    MAX(sts.collated_score) AS max_source_collated_score
                FROM   slick_to_source sts
                JOIN   source src ON src.id = sts.source
                WHERE  sts.slick = base.id
                AND  sts.active = TRUE
            ) AS srcs  ON TRUE

            WHERE (hs.cls IS NULL OR hs.cls != 1);
        """
        )
    )

    op.execute(
        text(
            """
        CREATE OR REPLACE FUNCTION public.get_slicks_by_aoi_or_source(
            aoi_id text DEFAULT 'NULL',
            source_id text DEFAULT 'NULL',
            source_rank integer DEFAULT 1,
            collation_threshold double precision DEFAULT NULL,
            OUT id integer,
            OUT linearity double precision,
            OUT slick_timestamp timestamp without time zone,
            OUT geometry geography,
            OUT machine_confidence double precision,
            OUT length double precision,
            OUT area double precision,
            OUT perimeter double precision,
            OUT centroid geography,
            OUT polsby_popper double precision,
            OUT fill_factor double precision,
            OUT centerlines json,
            OUT aspect_ratio_factor double precision,
            OUT hitl_cls integer,
            OUT hitl_cls_name text,
            OUT s1_scene_id character varying,
            OUT s1_geometry geography,
            OUT aoi_type_1_ids bigint[],
            OUT aoi_type_2_ids bigint[],
            OUT aoi_type_3_ids bigint[],
            OUT source_type_1_ids text[],
            OUT source_type_2_ids text[],
            OUT source_type_3_ids text[],
            OUT max_source_collated_score double precision,
            OUT slick_url text
        )
            RETURNS SETOF record
            LANGUAGE 'sql'
            COST 100
            IMMUTABLE PARALLEL SAFE
            ROWS 1000
        AS $BODY$
            select distinct on (sp.id)
                sp.id,
                sp.linearity,
                sp.slick_timestamp,
                sp.geometry,
                sp.machine_confidence,
                sp.length,
                sp.area,
                sp.perimeter,
                sp.centroid,
                sp.polsby_popper,
                sp.fill_factor,
                sp.centerlines,
                sp.aspect_ratio_factor,
                sp.hitl_cls,
                sp.hitl_cls_name,
                sp.s1_scene_id,
                sp.s1_geometry,
                sp.aoi_type_1_ids,
                sp.aoi_type_2_ids,
                sp.aoi_type_3_ids,
                sp.source_type_1_ids,
                sp.source_type_2_ids,
                sp.source_type_3_ids,
                sp.max_source_collated_score,
                sp.slick_url
            FROM public.slick_plus sp
            LEFT JOIN slick_to_source sts
                   ON sts.slick  = sp.id
                  AND source_id  != 'NULL'
                  AND sts.active
            LEFT JOIN slick_to_aoi sta
                   ON sta.slick  = sp.id
                  AND aoi_id     != 'NULL'
            WHERE  (source_id = 'NULL'
                    OR sts.source = ANY (string_to_array(source_id, ',')::int[])
                    AND sts.rank  <= source_rank)
              AND  (aoi_id   = 'NULL'
                    OR sta.aoi  = ANY (string_to_array(aoi_id, ',')::int[]))
              AND  (collation_threshold IS NULL
                    OR sp.max_source_collated_score >= collation_threshold);
        $BODY$;
        """
        )
    )

    op.execute(
        text(
            """
        DROP TRIGGER IF EXISTS sync_slick_to_mpa_from_aoi_insert ON slick_to_aoi;
        DROP TRIGGER IF EXISTS sync_slick_to_mpa_from_aoi_delete ON slick_to_aoi;
        DROP FUNCTION IF EXISTS sync_slick_to_mpa_from_aoi_func();
        DROP INDEX IF EXISTS idx_slick_to_mpa_wdpaid;
        DROP INDEX IF EXISTS idx_slick_to_mpa_slick;
        DROP TABLE IF EXISTS slick_to_mpa;
        """
        )
    )

    op.execute(
        text(
            """
        CREATE OR REPLACE FUNCTION public.get_slicks_by_source(
            source_id text,
            source_rank integer DEFAULT 1,
            collation_threshold double precision DEFAULT NULL,
            OUT id integer,
            OUT linearity double precision,
            OUT slick_timestamp timestamp without time zone,
            OUT geometry geography,
            OUT machine_confidence double precision,
            OUT length double precision,
            OUT area double precision,
            OUT perimeter double precision,
            OUT centroid geography,
            OUT polsby_popper double precision,
            OUT fill_factor double precision,
            OUT centerlines json,
            OUT aspect_ratio_factor double precision,
            OUT hitl_cls integer,
            OUT hitl_cls_name text,
            OUT s1_scene_id character varying,
            OUT s1_geometry geography,
            OUT aoi_type_1_ids bigint[],
            OUT aoi_type_2_ids bigint[],
            OUT aoi_type_3_ids bigint[],
            OUT source_type_1_ids text[],
            OUT source_type_2_ids text[],
            OUT source_type_3_ids text[],
            OUT max_source_collated_score double precision,
            OUT slick_url text
        )
            RETURNS SETOF record
            LANGUAGE 'sql'
            COST 100
            IMMUTABLE PARALLEL SAFE
            ROWS 1000
        AS $BODY$
            select distinct on (sp.id)
                sp.id,
                sp.linearity,
                sp.slick_timestamp,
                sp.geometry,
                sp.machine_confidence,
                sp.length,
                sp.area,
                sp.perimeter,
                sp.centroid,
                sp.polsby_popper,
                sp.fill_factor,
                sp.centerlines,
                sp.aspect_ratio_factor,
                sp.hitl_cls,
                sp.hitl_cls_name,
                sp.s1_scene_id,
                sp.s1_geometry,
                sp.aoi_type_1_ids,
                sp.aoi_type_2_ids,
                sp.aoi_type_3_ids,
                sp.source_type_1_ids,
                sp.source_type_2_ids,
                sp.source_type_3_ids,
                sp.max_source_collated_score,
                sp.slick_url
            FROM public.slick_plus sp
            JOIN slick_to_source sts ON sts.slick = sp.id AND sts.active
            WHERE sts.source = ANY(string_to_array(source_id, ',')::int[])
            AND (sts.rank <= source_rank)
            AND (collation_threshold IS NULL OR sp.max_source_collated_score >= collation_threshold);
        $BODY$;
        """
        )
    )

    op.execute(
        text(
            """
        CREATE OR REPLACE FUNCTION public.get_slicks_by_aoi(
            aoi_id text,
            collation_threshold double precision DEFAULT NULL,
            OUT id integer,
            OUT linearity double precision,
            OUT slick_timestamp timestamp without time zone,
            OUT geometry geography,
            OUT machine_confidence double precision,
            OUT length double precision,
            OUT area double precision,
            OUT perimeter double precision,
            OUT centroid geography,
            OUT polsby_popper double precision,
            OUT fill_factor double precision,
            OUT centerlines json,
            OUT aspect_ratio_factor double precision,
            OUT hitl_cls integer,
            OUT hitl_cls_name text,
            OUT s1_scene_id character varying,
            OUT s1_geometry geography,
            OUT aoi_type_1_ids bigint[],
            OUT aoi_type_2_ids bigint[],
            OUT aoi_type_3_ids bigint[],
            OUT source_type_1_ids text[],
            OUT source_type_2_ids text[],
            OUT source_type_3_ids text[],
            OUT max_source_collated_score double precision,
            OUT slick_url text
        )
            RETURNS SETOF record
            LANGUAGE 'sql'
            COST 100
            IMMUTABLE PARALLEL SAFE
            ROWS 1000
        AS $BODY$
            select distinct on (sp.id)
                sp.id,
                sp.linearity,
                sp.slick_timestamp,
                sp.geometry,
                sp.machine_confidence,
                sp.length,
                sp.area,
                sp.perimeter,
                sp.centroid,
                sp.polsby_popper,
                sp.fill_factor,
                sp.centerlines,
                sp.aspect_ratio_factor,
                sp.hitl_cls,
                sp.hitl_cls_name,
                sp.s1_scene_id,
                sp.s1_geometry,
                sp.aoi_type_1_ids,
                sp.aoi_type_2_ids,
                sp.aoi_type_3_ids,
                sp.source_type_1_ids,
                sp.source_type_2_ids,
                sp.source_type_3_ids,
                sp.max_source_collated_score,
                sp.slick_url
            FROM public.slick_plus sp
            JOIN slick_to_aoi sta ON sta.slick = sp.id
            WHERE sta.aoi = ANY(string_to_array(aoi_id, ',')::int[])
            AND (collation_threshold IS NULL OR sp.max_source_collated_score >= collation_threshold);
        $BODY$;
        """
        )
    )
