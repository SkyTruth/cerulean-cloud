"""Add exposed funcs

Revision ID: 3736e85bc273
Revises: f9b7166c86b7
Create Date: 2023-07-18 01:07:58.731501

"""

from alembic_utils.pg_function import PGFunction

from alembic import op

# revision identifiers, used by Alembic.
revision = "3736e85bc273"
down_revision = "f9b7166c86b7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add funcs"""
    op.execute(
        """
        CREATE OR REPLACE FUNCTION get_slick_subclses(cls_id bigint)
        RETURNS SETOF cls AS $$
        BEGIN
            RETURN QUERY (
                WITH RECURSIVE recurse_clses_cte AS (
                    SELECT * FROM cls WHERE id = cls_id
                    UNION
                    SELECT nextcls.* FROM cls nextcls
                    JOIN recurse_clses_cte rc ON nextcls.supercls = rc.id
                )
                SELECT * FROM recurse_clses_cte
            );
        END;
        $$ LANGUAGE plpgsql;
        """
    )

    op.execute(
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
            OUT active boolean,
            OUT orchestrator_run integer,
            OUT create_time timestamp without time zone,
            OUT inference_idx integer,
            OUT cls integer,
            OUT hitl_cls integer,
            OUT machine_confidence double precision,
            OUT length double precision,
            OUT area double precision,
            OUT perimeter double precision,
            OUT centroid geography,
            OUT polsby_popper double precision,
            OUT fill_factor double precision,
            OUT s1_scene_id character varying,
            OUT s1_geometry geography,
            OUT cls_short_name text,
            OUT cls_long_name text,
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
            select distinct
                sp.id,
                sp.linearity,
                sp.slick_timestamp,
                sp.geometry,
                sp.active,
                sp.orchestrator_run,
                sp.create_time,
                sp.inference_idx,
                sp.cls,
                sp.hitl_cls,
                sp.machine_confidence,
                sp.length,
                sp.area,
                sp.perimeter,
                sp.centroid,
                sp.polsby_popper,
                sp.fill_factor,
                sp.s1_scene_id,
                sp.s1_geometry,
                sp.cls_short_name,
                sp.cls_long_name,
                sp.aoi_type_1_ids,
                sp.aoi_type_2_ids,
                sp.aoi_type_3_ids,
                sp.source_type_1_ids,
                sp.source_type_2_ids,
                sp.source_type_3_ids,
                sp.max_source_collated_score,
                sp.slick_url
            FROM public.slick_plus sp
            LEFT JOIN slick_to_source sts ON sts.slick = sp.id AND source_id != 'NULL' AND sts.active
            LEFT JOIN slick_to_aoi sta ON sta.slick = sp.id AND aoi_id != 'NULL'
            WHERE (source_id = 'NULL' OR sts.source = ANY(string_to_array(source_id, ',')::int[]) AND sts.rank <= source_rank)
            AND (aoi_id = 'NULL' OR sta.aoi = ANY(string_to_array(aoi_id, ',')::int[]))
            AND (collation_threshold IS NULL OR sp.max_source_collated_score >= collation_threshold);
        $BODY$;
        """
    )

    op.execute(
        """
        CREATE OR REPLACE FUNCTION public.get_slicks_by_source(
            source_id text,
            source_rank integer DEFAULT 1,
            collation_threshold double precision DEFAULT NULL,
            OUT id integer,
            OUT linearity double precision,
            OUT slick_timestamp timestamp without time zone,
            OUT geometry geography,
            OUT active boolean,
            OUT orchestrator_run integer,
            OUT create_time timestamp without time zone,
            OUT inference_idx integer,
            OUT cls integer,
            OUT hitl_cls integer,
            OUT machine_confidence double precision,
            OUT length double precision,
            OUT area double precision,
            OUT perimeter double precision,
            OUT centroid geography,
            OUT polsby_popper double precision,
            OUT fill_factor double precision,
            OUT s1_scene_id character varying,
            OUT s1_geometry geography,
            OUT cls_short_name text,
            OUT cls_long_name text,
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
            select distinct
                sp.id,
                sp.linearity,
                sp.slick_timestamp,
                sp.geometry,
                sp.active,
                sp.orchestrator_run,
                sp.create_time,
                sp.inference_idx,
                sp.cls,
                sp.hitl_cls,
                sp.machine_confidence,
                sp.length,
                sp.area,
                sp.perimeter,
                sp.centroid,
                sp.polsby_popper,
                sp.fill_factor,
                sp.s1_scene_id,
                sp.s1_geometry,
                sp.cls_short_name,
                sp.cls_long_name,
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

    op.execute(
        """
        CREATE OR REPLACE FUNCTION public.get_slicks_by_aoi(
            aoi_id text,
            collation_threshold double precision DEFAULT NULL,
            OUT id integer,
            OUT linearity double precision,
            OUT slick_timestamp timestamp without time zone,
            OUT geometry geography,
            OUT active boolean,
            OUT orchestrator_run integer,
            OUT create_time timestamp without time zone,
            OUT inference_idx integer,
            OUT cls integer,
            OUT hitl_cls integer,
            OUT machine_confidence double precision,
            OUT length double precision,
            OUT area double precision,
            OUT perimeter double precision,
            OUT centroid geography,
            OUT polsby_popper double precision,
            OUT fill_factor double precision,
            OUT s1_scene_id character varying,
            OUT s1_geometry geography,
            OUT cls_short_name text,
            OUT cls_long_name text,
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
            select distinct
                sp.id,
                sp.linearity,
                sp.slick_timestamp,
                sp.geometry,
                sp.active,
                sp.orchestrator_run,
                sp.create_time,
                sp.inference_idx,
                sp.cls,
                sp.hitl_cls,
                sp.machine_confidence,
                sp.length,
                sp.area,
                sp.perimeter,
                sp.centroid,
                sp.polsby_popper,
                sp.fill_factor,
                sp.s1_scene_id,
                sp.s1_geometry,
                sp.cls_short_name,
                sp.cls_long_name,
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

    # Fiddle: https://dbfiddle.uk/?rdbms=postgres_14&fiddle=a78602f74ea6c2d87a9fa82f1b3a5868
    get_history_slick = PGFunction(
        schema="public",
        signature="slick_history(_slick_id INT)",
        definition="""
        RETURNS TABLE(id integer, precursor_slicks integer ARRAY, create_time timestamp, active BOOLEAN) as
        $$
        WITH RECURSIVE ctename AS (
            SELECT id, precursor_slicks, create_time, active
            FROM slick
            WHERE id = _slick_id
            UNION ALL
            SELECT slick.id, slick.precursor_slicks, slick.create_time, slick.active
            FROM slick
            JOIN ctename ON slick.id = ANY(ctename.precursor_slicks)
        )
        SELECT * FROM ctename;
        $$ language SQL
        """,
    )
    op.create_entity(get_history_slick)


def downgrade() -> None:
    """Add funcs"""
    op.execute(
        """
        DROP FUNCTION IF EXISTS get_slick_subclses(bigint);
        """
    )
    op.execute(
        """
        DROP FUNCTION public.get_slicks_by_source(text, integer);
        """
    )
    op.execute(
        """
        DROP FUNCTION public.get_slicks_by_aoi(text);
        """
    )
    op.execute(
        """
        DROP FUNCTION public.get_slicks_by_aoi_or_source(text, text, integer);
        """
    )
    get_history_slick = PGFunction(
        schema="public",
        signature="slick_history(_slick_id INT)",
        definition="// not needed",
    )
    op.drop_entity(get_history_slick)
