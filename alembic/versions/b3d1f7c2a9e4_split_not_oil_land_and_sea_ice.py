"""Split not-oil, land, sea ice, and artefact classes

Revision ID: b3d1f7c2a9e4
Revises: 8f0c0f3f1f6d
Create Date: 2026-04-03 12:00:00.000000

"""

from sqlalchemy import text

from alembic import op

# revision identifiers, used by Alembic.
revision = "b3d1f7c2a9e4"
down_revision = "8f0c0f3f1f6d"
branch_labels = None
depends_on = None

CLASS_PARENT_ID = 1
NOT_OIL_DESCRIPTION = (
    "Detections that are not oil (e.g. wind shadow, weather, ice, visual artifacts, "
    "over land, internal waves, etc.)"
)
BACKGROUND_DESCRIPTION = (
    "Detections that are unlikely to be slicks (e.g. wind shadow, weather, ice, "
    "visual artifacts, over land, internal waves, etc.)"
)
NOT_OIL_CHILDREN = (
    (
        "LAND",
        "Land",
        "Detections that intersect land and should not be treated as oil.",
    ),
    (
        "SEA_ICE",
        "Sea Ice",
        "Detections that intersect sea ice and should not be treated as oil.",
    ),
    (
        "ARTEFACT",
        "Artefact",
        "Detections that are visual or processing artefacts, such as scene-edge "
        "effects.",
    ),
)
NOT_OIL_CHILDREN_SQL = ", ".join(
    f"'{short_name}'" for short_name, _, _ in NOT_OIL_CHILDREN
)
CLS_UPSERT_SQL = text(
    """
    INSERT INTO cls (short_name, long_name, supercls, description)
    VALUES (:short_name, :long_name, :supercls, :description)
    ON CONFLICT (short_name) DO UPDATE
    SET long_name = EXCLUDED.long_name,
        supercls = EXCLUDED.supercls,
        description = EXCLUDED.description
    """
)
DEPENDENT_SLICK_PLUS_FUNCTIONS = (
    "public.get_slicks_by_source(text, integer, double precision)",
    "public.get_slicks_by_aoi(text, double precision)",
    "public.get_slicks_by_aoi_or_source(text, text, integer, double precision)",
)
EXPOSED_SLICK_PLUS_FUNCTIONS_SQL = """
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
TRIGGER_CLS_EXPR_NOT_OIL = """
                CASE
                    WHEN value = 'BACKGROUND' THEN 'NOT_OIL'
                    ELSE value
                END
"""
TRIGGER_CLS_EXPR_BACKGROUND = "value"


def _upsert_cls(
    bind,
    short_name: str,
    long_name: str,
    description: str,
    supercls: int = CLASS_PARENT_ID,
) -> None:
    bind.execute(
        CLS_UPSERT_SQL,
        {
            "short_name": short_name,
            "long_name": long_name,
            "supercls": supercls,
            "description": description,
        },
    )


def _slick_trigger_func(cls_name_expr: str) -> str:
    return f"""
        CREATE OR REPLACE FUNCTION slick_before_trigger_func()
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
                    WHERE cls.short_name = {cls_name_expr}
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


def _slick_plus_view(
    base_cls_filter: str,
    hitl_cls_filter: str,
    include_not_oil_cte: bool = False,
) -> str:
    cte_sql = (
        f"""
        WITH not_oil_clses AS (
            SELECT id
            FROM public.get_slick_subclses({CLASS_PARENT_ID})
        ),
        base AS (
"""
        if include_not_oil_cte
        else """
        WITH base AS (
"""
    )
    return f"""
        CREATE OR REPLACE VIEW slick_plus AS
{cte_sql}
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
              AND {base_cls_filter}
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
            FROM hitl_slick hs
            WHERE hs.slick = base.id
            ORDER BY hs.update_time DESC
            LIMIT 1
        ) AS hs ON TRUE
        LEFT JOIN cls ON cls.id = hs.cls
        LEFT JOIN LATERAL (
            SELECT
                array_agg(aoi.id) FILTER (WHERE aoi.type = 1) AS aoi_type_1_ids,
                array_agg(aoi.id) FILTER (WHERE aoi.type = 2) AS aoi_type_2_ids,
                array_agg(aoi.id) FILTER (WHERE aoi.type = 3) AS aoi_type_3_ids
            FROM slick_to_aoi sta
            JOIN aoi ON aoi.id = sta.aoi
            WHERE sta.slick = base.id
        ) AS aois ON TRUE
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
        WHERE {hitl_cls_filter};
    """


def _repeat_source_view(
    slick_cls_filter: str,
    hitl_cls_filter: str,
    include_not_oil_cte: bool = False,
) -> str:
    cte_sql = (
        f"""
        WITH not_oil_clses AS (
            SELECT id
            FROM public.get_slick_subclses({CLASS_PARENT_ID})
        ),
        agg AS (
"""
        if include_not_oil_cte
        else """
        WITH agg AS (
"""
    )
    return f"""
        CREATE OR REPLACE VIEW repeat_source AS
{cte_sql}
            SELECT
                s.id AS source_id,
                count(DISTINCT sl.orchestrator_run) AS occurrence_count,
                sum(sl.area) / 1000000 AS total_area
            FROM slick_to_source sts
            JOIN source s ON s.id = sts.source
            JOIN slick sl ON sl.id = sts.slick
            LEFT JOIN source_to_tag stt ON stt.source_ext_id = s.ext_id AND stt.source_type = s.type
            LEFT JOIN hitl_slick hs ON hs.slick = sl.id
            WHERE true
                AND sl.active
                AND {slick_cls_filter}
                AND ({hitl_cls_filter})
                AND (stt.tag IS NULL OR stt.tag <> 12)
                AND sts.active
                AND sts.hitl_verification IS NOT FALSE
                AND (
                    (s.type = 2 AND sts.rank = 1)
                    OR (
                        s.type = 1
                        AND (sts.hitl_verification OR sts.collated_score > 0::double precision)
                        AND (stt.tag IS NULL OR (stt.tag <> ALL (ARRAY[5, 6, 7])))
                    )
                )
            GROUP BY s.id, s.ext_id, s.type
        )
        SELECT
            agg.source_id,
            agg.occurrence_count,
            agg.total_area,
            row_number() OVER (ORDER BY agg.occurrence_count DESC, agg.total_area DESC) AS global_rank
        FROM agg
        ORDER BY agg.occurrence_count DESC, agg.total_area DESC;
    """


def _drop_and_recreate_derived_objects(
    slick_plus_sql: str, repeat_source_sql: str
) -> None:
    for function_signature in DEPENDENT_SLICK_PLUS_FUNCTIONS:
        op.execute(f"DROP FUNCTION IF EXISTS {function_signature}")
    op.execute("DROP VIEW IF EXISTS repeat_source")
    op.execute("DROP VIEW IF EXISTS slick_plus")
    op.execute(slick_plus_sql)
    op.execute(repeat_source_sql)
    op.execute(EXPOSED_SLICK_PLUS_FUNCTIONS_SQL)


def upgrade() -> None:
    """Split not-oil, land, sea ice, and artefact classes."""
    bind = op.get_bind()
    bind.execute(
        text("ALTER TABLE orchestrator_run ADD COLUMN IF NOT EXISTS sea_ice_date DATE")
    )

    bind.execute(
        text(
            f"""
            UPDATE cls
            SET short_name = 'NOT_OIL',
                long_name = 'Not Oil',
                description = :not_oil_description
            WHERE id = {CLASS_PARENT_ID}
            """
        ),
        {"not_oil_description": NOT_OIL_DESCRIPTION},
    )

    for short_name, long_name, description in NOT_OIL_CHILDREN:
        _upsert_cls(bind, short_name, long_name, description)

    op.execute(_slick_trigger_func(TRIGGER_CLS_EXPR_NOT_OIL))
    bind.execute(
        text("UPDATE slick SET cls = :not_oil_id WHERE cls IS NULL"),
        {"not_oil_id": CLASS_PARENT_ID},
    )
    bind.execute(text("ALTER TABLE slick ALTER COLUMN cls SET NOT NULL"))

    _drop_and_recreate_derived_objects(
        _slick_plus_view(
            "cls NOT IN (SELECT id FROM not_oil_clses)",
            "hs.cls IS NULL OR hs.cls NOT IN (SELECT id FROM not_oil_clses)",
            include_not_oil_cte=True,
        ),
        _repeat_source_view(
            "sl.cls NOT IN (SELECT id FROM not_oil_clses)",
            "hs.cls IS NULL OR hs.cls NOT IN (SELECT id FROM not_oil_clses)",
            include_not_oil_cte=True,
        ),
    )


def downgrade() -> None:
    """Restore background-only taxonomy."""
    bind = op.get_bind()

    bind.execute(text("ALTER TABLE slick ALTER COLUMN cls DROP NOT NULL"))

    bind.execute(
        text(
            f"""
            UPDATE slick
            SET cls = {CLASS_PARENT_ID}
            WHERE cls IN (
                SELECT id
                FROM cls
                WHERE short_name IN ({NOT_OIL_CHILDREN_SQL})
            )
            """
        )
    )
    bind.execute(
        text(
            f"""
            UPDATE slick
            SET hitl_cls = {CLASS_PARENT_ID}
            WHERE hitl_cls IN (
                SELECT id
                FROM cls
                WHERE short_name IN ({NOT_OIL_CHILDREN_SQL})
            )
            """
        )
    )
    bind.execute(
        text(
            f"""
            UPDATE hitl_slick
            SET cls = {CLASS_PARENT_ID}
            WHERE cls IN (
                SELECT id
                FROM cls
                WHERE short_name IN ({NOT_OIL_CHILDREN_SQL})
            )
            """
        )
    )
    bind.execute(text(f"DELETE FROM cls WHERE short_name IN ({NOT_OIL_CHILDREN_SQL})"))

    bind.execute(
        text(
            f"""
            UPDATE cls
            SET short_name = 'BACKGROUND',
                long_name = 'Background',
                description = :background_description
            WHERE id = {CLASS_PARENT_ID}
            """
        ),
        {"background_description": BACKGROUND_DESCRIPTION},
    )

    op.execute(_slick_trigger_func(TRIGGER_CLS_EXPR_BACKGROUND))
    _drop_and_recreate_derived_objects(
        _slick_plus_view(
            f"cls != {CLASS_PARENT_ID}",
            f"hs.cls IS NULL OR hs.cls != {CLASS_PARENT_ID}",
        ),
        _repeat_source_view(
            f"sl.cls <> {CLASS_PARENT_ID}",
            f"hs.cls IS NULL OR hs.cls <> {CLASS_PARENT_ID}",
        ),
    )
    bind.execute(
        text("ALTER TABLE orchestrator_run DROP COLUMN IF EXISTS sea_ice_date")
    )
