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

NOT_OIL_DESCRIPTION = (
    "Detections that are not oil (e.g. wind shadow, weather, ice, visual artifacts, "
    "over land, internal waves, etc.)"
)
LAND_DESCRIPTION = "Detections that intersect land and should not be treated as oil."
SEA_ICE_DESCRIPTION = (
    "Detections that intersect sea ice and should not be treated as oil."
)
ARTEFACT_DESCRIPTION = (
    "Detections that are visual or processing artefacts, such as scene-edge effects."
)
BACKGROUND_DESCRIPTION = (
    "Detections that are unlikely to be slicks (e.g. wind shadow, weather, ice, "
    "visual artifacts, over land, internal waves, etc.)"
)

SLICK_TRIGGER_FUNC_WITH_NOT_OIL = """
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

SLICK_TRIGGER_FUNC_WITH_BACKGROUND = """
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
                WHERE cls.short_name = value
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

SLICK_PLUS_WITH_NOT_OIL_SUBCLASSES = """
    CREATE OR REPLACE VIEW slick_plus AS
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
        FROM slick
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
    WHERE hs.cls IS NULL OR hs.cls NOT IN (SELECT id FROM not_oil_clses);
"""

REPEAT_SOURCE_WITH_NOT_OIL_SUBCLASSES = """
    CREATE OR REPLACE VIEW repeat_source AS
    WITH not_oil_clses AS (
        SELECT id
        FROM public.get_slick_subclses(1)
    ),
    agg AS (
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
            AND sl.cls NOT IN (SELECT id FROM not_oil_clses)
            AND (hs.cls IS NULL OR hs.cls NOT IN (SELECT id FROM not_oil_clses))
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

SLICK_PLUS_WITH_BACKGROUND_ONLY = """
    CREATE OR REPLACE VIEW slick_plus AS
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
    WHERE hs.cls IS NULL OR hs.cls != 1;
"""

REPEAT_SOURCE_WITH_BACKGROUND_ONLY = """
    CREATE OR REPLACE VIEW repeat_source AS
    WITH agg AS (
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
            AND sl.cls <> 1
            AND (hs.cls IS NULL OR hs.cls <> 1)
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


def upgrade() -> None:
    """Split not-oil, land, sea ice, and artefact classes."""
    bind = op.get_bind()
    bind.execute(
        text("ALTER TABLE orchestrator_run ADD COLUMN IF NOT EXISTS sea_ice_date DATE")
    )

    bind.execute(
        text(
            """
            UPDATE cls
            SET short_name = 'NOT_OIL',
                long_name = 'Not Oil',
                description = :not_oil_description
            WHERE id = 1
            """
        ),
        {"not_oil_description": NOT_OIL_DESCRIPTION},
    )

    land_id = bind.execute(
        text(
            """
            INSERT INTO cls (short_name, long_name, supercls, description)
            VALUES ('LAND', 'Land', 1, :description)
            ON CONFLICT (short_name) DO UPDATE
            SET long_name = EXCLUDED.long_name,
                supercls = EXCLUDED.supercls,
                description = EXCLUDED.description
            RETURNING id
            """
        ),
        {"description": LAND_DESCRIPTION},
    ).scalar_one()

    bind.execute(
        text(
            """
            INSERT INTO cls (short_name, long_name, supercls, description)
            VALUES ('SEA_ICE', 'Sea Ice', 1, :description)
            ON CONFLICT (short_name) DO UPDATE
            SET long_name = EXCLUDED.long_name,
                supercls = EXCLUDED.supercls,
                description = EXCLUDED.description
            """
        ),
        {"description": SEA_ICE_DESCRIPTION},
    )
    bind.execute(
        text(
            """
            INSERT INTO cls (short_name, long_name, supercls, description)
            VALUES ('ARTEFACT', 'Artefact', 1, :description)
            ON CONFLICT (short_name) DO UPDATE
            SET long_name = EXCLUDED.long_name,
                supercls = EXCLUDED.supercls,
                description = EXCLUDED.description
            """
        ),
        {"description": ARTEFACT_DESCRIPTION},
    )

    op.execute(SLICK_TRIGGER_FUNC_WITH_NOT_OIL)

    bind.execute(
        text("UPDATE slick SET cls = :land_id WHERE cls = 1"), {"land_id": land_id}
    )
    bind.execute(text("ALTER TABLE slick ALTER COLUMN cls SET NOT NULL"))

    op.execute(SLICK_PLUS_WITH_NOT_OIL_SUBCLASSES)
    op.execute(REPEAT_SOURCE_WITH_NOT_OIL_SUBCLASSES)


def downgrade() -> None:
    """Restore background-only taxonomy."""
    bind = op.get_bind()

    bind.execute(text("ALTER TABLE slick ALTER COLUMN cls DROP NOT NULL"))

    bind.execute(
        text(
            """
            UPDATE slick
            SET cls = 1
            WHERE cls IN (
                SELECT id
                FROM cls
                WHERE short_name IN ('LAND', 'SEA_ICE', 'ARTEFACT')
            )
            """
        )
    )
    bind.execute(
        text(
            """
            UPDATE slick
            SET hitl_cls = 1
            WHERE hitl_cls IN (
                SELECT id
                FROM cls
                WHERE short_name IN ('LAND', 'SEA_ICE', 'ARTEFACT')
            )
            """
        )
    )
    bind.execute(
        text(
            """
            UPDATE hitl_slick
            SET cls = 1
            WHERE cls IN (
                SELECT id
                FROM cls
                WHERE short_name IN ('LAND', 'SEA_ICE', 'ARTEFACT')
            )
            """
        )
    )
    bind.execute(
        text("DELETE FROM cls WHERE short_name IN ('LAND', 'SEA_ICE', 'ARTEFACT')")
    )

    bind.execute(
        text(
            """
            UPDATE cls
            SET short_name = 'BACKGROUND',
                long_name = 'Background',
                description = :background_description
            WHERE id = 1
            """
        ),
        {"background_description": BACKGROUND_DESCRIPTION},
    )

    op.execute(SLICK_TRIGGER_FUNC_WITH_BACKGROUND)

    op.execute(SLICK_PLUS_WITH_BACKGROUND_ONLY)
    op.execute(REPEAT_SOURCE_WITH_BACKGROUND_ONLY)
    bind.execute(
        text("ALTER TABLE orchestrator_run DROP COLUMN IF EXISTS sea_ice_date")
    )
