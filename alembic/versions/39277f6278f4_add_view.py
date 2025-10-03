"""Add view

Revision ID: 39277f6278f4
Revises: 7cd715196b8d
Create Date: 2022-07-01 16:59:54.560440

"""

from alembic_utils.pg_view import PGView

from alembic import op

# revision identifiers, used by Alembic.
revision = "39277f6278f4"
down_revision = "7cd715196b8d"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """add views"""
    slick_plus = PGView(
        schema="public",
        signature="slick_plus",
        definition="""
            WITH base AS (
                SELECT
                    id,
                    slick_timestamp,
                    geometry::geometry,
                    machine_confidence,
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
                CASE
                    WHEN s.type = 1 THEN ('https://cerulean.skytruth.org/sources/'::text || s.ext_id) || '?ref=api'::text
                    WHEN s.type = 2 THEN (('https://cerulean.skytruth.org/?ref=api&source_score=0_Infinity&'::text || st.ext_id_name) || '='::text) || s.ext_id
                    ELSE NULL::text
                END AS source_url
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
                LEFT JOIN source_to_tag stt ON stt.source_ext_id = s.ext_id AND stt.source_type = s.type
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


def downgrade() -> None:
    """remove views"""
    slick_plus = PGView(
        schema="public", signature="slick_plus", definition="// not needed"
    )
    op.drop_entity(slick_plus)

    source_plus = PGView(
        schema="public", signature="source_plus", definition="// not needed"
    )
    op.drop_entity(source_plus)

    repeat_source = PGView(
        schema="public", signature="repeat_source", definition="// not needed"
    )
    op.drop_entity(repeat_source)
