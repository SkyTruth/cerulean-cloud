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
            array_agg(source.ext_id) FILTER (WHERE source.type = 1) AS source_type_1_ids,
            array_agg(source.ext_id) FILTER (WHERE source.type = 2) AS source_type_2_ids,
            array_agg(source.ext_id) FILTER (WHERE source.type = 3) AS source_type_3_ids
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
