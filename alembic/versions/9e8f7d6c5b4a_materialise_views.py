"""Materialise and index the *_plus views with change-capture refresh.

Revision ID: 9e8f7d6c5b4a
Revises: b1a2c3d4e5f6
Create Date: 2025-05-31 14:42:00
"""

from alembic import op

revision = "9e8f7d6c5b4a"
down_revision = "b1a2c3d4e5f6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ------------------------------------------------------------------ #
    # 0. Drop the plain views (defined in 39277f6278f4) – they’ll be
    #    replaced by materialised views.
    # ------------------------------------------------------------------ #
    for v in ("repeat_source", "source_plus", "slick_plus"):
        op.execute(f"DROP VIEW IF EXISTS {v} CASCADE")

    # ------------------------------------------------------------------ #
    # 1. Guardrail: one *active* row per (slick, source)
    # ------------------------------------------------------------------ #
    op.execute("""
        ALTER TABLE slick_to_source
        ADD CONSTRAINT IF NOT EXISTS slick_to_source_one_active_per_pair
        UNIQUE (slick, source)
        WHERE active
    """)

    # ------------------------------------------------------------------ #
    # 2. Change-capture table & trigger
    # ------------------------------------------------------------------ #
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS slick_plus_changes (
            slick_id bigint PRIMARY KEY
        );
        """
    )

    op.execute(
        """
        CREATE OR REPLACE FUNCTION log_slick_change() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            INSERT INTO slick_plus_changes (slick_id)
            VALUES (COALESCE(NEW.id, OLD.id))
            ON CONFLICT DO NOTHING;
            RETURN NULL;
        END $$;
        """
    )

    for tbl in ("slick", "slick_to_aoi", "slick_to_source"):
        op.execute(
            f"""
            DROP TRIGGER IF EXISTS log_change_{tbl} ON {tbl};
            CREATE TRIGGER log_change_{tbl}
            AFTER INSERT OR DELETE OR UPDATE ON {tbl}
            FOR EACH ROW EXECUTE FUNCTION log_slick_change();
            """
        )

    # ------------------------------------------------------------------ #
    # 3. Materialised VIEW slick_plus
    # ------------------------------------------------------------------ #
    op.execute(
        """
        CREATE MATERIALIZED VIEW slick_plus AS
        SELECT
            sl.*,
            sl.length^2 / sl.area / sl.polsby_popper                      AS linearity,
            s1.scene_id                                                   AS s1_scene_id,
            s1.geometry                                                   AS s1_geometry,
            c.short_name                                                  AS cls_short_name,
            c.long_name                                                   AS cls_long_name,
            a.aoi_type_1_ids,
            a.aoi_type_2_ids,
            a.aoi_type_3_ids,
            src.source_type_1_ids,
            src.source_type_2_ids,
            src.source_type_3_ids,
            'https://cerulean.skytruth.org/slicks/'||sl.id||
            '?ref=api&slick_id='||sl.id                                   AS slick_url
        FROM slick             sl
        JOIN orchestrator_run   o   ON o.id           = sl.orchestrator_run
        JOIN sentinel1_grd      s1  ON s1.id          = o.sentinel1_grd
        JOIN cls                c   ON c.id           = sl.cls
        LEFT JOIN LATERAL (
            SELECT
                ARRAY_AGG(a.id) FILTER (WHERE a.type = 1) AS aoi_type_1_ids,
                ARRAY_AGG(a.id) FILTER (WHERE a.type = 2) AS aoi_type_2_ids,
                ARRAY_AGG(a.id) FILTER (WHERE a.type = 3) AS aoi_type_3_ids
            FROM slick_to_aoi sta
            JOIN aoi a ON a.id = sta.aoi
            WHERE sta.slick = sl.id
        ) a ON TRUE
        LEFT JOIN LATERAL (
            SELECT
                ARRAY_AGG(s.id) FILTER (WHERE s.type = 1) AS source_type_1_ids,
                ARRAY_AGG(s.id) FILTER (WHERE s.type = 2) AS source_type_2_ids,
                ARRAY_AGG(s.id) FILTER (WHERE s.type = 3) AS source_type_3_ids
            FROM slick_to_source sts
            JOIN source s ON s.id = sts.source
            WHERE sts.slick  = sl.id
              AND sts.active = TRUE
        ) src ON TRUE
        WHERE sl.active = TRUE;
        """
    )

    # ------------------------------------------------------------------ #
    # 4. Materialised VIEW source_plus
    # ------------------------------------------------------------------ #
    op.execute(
        """
        CREATE MATERIALIZED VIEW source_plus AS
        SELECT
            sk.geometry,
            sts.slick             AS slick_id,
            sk.machine_confidence AS slick_confidence,
            s.id                  AS source_id,
            s.ext_id              AS mmsi_or_structure_id,
            st.short_name         AS source_type,
            sts.collated_score    AS source_collated_score,
            sts.rank              AS source_rank,
            sts.create_time,
            sts.git_hash          AS git_tag,
            'https://cerulean.skytruth.org/slicks/'||sk.id||
            '?ref=api&slick_id='||sk.id                                   AS slick_url,
            'https://cerulean.skytruth.org/?ref=api&'||st.ext_id_name||
            '='||s.ext_id                                               AS source_url
        FROM slick_to_source  sts
        JOIN source           s   ON s.id  = sts.source
        JOIN slick            sk  ON sk.id = sts.slick AND sk.active
        JOIN source_type      st  ON st.id = s.type
        WHERE sts.active
        ORDER BY sts.slick DESC, sts.rank;
        """
    )

    # ------------------------------------------------------------------ #
    # 5. Materialised VIEW repeat_source
    # ------------------------------------------------------------------ #
    op.execute(
        """
        CREATE MATERIALIZED VIEW repeat_source AS
        SELECT
            agg.source_id,
            agg.occurrence_count,
            agg.total_area,
            ROW_NUMBER() OVER (ORDER BY agg.occurrence_count DESC,
                                        agg.total_area      DESC)        AS global_rank
        FROM (
            SELECT
                s.id                                  AS source_id,
                COUNT(DISTINCT sl.orchestrator_run)   AS occurrence_count,
                SUM(sl.area) / 1e6                    AS total_area
            FROM slick_to_source sts
            JOIN source        s  ON s.id = sts.source
            JOIN slick         sl ON sl.id = sts.slick
            LEFT JOIN source_to_tag stt ON stt.source = sts.source
            LEFT JOIN hitl_slick   hs  ON hs.slick  = sl.id
            WHERE sl.active
              AND sl.cls <> 1
              AND (hs.cls IS NULL OR hs.cls <> 1)
              AND sts.active
              AND sts.hitl_verification IS NOT FALSE
              AND (
                     (s.type = 2 AND sts.rank = 1)
                     OR
                     (s.type = 1 AND sts.collated_score > 0
                      AND (stt.tag IS NULL
                           OR stt.tag <> ALL (ARRAY[5,6,7])))
                  )
            GROUP BY s.id
        ) agg;
        """
    )

    # ------------------------------------------------------------------ #
    # 6. Indexes on the materialised views
    # ------------------------------------------------------------------ #
    op.execute(
        """
        CREATE UNIQUE INDEX mv_slick_plus_pk  ON slick_plus(id);
        CLUSTER slick_plus USING mv_slick_plus_pk;
        CREATE INDEX mv_slick_plus_geom_gist  ON slick_plus USING GIST(geometry);

        CREATE UNIQUE INDEX mv_source_plus_pk ON source_plus(slick_id, source_id);
        CREATE INDEX mv_source_plus_order     ON source_plus(slick_id DESC, source_rank);

        CREATE UNIQUE INDEX mv_repeat_source_pk ON repeat_source(source_id);
        CLUSTER repeat_source USING mv_repeat_source_pk;
        """
    )

    # ------------------------------------------------------------------ #
    # 7. Incremental refresh procedure
    # ------------------------------------------------------------------ #
    op.execute(
        """
        CREATE OR REPLACE PROCEDURE refresh_slick_views()
        LANGUAGE plpgsql AS $$
        BEGIN
            -- Harvest changed slick ids
            CREATE TEMP TABLE _chg AS
              SELECT slick_id FROM slick_plus_changes;
            TRUNCATE slick_plus_changes;

            IF NOT EXISTS (SELECT 1 FROM _chg) THEN
                RETURN;
            END IF;

            REFRESH MATERIALIZED VIEW CONCURRENTLY slick_plus;
            REFRESH MATERIALIZED VIEW CONCURRENTLY source_plus;
            REFRESH MATERIALIZED VIEW CONCURRENTLY repeat_source;
        END $$;
        """
    )


def downgrade() -> None:
    # ------------------------------------------------------------------ #
    # Drop the incremental refresh objects
    # ------------------------------------------------------------------ #
    op.execute("DROP PROCEDURE IF EXISTS refresh_slick_views() CASCADE")
    op.execute("DROP TABLE IF EXISTS slick_plus_changes CASCADE")

    # ------------------------------------------------------------------ #
    # Drop the materialised views
    # ------------------------------------------------------------------ #
    for v in ("repeat_source", "source_plus", "slick_plus"):
        op.execute(f"DROP MATERIALIZED VIEW IF EXISTS {v}")

    # ------------------------------------------------------------------ #
    # Drop the partial-unique constraint
    # ------------------------------------------------------------------ #
    op.execute("""
        ALTER TABLE slick_to_source
        DROP CONSTRAINT IF EXISTS slick_to_source_one_active_per_pair
    """)
