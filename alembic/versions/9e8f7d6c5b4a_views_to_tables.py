"""Incrementally-maintained cache tables for *_plus, with race-safe refresh.

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
    # ───────── 0. Remove any legacy views ─────────
    for v in ("repeat_source", "source_plus", "slick_plus"):
        op.execute(f"DROP MATERIALIZED VIEW IF EXISTS {v} CASCADE")
        op.execute(f"DROP VIEW            IF EXISTS {v} CASCADE")

    # ───────── 1. Hot-path indexes on link table ─────────
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS slick_to_source_one_active_per_pair_idx
            ON slick_to_source (slick, source)
            WHERE active;

        CREATE INDEX IF NOT EXISTS idx_slick_to_source_slick_active
            ON slick_to_source (slick)
            WHERE active;
    """)

    # ───────── 2. Δ-log tables & trigger ─────────
    op.execute("""
        CREATE TABLE IF NOT EXISTS slick_plus_changes  (slick_id bigint PRIMARY KEY);
        CREATE TABLE IF NOT EXISTS source_plus_changes (slick_id bigint PRIMARY KEY);
    """)

    op.execute("""
        CREATE OR REPLACE FUNCTION log_mv_change() RETURNS trigger
        LANGUAGE plpgsql AS $$
        DECLARE
            v_slick_id bigint;
        BEGIN
            IF TG_TABLE_NAME = 'slick' THEN
                v_slick_id := COALESCE(NEW.id, OLD.id);
            ELSE
                v_slick_id := COALESCE(NEW.slick, OLD.slick);
            END IF;

            INSERT INTO slick_plus_changes  VALUES (v_slick_id) ON CONFLICT DO NOTHING;

            IF TG_TABLE_NAME IN ('slick', 'slick_to_source') THEN
                INSERT INTO source_plus_changes VALUES (v_slick_id) ON CONFLICT DO NOTHING;
            END IF;

            RETURN NULL;
        END $$;
    """)

    for tbl in ("slick", "slick_to_aoi", "slick_to_source"):
        op.execute(f"""
            DROP TRIGGER IF EXISTS log_change_{tbl} ON {tbl};
            CREATE TRIGGER log_change_{tbl}
            AFTER INSERT OR DELETE OR UPDATE ON {tbl}
            FOR EACH ROW EXECUTE FUNCTION log_mv_change();
        """)

    # ───────── 3. Cache TABLES (empty) ─────────
    op.execute("""
        CREATE TABLE slick_plus AS
        SELECT * FROM (
            SELECT
                sl.*,
                sl.length^2 / sl.area / sl.polsby_popper AS linearity,
                s1.scene_id AS s1_scene_id,
                s1.geometry AS s1_geometry,
                c.short_name AS cls_short_name,
                c.long_name AS cls_long_name,
                NULL::bigint[] AS aoi_type_1_ids,
                NULL::bigint[] AS aoi_type_2_ids,
                NULL::bigint[] AS aoi_type_3_ids,
                NULL::bigint[] AS source_type_1_ids,
                NULL::bigint[] AS source_type_2_ids,
                NULL::bigint[] AS source_type_3_ids,
                ''::text       AS slick_url
            FROM slick sl
            JOIN orchestrator_run o ON o.id=sl.orchestrator_run
            JOIN sentinel1_grd  s1 ON s1.id=o.sentinel1_grd
            JOIN cls           c  ON c.id =sl.cls
            WHERE FALSE
        ) q;
    """)  # schema clone, zero rows

    op.execute("""
        CREATE TABLE source_plus AS
        SELECT * FROM (
            SELECT
                NULL::geography      AS geometry,
                0::bigint           AS slick_id,
                0::numeric          AS slick_confidence,
                0::bigint           AS source_id,
                ''::text            AS mmsi_or_structure_id,
                ''::text            AS source_type,
                0::numeric          AS source_collated_score,
                0::bigint           AS source_rank,
                now()               AS create_time,
                ''::text            AS git_tag,
                ''::text            AS slick_url,
                ''::text            AS source_url
            WHERE FALSE
        ) q;
    """)

    # ───────── 4. repeat_source stays a MATERIALISED VIEW ─────────
    op.execute("""
        CREATE MATERIALIZED VIEW repeat_source AS
        SELECT
            agg.source_id,
            agg.occurrence_count,
            agg.total_area,
            ROW_NUMBER() OVER (ORDER BY agg.occurrence_count DESC,
                                        agg.total_area DESC) AS global_rank
        FROM (
            SELECT
                s.id                                AS source_id,
                COUNT(DISTINCT sl.orchestrator_run) AS occurrence_count,
                SUM(sl.area)/1e6                    AS total_area
            FROM slick_to_source sts
            JOIN source s ON s.id = sts.source
            JOIN slick  sl ON sl.id = sts.slick
            LEFT JOIN source_to_tag stt ON stt.source = sts.source
            LEFT JOIN hitl_slick   hs  ON hs.slick  = sl.id
            WHERE sl.active
              AND sl.cls <> 1
              AND (hs.cls IS NULL OR hs.cls <> 1)
              AND sts.active
              AND sts.hitl_verification IS NOT FALSE
              AND (
                     (s.type = 2 AND sts.rank = 1) OR
                     (s.type = 1 AND sts.collated_score > 0
                      AND (stt.tag IS NULL OR stt.tag <> ALL(ARRAY[5,6,7])))
                  )
            GROUP BY s.id
        ) agg;
    """)

    # ───────── 5. Indexes on cache tables & MV ─────────
    op.execute("""
        ALTER TABLE slick_plus  ADD CONSTRAINT slick_plus_pk  PRIMARY KEY(id);
        CREATE INDEX  slick_plus_geom_gist  ON slick_plus USING GIST(geometry);
        CLUSTER slick_plus USING slick_plus_pk;

        ALTER TABLE source_plus ADD PRIMARY KEY (slick_id, source_id);
        CREATE INDEX source_plus_order ON source_plus(slick_id DESC, source_rank);

        CREATE UNIQUE INDEX mv_repeat_source_pk ON repeat_source(source_id);
        CLUSTER repeat_source USING mv_repeat_source_pk;
    """)

    # ───────── 6. Utility functions for incremental & first-time rebuilds ─────────
    op.execute("""
        CREATE OR REPLACE FUNCTION _rebuild_slick_plus_rows(p_ids bigint[])
        RETURNS void LANGUAGE plpgsql AS $$
        BEGIN
            IF p_ids IS NULL OR array_length(p_ids,1) IS NULL THEN
                DELETE FROM slick_plus;                                   -- full reset
                p_ids := ARRAY(SELECT id FROM slick WHERE active);
            END IF;

            DELETE FROM slick_plus WHERE id = ANY(p_ids);

            INSERT INTO slick_plus
            SELECT
                sl.*,
                sl.length^2 / sl.area / sl.polsby_popper                      AS linearity,
                s1.scene_id                                                   AS s1_scene_id,
                s1.geometry                                                   AS s1_geometry,
                c.short_name                                                  AS cls_short_name,
                c.long_name                                                   AS cls_long_name,
                a.aoi_type_1_ids, a.aoi_type_2_ids, a.aoi_type_3_ids,
                src.source_type_1_ids, src.source_type_2_ids, src.source_type_3_ids,
                'https://cerulean.skytruth.org/slicks/'||sl.id||
                '?ref=api&slick_id='||sl.id                                   AS slick_url
            FROM slick sl
            JOIN orchestrator_run o  ON o.id  = sl.orchestrator_run
            JOIN sentinel1_grd  s1 ON s1.id = o.sentinel1_grd
            JOIN cls           c  ON c.id  = sl.cls
            LEFT JOIN LATERAL (
                SELECT
                  ARRAY_AGG(a.id) FILTER (WHERE a.type = 1) AS aoi_type_1_ids,
                  ARRAY_AGG(a.id) FILTER (WHERE a.type = 2) AS aoi_type_2_ids,
                  ARRAY_AGG(a.id) FILTER (WHERE a.type = 3) AS aoi_type_3_ids
                FROM slick_to_aoi sta JOIN aoi a ON a.id = sta.aoi
                WHERE sta.slick = sl.id
            ) a ON TRUE
            LEFT JOIN LATERAL (
                SELECT
                  ARRAY_AGG(s.id) FILTER (WHERE s.type = 1) AS source_type_1_ids,
                  ARRAY_AGG(s.id) FILTER (WHERE s.type = 2) AS source_type_2_ids,
                  ARRAY_AGG(s.id) FILTER (WHERE s.type = 3) AS source_type_3_ids
                FROM slick_to_source sts JOIN source s ON s.id = sts.source
                WHERE sts.slick = sl.id AND sts.active
            ) src ON TRUE
            WHERE sl.id = ANY(p_ids) AND sl.active;
        END $$;
    """)

    op.execute("""
        CREATE OR REPLACE FUNCTION _rebuild_source_plus_rows(p_ids bigint[])
        RETURNS void LANGUAGE plpgsql AS $$
        BEGIN
            IF p_ids IS NULL OR array_length(p_ids,1) IS NULL THEN
                DELETE FROM source_plus;                                    -- full reset
                p_ids := ARRAY(
                    SELECT DISTINCT slick
                    FROM slick_to_source sts
                    JOIN slick sk ON sk.id = sts.slick
                    WHERE sk.active AND sts.active
                );
            END IF;

            DELETE FROM source_plus WHERE slick_id = ANY(p_ids);

            INSERT INTO source_plus
            SELECT
                sk.geometry::geography,
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
            FROM slick_to_source sts
            JOIN source      s  ON s.id  = sts.source
            JOIN slick       sk ON sk.id = sts.slick
            JOIN source_type st ON st.id = s.type
            WHERE sts.slick = ANY(p_ids)
              AND sk.active
              AND sts.active;
        END $$;
    """)

    # ───────── 7A. Orchestrator pipeline ─────────
    op.execute("""
        CREATE OR REPLACE PROCEDURE refresh_slick_plus()
        LANGUAGE plpgsql AS $$
        DECLARE v_ids bigint[];
        BEGIN
            PERFORM pg_advisory_lock(54321,1);

            SELECT ARRAY(SELECT slick_id FROM slick_plus_changes) INTO v_ids;
            IF v_ids IS NOT NULL AND array_length(v_ids,1) > 0 THEN
                TRUNCATE slick_plus_changes;
                PERFORM _rebuild_slick_plus_rows(v_ids);
            END IF;

            PERFORM pg_advisory_unlock(54321,1);
        END $$;
    """)

    # ───────── 7B. ASA pipeline ─────────
    op.execute("""
        CREATE OR REPLACE PROCEDURE refresh_source_views()
        LANGUAGE plpgsql AS $$
        DECLARE v_ids bigint[];
        BEGIN
            PERFORM pg_advisory_lock(54321,2);

            SELECT ARRAY(SELECT slick_id FROM source_plus_changes) INTO v_ids;
            IF v_ids IS NOT NULL AND array_length(v_ids,1) > 0 THEN
                TRUNCATE source_plus_changes;
                PERFORM _rebuild_source_plus_rows(v_ids);
                REFRESH MATERIALIZED VIEW CONCURRENTLY repeat_source;
            END IF;

            PERFORM pg_advisory_unlock(54321,2);
        END $$;
    """)

    # ───────── 7C. Convenience wrapper ─────────
    op.execute("""
        CREATE OR REPLACE PROCEDURE refresh_all_slick_views()
        LANGUAGE plpgsql AS $$
        BEGIN
            PERFORM refresh_slick_plus();
            PERFORM refresh_source_views();
        END $$;
    """)

    # ───────── 8. First-time population ─────────
    op.execute("SELECT _rebuild_slick_plus_rows(NULL)")
    op.execute("SELECT _rebuild_source_plus_rows(NULL)")
    op.execute("REFRESH MATERIALIZED VIEW repeat_source")


def downgrade() -> None:
    op.execute("""
        DROP PROCEDURE IF EXISTS refresh_all_slick_views()   CASCADE;
        DROP PROCEDURE IF EXISTS refresh_source_views()      CASCADE;
        DROP PROCEDURE IF EXISTS refresh_slick_plus()        CASCADE;
        DROP FUNCTION  IF EXISTS _rebuild_source_plus_rows(bigint[]);
        DROP FUNCTION  IF EXISTS _rebuild_slick_plus_rows(bigint[]);

        DROP MATERIALIZED VIEW IF EXISTS repeat_source;

        DROP TABLE IF EXISTS source_plus;
        DROP TABLE IF EXISTS slick_plus;

        DROP TRIGGER IF EXISTS log_change_slick_to_source ON slick_to_source;
        DROP TRIGGER IF EXISTS log_change_slick_to_aoi   ON slick_to_aoi;
        DROP TRIGGER IF EXISTS log_change_slick          ON slick;
        DROP FUNCTION IF EXISTS log_mv_change();

        DROP TABLE IF EXISTS source_plus_changes;
        DROP TABLE IF EXISTS slick_plus_changes;

        DROP INDEX IF EXISTS idx_slick_to_source_slick_active;
        DROP INDEX IF EXISTS slick_to_source_one_active_per_pair_idx;
    """)
