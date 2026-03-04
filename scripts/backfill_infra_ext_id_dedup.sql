-- One-time backfill for infrastructure source deduplication.
-- Scope:
--   - Input mappings are ext_id tuples (large_ext_id -> small_ext_id), source.type = 2 only.
--   - If large exists but small is missing, seed small source/source_infra from the large donor row.
--   - Merge slick_to_source conflicts by preferring any row with hitl_verification = TRUE.
--   - Keep orphaned source rows (do not delete from source/source_infra).
--   - Do not modify source_to_tag / exc behavior.
--   - Mappings whose large_ext_id are not present in source(type=2) are skipped and reported.
--
-- Recommended usage:
--   1) BEGIN;
--   2) Load infra_dedup_map (either INSERT values or \copy from CSV).
--   3) Run the migration sections.
--   4) Inspect validation queries.
--   5) COMMIT;  -- or ROLLBACK for dry run


-- -----------------------------------------------------------------------------
-- 0) Load mapping: large ext_id -> canonical small ext_id
-- -----------------------------------------------------------------------------
CREATE TEMP TABLE infra_dedup_map (
    large_ext_id text PRIMARY KEY,
    small_ext_id text NOT NULL,
    CHECK (large_ext_id <> small_ext_id)
);

-- -- Option A: paste rows directly
-- -- These values come from infra_scratchpad gdf from:
-- -- print(",\n".join(f"('{large}','{small}')" for small, ids in gdf[["structure_id","structure_list"]].itertuples(index=False) for large in ids if large != small))
-- INSERT INTO infra_dedup_map (large_ext_id, small_ext_id) VALUES
-- ('1054391','1051815'),
-- ('1061583','1045408'),
-- ...
-- ('1040027','98923'),
-- ('837616','111087');

-- Safety check: mapping must be loaded.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM infra_dedup_map) THEN
        RAISE EXCEPTION 'infra_dedup_map is empty. Load mapping rows before running migration.';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM infra_dedup_map a
        JOIN infra_dedup_map b
          ON a.small_ext_id = b.large_ext_id
    ) THEN
        RAISE EXCEPTION 'Mapping contains chained remaps (small_ext_id also used as large_ext_id). Flatten mappings to final canonical small ext_id first.';
    END IF;
END $$;

-- -----------------------------------------------------------------------------
-- 1) Resolve source IDs for type=2 and validate assumptions
-- -----------------------------------------------------------------------------
CREATE TEMP TABLE infra_dedup_ids AS
SELECT
    m.large_ext_id,
    m.small_ext_id,
    s_large.id AS large_source_id,
    s_small.id AS small_source_id
FROM infra_dedup_map m
LEFT JOIN source s_large
    ON s_large.type = 2
   AND s_large.ext_id = m.large_ext_id
LEFT JOIN source s_small
    ON s_small.type = 2
   AND s_small.ext_id = m.small_ext_id;

-- Case (2): large exists but small is missing.
-- Seed the missing canonical small source/type=2 row (and source_infra) from a
-- deterministic donor large source so retargeting can proceed.
CREATE TEMP TABLE infra_dedup_need_small AS
SELECT
    i.small_ext_id,
    MIN(i.large_source_id) AS donor_large_source_id
FROM infra_dedup_ids i
WHERE i.large_source_id IS NOT NULL
  AND i.small_source_id IS NULL
GROUP BY i.small_ext_id;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM infra_dedup_need_small n
        LEFT JOIN source_infra si
          ON si.source_id = n.donor_large_source_id
        WHERE si.source_id IS NULL
    ) THEN
        RAISE EXCEPTION 'Cannot seed missing small source rows: at least one donor large source has no source_infra row.';
    END IF;
END $$;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'source'
          AND column_name = 'st_name'
    ) THEN
        INSERT INTO source (type, st_name, ext_id)
        SELECT 2, n.small_ext_id, n.small_ext_id
        FROM infra_dedup_need_small n
        ON CONFLICT (ext_id, type) DO NOTHING;
    ELSE
        INSERT INTO source (type, ext_id)
        SELECT 2, n.small_ext_id
        FROM infra_dedup_need_small n
        ON CONFLICT (ext_id, type) DO NOTHING;
    END IF;
END $$;

INSERT INTO source_infra (
    source_id,
    geometry,
    ext_name,
    operator,
    sovereign,
    orig_yr,
    last_known_status,
    first_detection,
    last_detection,
    mmsi
)
SELECT
    s_small.id,
    si.geometry,
    si.ext_name,
    si.operator,
    si.sovereign,
    si.orig_yr,
    si.last_known_status,
    si.first_detection,
    si.last_detection,
    si.mmsi
FROM infra_dedup_need_small n
JOIN source s_small
  ON s_small.type = 2
 AND s_small.ext_id = n.small_ext_id
JOIN source_infra si
  ON si.source_id = n.donor_large_source_id
ON CONFLICT (source_id) DO NOTHING;

UPDATE infra_dedup_ids i
SET small_source_id = s_small.id
FROM source s_small
WHERE i.small_source_id IS NULL
  AND s_small.type = 2
  AND s_small.ext_id = i.small_ext_id;

CREATE TEMP TABLE infra_dedup_skipped_large AS
SELECT *
FROM infra_dedup_ids
WHERE large_source_id IS NULL;

CREATE TEMP TABLE infra_dedup_valid AS
SELECT *
FROM infra_dedup_ids
WHERE large_source_id IS NOT NULL;

CREATE TEMP TABLE infra_dedup_skipped_small AS
SELECT *
FROM infra_dedup_valid
WHERE small_source_id IS NULL;

CREATE TEMP TABLE infra_dedup_valid_final AS
SELECT *
FROM infra_dedup_valid
WHERE small_source_id IS NOT NULL;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM infra_dedup_valid) THEN
        RAISE EXCEPTION 'No mapping rows resolved to existing large source(type=2). Nothing to do.';
    END IF;

    IF NOT EXISTS (SELECT 1 FROM infra_dedup_valid_final) THEN
        RAISE EXCEPTION 'No mapping rows have both large and small source(type=2) resolved. Nothing to do.';
    END IF;
END $$;

-- Optional visibility before mutation
SELECT COUNT(*) AS mapping_pairs FROM infra_dedup_ids;
SELECT COUNT(*) AS skipped_missing_large_pairs FROM infra_dedup_skipped_large;
SELECT COUNT(*) AS valid_mapping_pairs FROM infra_dedup_valid;
SELECT COUNT(*) AS skipped_missing_small_pairs FROM infra_dedup_skipped_small;
SELECT COUNT(*) AS valid_final_mapping_pairs FROM infra_dedup_valid_final;
SELECT COUNT(*) AS candidate_rows_to_move
FROM slick_to_source sts
JOIN infra_dedup_valid_final i ON i.large_source_id = sts.source;

-- -----------------------------------------------------------------------------
-- 2) Retarget large-source associations to small-source, merging conflicts
-- -----------------------------------------------------------------------------
CREATE TEMP TABLE sts_retarget AS
SELECT
    sts.slick,
    i.small_source_id AS source,
    sts.active,
    sts.git_hash,
    sts.git_tag,
    sts.coincidence_score,
    sts.collated_score,
    sts.rank,
    sts.geojson_fc,
    sts.geometry,
    sts.create_time,
    sts.hitl_verification,
    sts.hitl_confidence,
    sts.hitl_user,
    sts.hitl_time,
    sts.hitl_notes
FROM slick_to_source sts
JOIN infra_dedup_valid_final i
  ON i.large_source_id = sts.source;

INSERT INTO slick_to_source (
    slick,
    source,
    active,
    git_hash,
    git_tag,
    coincidence_score,
    collated_score,
    rank,
    geojson_fc,
    geometry,
    create_time,
    hitl_verification,
    hitl_confidence,
    hitl_user,
    hitl_time,
    hitl_notes
)
WITH retarget_grouped AS (
    SELECT
        slick,
        source,
        BOOL_OR(COALESCE(active, FALSE)) AS active_any,
        BOOL_OR(COALESCE(hitl_verification, FALSE)) AS any_hitl_true
    FROM sts_retarget
    GROUP BY slick, source
),
retarget_picked AS (
    SELECT DISTINCT ON (slick, source)
        slick,
        source,
        git_hash,
        git_tag,
        coincidence_score,
        collated_score,
        rank,
        geojson_fc,
        geometry,
        create_time,
        hitl_verification,
        hitl_confidence,
        hitl_user,
        hitl_time,
        hitl_notes
    FROM sts_retarget
    ORDER BY
        slick,
        source,
        COALESCE(active, FALSE) DESC,
        COALESCE(hitl_verification, FALSE) DESC,
        collated_score DESC NULLS LAST,
        create_time DESC NULLS LAST,
        rank ASC NULLS LAST
),
retarget_final AS (
    SELECT
        p.slick,
        p.source,
        g.active_any AS active,
        p.git_hash,
        p.git_tag,
        p.coincidence_score,
        p.collated_score,
        p.rank,
        p.geojson_fc,
        p.geometry,
        p.create_time,
        CASE
            WHEN g.any_hitl_true THEN TRUE
            ELSE p.hitl_verification
        END AS hitl_verification,
        p.hitl_confidence,
        p.hitl_user,
        p.hitl_time,
        p.hitl_notes
    FROM retarget_picked p
    JOIN retarget_grouped g USING (slick, source)
)
SELECT
    slick,
    source,
    active,
    git_hash,
    git_tag,
    coincidence_score,
    collated_score,
    rank,
    geojson_fc,
    geometry,
    create_time,
    hitl_verification,
    hitl_confidence,
    hitl_user,
    hitl_time,
    hitl_notes
FROM retarget_final
ON CONFLICT (slick, source) DO UPDATE
SET
    -- Preserve active if either row is active.
    active = COALESCE(slick_to_source.active, FALSE) OR COALESCE(EXCLUDED.active, FALSE),

    -- HITL merge rule: TRUE wins over all other values.
    hitl_verification = CASE
        WHEN COALESCE(slick_to_source.hitl_verification, FALSE) OR COALESCE(EXCLUDED.hitl_verification, FALSE)
            THEN TRUE
        WHEN slick_to_source.hitl_verification IS FALSE OR EXCLUDED.hitl_verification IS FALSE
            THEN FALSE
        ELSE NULL
    END,

    -- If retargeted row is HITL TRUE, carry forward HITL provenance fields.
    hitl_confidence = CASE
        WHEN COALESCE(EXCLUDED.hitl_verification, FALSE)
            THEN COALESCE(EXCLUDED.hitl_confidence, slick_to_source.hitl_confidence)
        ELSE slick_to_source.hitl_confidence
    END,
    hitl_user = CASE
        WHEN COALESCE(EXCLUDED.hitl_verification, FALSE)
            THEN COALESCE(EXCLUDED.hitl_user, slick_to_source.hitl_user)
        ELSE slick_to_source.hitl_user
    END,
    hitl_time = CASE
        WHEN COALESCE(EXCLUDED.hitl_verification, FALSE)
            THEN COALESCE(EXCLUDED.hitl_time, slick_to_source.hitl_time)
        ELSE slick_to_source.hitl_time
    END,
    hitl_notes = CASE
        WHEN COALESCE(EXCLUDED.hitl_verification, FALSE)
            THEN COALESCE(EXCLUDED.hitl_notes, slick_to_source.hitl_notes)
        ELSE slick_to_source.hitl_notes
    END;

-- -----------------------------------------------------------------------------
-- 3) Remove old large-source links (source rows remain in source/source_infra)
-- -----------------------------------------------------------------------------
DELETE FROM slick_to_source sts
USING infra_dedup_valid_final i
WHERE sts.source = i.large_source_id;

-- -----------------------------------------------------------------------------
-- 4) Re-rank active links per slick
-- -----------------------------------------------------------------------------
CREATE TEMP TABLE affected_slicks AS
SELECT DISTINCT slick
FROM sts_retarget;  -- from your migration step

CREATE INDEX affected_slicks_slick_idx ON affected_slicks (slick);
ANALYZE affected_slicks;

WITH ranked AS (
    SELECT
        sts.id,
        ROW_NUMBER() OVER (
            PARTITION BY sts.slick
            ORDER BY sts.collated_score DESC NULLS LAST, sts.id
        ) AS new_rank
    FROM affected_slicks a
    JOIN slick_to_source sts
      ON sts.slick = a.slick
    WHERE sts.active
),
to_update AS (
    SELECT
        s.id,
        r.new_rank
    FROM ranked r
    JOIN slick_to_source s
      ON s.id = r.id
    WHERE s.rank IS DISTINCT FROM r.new_rank
)
UPDATE slick_to_source s
SET rank = u.new_rank
FROM to_update u
WHERE s.id = u.id;


-- -----------------------------------------------------------------------------
-- 5) Validation queries
-- -----------------------------------------------------------------------------
-- Expect 0 rows: no slick_to_source still pointing at large source IDs.
SELECT COUNT(*) AS remaining_large_links
FROM slick_to_source sts
JOIN infra_dedup_valid_final i ON i.large_source_id = sts.source;

-- Optional audit: list skipped mappings (large_ext_id not present in source(type=2)).
SELECT large_ext_id, small_ext_id
FROM infra_dedup_skipped_large
ORDER BY large_ext_id;

-- Optional audit: list skipped mappings (small_ext_id not present in source(type=2)).
SELECT large_ext_id, small_ext_id
FROM infra_dedup_skipped_small
ORDER BY large_ext_id;

-- Expect 0 rows: no duplicate active rank values per slick.
SELECT slick, rank, COUNT(*) AS ct
FROM slick_to_source
WHERE active
GROUP BY slick, rank
HAVING COUNT(*) > 1
ORDER BY slick, rank;

-- Optional audit: migrated rows that now have HITL TRUE.
SELECT COUNT(*) AS active_hitl_true_rows
FROM slick_to_source sts
WHERE sts.active
  AND sts.hitl_verification IS TRUE;

-- If results look correct, COMMIT.
