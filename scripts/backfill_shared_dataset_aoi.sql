-- Chunked slick_to_aoi backfill for a shared-dataset AOI type.
--
-- This script prepares only durable run metadata plus one reusable stage table.
-- Python orchestration owns chunk planning, chunk staging, retries, and splitting.
-- Postgres owns AOI upserts plus one bounded staged-chunk join at a time.

\set ON_ERROR_STOP on

\if :{?aoi_short_name}
\else
  \echo 'Required variable missing: aoi_short_name'
  \quit 1
\endif
\if :{?aoi_long_name}
\else
  \echo 'Required variable missing: aoi_long_name'
  \quit 1
\endif
\if :{?asset_slug}
\else
  \echo 'Required variable missing: asset_slug'
  \quit 1
\endif
\if :{?ext_id_field}
\else
  \echo 'Required variable missing: ext_id_field'
  \quit 1
\endif
\if :{?stage_table}
\else
  \echo 'Required variable missing: stage_table'
  \quit 1
\endif
\if :{?display_name_field}
\else
  \set display_name_field ''
\endif
\if :{?citation}
\else
  \set citation ''
\endif
\if :{?source_url}
\else
  \set source_url ''
\endif
\if :{?dataset_version}
\else
  \set dataset_version ''
\endif
\if :{?batch_size}
\else
  \set batch_size 5000
\endif

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'aoi_type'
          AND column_name IN ('filter_toggle', 'read_perm', 'access_type', 'properties')
        GROUP BY table_schema, table_name
        HAVING count(*) = 4
    ) THEN
        RAISE EXCEPTION 'Target DB is missing AOI access columns. Apply the AOI access migration/manual script before this backfill.';
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'aoi'
          AND column_name = 'ext_id'
    ) THEN
        RAISE EXCEPTION 'Target DB is missing public.aoi.ext_id. Apply the AOI access migration/manual script before this backfill.';
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'ck_aoi_type_access_properties'
          AND conrelid = 'public.aoi_type'::regclass
    ) THEN
        RAISE EXCEPTION 'Target DB is missing ck_aoi_type_access_properties. Apply the AOI access migration/manual script before this backfill.';
    END IF;
END $$;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_slick_to_aoi_slick
    ON public.slick_to_aoi (slick);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_slick_to_aoi_aoi
    ON public.slick_to_aoi (aoi);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_aoi_type_ext_id
    ON public.aoi (type, ext_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_slick_geom
    ON public.slick
    USING gist ((geometry::geometry));

CREATE SCHEMA IF NOT EXISTS maintenance;

CREATE TABLE IF NOT EXISTS maintenance.shared_dataset_aoi_backfill_run (
    aoi_type_short_name text PRIMARY KEY,
    aoi_type_id bigint NOT NULL REFERENCES public.aoi_type(id),
    stage_table regclass NOT NULL,
    asset_slug text NOT NULL,
    dataset_version text,
    batch_size bigint NOT NULL CHECK (batch_size > 0),
    status text NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    completed_at timestamptz
);

CREATE TABLE IF NOT EXISTS maintenance.shared_dataset_aoi_backfill_chunk (
    id bigserial PRIMARY KEY,
    aoi_type_short_name text NOT NULL
        REFERENCES maintenance.shared_dataset_aoi_backfill_run(aoi_type_short_name)
        ON DELETE CASCADE,
    chunk_index bigint NOT NULL,
    parent_chunk_id bigint REFERENCES maintenance.shared_dataset_aoi_backfill_chunk(id),
    split_depth integer NOT NULL DEFAULT 0 CHECK (split_depth >= 0),
    minx double precision NOT NULL,
    miny double precision NOT NULL,
    maxx double precision NOT NULL,
    maxy double precision NOT NULL,
    status text NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'running', 'completed', 'split', 'failed')),
    retry_count integer NOT NULL DEFAULT 0 CHECK (retry_count >= 0),
    stage_rows_loaded bigint NOT NULL DEFAULT 0,
    candidate_slick_rows bigint NOT NULL DEFAULT 0,
    match_rows bigint NOT NULL DEFAULT 0,
    aois_inserted bigint NOT NULL DEFAULT 0,
    links_inserted bigint NOT NULL DEFAULT 0,
    sub_batches integer NOT NULL DEFAULT 0,
    last_error text,
    started_at timestamptz,
    finished_at timestamptz,
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (aoi_type_short_name, chunk_index)
);

SELECT set_config('maintenance.aoi_short_name', :'aoi_short_name', false);
SELECT set_config('maintenance.aoi_long_name', :'aoi_long_name', false);
SELECT set_config('maintenance.asset_slug', :'asset_slug', false);
SELECT set_config('maintenance.ext_id_field', :'ext_id_field', false);
SELECT set_config('maintenance.display_name_field', :'display_name_field', false);
SELECT set_config('maintenance.citation', :'citation', false);
SELECT set_config('maintenance.source_url', :'source_url', false);
SELECT set_config('maintenance.dataset_version', :'dataset_version', false);
SELECT set_config('maintenance.stage_table', :'stage_table', false);
SELECT set_config('maintenance.batch_size', :'batch_size', false);

DO $$
DECLARE
    v_aoi_short_name text := current_setting('maintenance.aoi_short_name');
    v_aoi_long_name text := current_setting('maintenance.aoi_long_name');
    v_asset_slug text := current_setting('maintenance.asset_slug');
    v_ext_id_field text := current_setting('maintenance.ext_id_field');
    v_display_name_field text := current_setting('maintenance.display_name_field');
    v_citation text := current_setting('maintenance.citation');
    v_source_url text := current_setting('maintenance.source_url');
    v_dataset_version text := current_setting('maintenance.dataset_version');
    v_stage_table_text text := current_setting('maintenance.stage_table');
    v_batch_size bigint := current_setting('maintenance.batch_size')::bigint;
    v_stage_table regclass;
    v_stage_schema text;
    v_stage_relname text;
    v_stage_ident text;
    v_idx_prefix text;
    v_read_perm_id bigint;
    v_aoi_type_id bigint;
    v_properties jsonb;
BEGIN
    SELECT to_regclass(v_stage_table_text) INTO v_stage_table;
    IF v_stage_table IS NOT NULL THEN
        SELECT n.nspname, c.relname, format('%I.%I', n.nspname, c.relname)
        INTO v_stage_schema, v_stage_relname, v_stage_ident
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE c.oid = v_stage_table;

        IF v_stage_schema = 'public' THEN
            RAISE EXCEPTION 'Staging table must not live in public: %', v_stage_ident;
        END IF;
    ELSE
        v_stage_ident := v_stage_table_text;
    END IF;

    SELECT id INTO v_read_perm_id
    FROM public.permission
    WHERE short_name = 'any'
    ORDER BY id
    LIMIT 1;

    IF v_read_perm_id IS NULL THEN
        RAISE EXCEPTION 'Expected permission short_name=any before AOI backfill';
    END IF;

    v_properties := jsonb_strip_nulls(jsonb_build_object(
        'asset_slug', v_asset_slug,
        'ext_id_field', v_ext_id_field,
        'display_name_field', NULLIF(v_display_name_field, ''),
        'dataset_version', NULLIF(v_dataset_version, '')
    ));

    INSERT INTO public.aoi_type (
        table_name,
        long_name,
        short_name,
        source_url,
        citation,
        update_time,
        filter_toggle,
        read_perm,
        access_type,
        properties
    )
    VALUES (
        NULL,
        v_aoi_long_name,
        v_aoi_short_name,
        NULLIF(v_source_url, ''),
        NULLIF(v_citation, ''),
        now(),
        FALSE,
        v_read_perm_id,
        'SHARED_DATASET',
        v_properties
    )
    ON CONFLICT (short_name) DO UPDATE
    SET
        table_name = NULL,
        long_name = EXCLUDED.long_name,
        source_url = EXCLUDED.source_url,
        citation = EXCLUDED.citation,
        update_time = now(),
        filter_toggle = FALSE,
        read_perm = COALESCE(public.aoi_type.read_perm, EXCLUDED.read_perm),
        access_type = 'SHARED_DATASET',
        properties = EXCLUDED.properties
    RETURNING id INTO v_aoi_type_id;

    EXECUTE format('CREATE SCHEMA IF NOT EXISTS %I', split_part(v_stage_table_text, '.', 1));
    EXECUTE format(
        'CREATE TABLE IF NOT EXISTS %s (
            ext_id text NOT NULL,
            name text NOT NULL,
            geom geometry(MultiPolygon, 4326) NOT NULL
        )',
        v_stage_table_text
    );
    EXECUTE format('TRUNCATE TABLE %s', v_stage_table_text);

    v_idx_prefix := 'idx_' || substr(md5(v_stage_table_text), 1, 16);
    EXECUTE format(
        'CREATE INDEX IF NOT EXISTS %I ON %s USING gist (geom)',
        v_idx_prefix || '_geom',
        v_stage_table_text
    );
    EXECUTE format(
        'CREATE INDEX IF NOT EXISTS %I ON %s (ext_id)',
        v_idx_prefix || '_ext_id',
        v_stage_table_text
    );

    INSERT INTO maintenance.shared_dataset_aoi_backfill_run (
        aoi_type_short_name,
        aoi_type_id,
        stage_table,
        asset_slug,
        dataset_version,
        batch_size,
        status
    )
    VALUES (
        v_aoi_short_name,
        v_aoi_type_id,
        to_regclass(v_stage_table_text),
        v_asset_slug,
        NULLIF(v_dataset_version, ''),
        v_batch_size,
        'pending'
    )
    ON CONFLICT (aoi_type_short_name) DO UPDATE
    SET
        aoi_type_id = EXCLUDED.aoi_type_id,
        stage_table = EXCLUDED.stage_table,
        asset_slug = EXCLUDED.asset_slug,
        dataset_version = EXCLUDED.dataset_version,
        batch_size = EXCLUDED.batch_size,
        status = 'pending',
        completed_at = NULL,
        updated_at = now();

    DELETE FROM maintenance.shared_dataset_aoi_backfill_chunk
    WHERE aoi_type_short_name = v_aoi_short_name;
END $$;

CREATE OR REPLACE FUNCTION maintenance.process_shared_dataset_aoi_backfill_chunk(
    p_aoi_type_short_name text,
    p_chunk_id bigint,
    p_minx double precision,
    p_miny double precision,
    p_maxx double precision,
    p_maxy double precision,
    p_lock_timeout text DEFAULT '1s',
    p_statement_timeout text DEFAULT '10min'
)
RETURNS TABLE(
    chunk_status text,
    stage_rows bigint,
    candidate_slick_rows bigint,
    match_rows bigint,
    aoi_rows_inserted bigint,
    slick_to_aoi_rows_inserted bigint,
    sub_batches integer
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_aoi_type_id bigint;
    v_stage_ident text;
    v_batch_size bigint;
    v_stage_rows bigint;
    v_candidate_rows bigint;
    v_inserted_aoi_rows bigint;
    v_batch_match_rows bigint;
    v_batch_insert_rows bigint;
    v_total_match_rows bigint := 0;
    v_total_insert_rows bigint := 0;
    v_seq_start bigint := 1;
    v_split_candidate_limit bigint := 50000;
    v_sub_batches integer := 0;
BEGIN
    SELECT
        r.aoi_type_id,
        format('%I.%I', n.nspname, c.relname),
        r.batch_size
    INTO
        v_aoi_type_id,
        v_stage_ident,
        v_batch_size
    FROM maintenance.shared_dataset_aoi_backfill_run r
    JOIN pg_class c ON c.oid = r.stage_table
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE r.aoi_type_short_name = p_aoi_type_short_name;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'No prepared backfill run for AOI type %', p_aoi_type_short_name;
    END IF;

    EXECUTE format('SET LOCAL lock_timeout = %L', p_lock_timeout);
    EXECUTE format('SET LOCAL statement_timeout = %L', p_statement_timeout);

    EXECUTE format('SELECT count(*) FROM %s', v_stage_ident) INTO v_stage_rows;
    IF v_stage_rows = 0 THEN
        RETURN QUERY SELECT
            'completed'::text,
            0::bigint,
            0::bigint,
            0::bigint,
            0::bigint,
            0::bigint,
            0::integer;
        RETURN;
    END IF;

    EXECUTE format(
        'CREATE TEMP TABLE tmp_stage_aoi ON COMMIT DROP AS
         SELECT ext_id, MIN(name) AS name
         FROM %s
         GROUP BY ext_id',
        v_stage_ident
    );

    EXECUTE format(
        $sql$
        CREATE TEMP TABLE tmp_stage_chunks ON COMMIT DROP AS
        WITH normalized AS (
            SELECT
                ext_id,
                CASE
                    WHEN ST_NPoints(geom) > 255
                        THEN ST_Subdivide(ST_MakeValid(ST_Buffer(geom, 0)), 255)
                    ELSE ST_MakeValid(ST_Buffer(geom, 0))
                END AS geom
            FROM %s
        )
        SELECT ext_id, (ST_Dump(geom)).geom::geometry(Polygon, 4326) AS geom
        FROM normalized
        WHERE geom IS NOT NULL
        $sql$,
        v_stage_ident
    );

    CREATE INDEX tmp_stage_chunks_geom_idx ON tmp_stage_chunks USING gist (geom);

    CREATE TEMP TABLE tmp_existing_aoi ON COMMIT DROP AS
    SELECT MIN(id) AS aoi_id, ext_id
    FROM public.aoi
    WHERE type = v_aoi_type_id
      AND ext_id IN (SELECT ext_id FROM tmp_stage_aoi)
    GROUP BY ext_id;

    WITH inserted_aoi AS (
        INSERT INTO public.aoi (type, ext_id, name)
        SELECT v_aoi_type_id, s.ext_id, COALESCE(s.name, s.ext_id)
        FROM tmp_stage_aoi s
        LEFT JOIN tmp_existing_aoi e USING (ext_id)
        WHERE e.aoi_id IS NULL
        RETURNING 1
    )
    SELECT count(*) INTO v_inserted_aoi_rows FROM inserted_aoi;

    CREATE TEMP TABLE tmp_aoi_lookup ON COMMIT DROP AS
    SELECT MIN(id) AS aoi_id, ext_id
    FROM public.aoi
    WHERE type = v_aoi_type_id
      AND ext_id IN (SELECT ext_id FROM tmp_stage_aoi)
    GROUP BY ext_id;

    CREATE TEMP TABLE tmp_candidate_slicks ON COMMIT DROP AS
    SELECT
        row_number() OVER (ORDER BY s.id) AS seq,
        s.id AS slick_id,
        s.geometry::geometry AS geom
    FROM public.slick s
    WHERE s.active
      AND s.geometry::geometry && ST_MakeEnvelope(p_minx, p_miny, p_maxx, p_maxy, 4326);

    SELECT count(*) INTO v_candidate_rows FROM tmp_candidate_slicks;

    IF v_candidate_rows > v_split_candidate_limit THEN
        RETURN QUERY SELECT
            'split_required'::text,
            v_stage_rows,
            v_candidate_rows,
            0::bigint,
            COALESCE(v_inserted_aoi_rows, 0)::bigint,
            0::bigint,
            0::integer;
        RETURN;
    END IF;

    WHILE v_seq_start <= v_candidate_rows LOOP
        WITH batch_slicks AS MATERIALIZED (
            SELECT slick_id, geom
            FROM tmp_candidate_slicks
            WHERE seq >= v_seq_start
              AND seq < v_seq_start + v_batch_size
        ),
        unresolved_pairs AS MATERIALIZED (
            SELECT DISTINCT
                s.slick_id,
                l.aoi_id,
                s.geom AS slick_geom,
                st.geom AS aoi_geom
            FROM batch_slicks s
            JOIN tmp_stage_chunks st
              ON s.geom && st.geom
            JOIN tmp_aoi_lookup l
              ON l.ext_id = st.ext_id
            LEFT JOIN public.slick_to_aoi sta
              ON sta.slick = s.slick_id
             AND sta.aoi = l.aoi_id
            WHERE sta.slick IS NULL
        ),
        matches AS MATERIALIZED (
            SELECT DISTINCT slick_id, aoi_id
            FROM unresolved_pairs
            WHERE ST_Intersects(slick_geom, aoi_geom)
        ),
        inserted_slick_to_aoi AS (
            INSERT INTO public.slick_to_aoi (slick, aoi)
            SELECT slick_id, aoi_id
            FROM matches
            ON CONFLICT DO NOTHING
            RETURNING 1
        )
        SELECT
            (SELECT count(*) FROM matches)::bigint,
            (SELECT count(*) FROM inserted_slick_to_aoi)::bigint
        INTO
            v_batch_match_rows,
            v_batch_insert_rows;

        v_total_match_rows := v_total_match_rows + COALESCE(v_batch_match_rows, 0);
        v_total_insert_rows := v_total_insert_rows + COALESCE(v_batch_insert_rows, 0);
        v_seq_start := v_seq_start + v_batch_size;
        v_sub_batches := v_sub_batches + 1;
    END LOOP;

    RETURN QUERY SELECT
        'completed'::text,
        v_stage_rows,
        v_candidate_rows,
        v_total_match_rows,
        COALESCE(v_inserted_aoi_rows, 0)::bigint,
        v_total_insert_rows,
        v_sub_batches;
END;
$$;

CREATE OR REPLACE FUNCTION maintenance.validate_shared_dataset_aoi_backfill(
    p_aoi_type_short_name text
)
RETURNS TABLE(check_name text, value bigint)
LANGUAGE plpgsql
AS $$
DECLARE
    v_aoi_type_id bigint;
    v_stage_ident text;
    v_stage_rows bigint;
BEGIN
    SELECT
        r.aoi_type_id,
        format('%I.%I', n.nspname, c.relname)
    INTO v_aoi_type_id, v_stage_ident
    FROM maintenance.shared_dataset_aoi_backfill_run r
    JOIN pg_class c ON c.oid = r.stage_table
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE r.aoi_type_short_name = p_aoi_type_short_name;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'No prepared backfill run for AOI type %', p_aoi_type_short_name;
    END IF;

    RETURN QUERY
    SELECT
        'duplicate_aoi_ext_ids'::text,
        count(*)::bigint
    FROM (
        SELECT ext_id
        FROM public.aoi
        WHERE type = v_aoi_type_id
        GROUP BY ext_id
        HAVING count(*) > 1
    ) dup;

    RETURN QUERY
    SELECT
        'pending_chunks'::text,
        count(*)::bigint
    FROM maintenance.shared_dataset_aoi_backfill_chunk
    WHERE aoi_type_short_name = p_aoi_type_short_name
      AND status = 'pending';

    RETURN QUERY
    SELECT
        'failed_chunks'::text,
        count(*)::bigint
    FROM maintenance.shared_dataset_aoi_backfill_chunk
    WHERE aoi_type_short_name = p_aoi_type_short_name
      AND status = 'failed';

    EXECUTE format('SELECT count(*) FROM %s', v_stage_ident) INTO v_stage_rows;
    RETURN QUERY
    SELECT 'staged_rows_remaining'::text, COALESCE(v_stage_rows, 0);

    RETURN QUERY
    SELECT
        'links_inserted'::text,
        COALESCE(sum(links_inserted), 0)::bigint
    FROM maintenance.shared_dataset_aoi_backfill_chunk
    WHERE aoi_type_short_name = p_aoi_type_short_name;
END;
$$;

CREATE OR REPLACE FUNCTION maintenance.cleanup_shared_dataset_aoi_backfills(
    p_retention interval DEFAULT interval '7 days'
)
RETURNS integer
LANGUAGE plpgsql
AS $$
DECLARE
    v_row record;
    v_removed integer := 0;
BEGIN
    FOR v_row IN
        SELECT
            r.aoi_type_short_name,
            format('%I.%I', n.nspname, c.relname) AS stage_ident
        FROM maintenance.shared_dataset_aoi_backfill_run r
        JOIN pg_class c ON c.oid = r.stage_table
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE r.status IN ('completed', 'failed')
          AND r.updated_at < now() - p_retention
    LOOP
        EXECUTE format('DROP TABLE IF EXISTS %s', v_row.stage_ident);
        DELETE FROM maintenance.shared_dataset_aoi_backfill_run
        WHERE aoi_type_short_name = v_row.aoi_type_short_name;
        v_removed := v_removed + 1;
    END LOOP;

    RETURN v_removed;
END;
$$;

CREATE OR REPLACE PROCEDURE maintenance.finish_shared_dataset_aoi_backfill(
    p_aoi_type_short_name text
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_run_status text;
    v_aoi_type_id bigint;
    v_duplicate_count bigint;
    v_stage_ident text;
    v_failed_chunks bigint;
    v_pending_chunks bigint;
BEGIN
    SELECT
        r.status,
        r.aoi_type_id,
        format('%I.%I', n.nspname, c.relname)
    INTO v_run_status, v_aoi_type_id, v_stage_ident
    FROM maintenance.shared_dataset_aoi_backfill_run r
    JOIN pg_class c ON c.oid = r.stage_table
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE r.aoi_type_short_name = p_aoi_type_short_name;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'No prepared backfill run for AOI type %', p_aoi_type_short_name;
    END IF;

    SELECT
        count(*) FILTER (WHERE status = 'failed'),
        count(*) FILTER (WHERE status IN ('pending', 'running'))
    INTO v_failed_chunks, v_pending_chunks
    FROM maintenance.shared_dataset_aoi_backfill_chunk
    WHERE aoi_type_short_name = p_aoi_type_short_name;

    IF v_failed_chunks > 0 OR v_pending_chunks > 0 THEN
        RAISE EXCEPTION 'Backfill for AOI type % is not complete: % failed chunks, % pending/running chunks',
            p_aoi_type_short_name,
            v_failed_chunks,
            v_pending_chunks;
    END IF;

    SELECT count(*)
    INTO v_duplicate_count
    FROM (
        SELECT ext_id
        FROM public.aoi
        WHERE type = v_aoi_type_id
        GROUP BY ext_id
        HAVING count(*) > 1
    ) dup;

    IF v_duplicate_count > 0 THEN
        RAISE EXCEPTION 'AOI type % has % duplicate ext_id values in public.aoi', p_aoi_type_short_name, v_duplicate_count;
    END IF;

    UPDATE public.aoi_type
    SET
        filter_toggle = FALSE,
        update_time = now()
    WHERE id = v_aoi_type_id;

    UPDATE maintenance.shared_dataset_aoi_backfill_run
    SET
        status = 'completed',
        completed_at = now(),
        updated_at = now()
    WHERE aoi_type_short_name = p_aoi_type_short_name;

    EXECUTE format('DROP TABLE IF EXISTS %s', v_stage_ident);

    RAISE NOTICE 'AOI type % passed finish checks, stage table % was dropped, and filter_toggle remains FALSE for manual UI enablement',
        p_aoi_type_short_name,
        v_stage_ident;
END;
$$;

SELECT *
FROM maintenance.shared_dataset_aoi_backfill_run
WHERE aoi_type_short_name = :'aoi_short_name';
