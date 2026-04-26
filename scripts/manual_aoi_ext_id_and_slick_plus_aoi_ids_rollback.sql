DROP VIEW IF EXISTS public.aoi_type_public;

DROP VIEW IF EXISTS public.slick_plus_2;

CREATE OR REPLACE FUNCTION public.slick_before_trigger_func()
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

DROP INDEX IF EXISTS public.idx_aoi_user_geometry;

ALTER TABLE public.aoi_user
    DROP COLUMN IF EXISTS geometry;

ALTER TABLE public.orchestrator_run
    DROP COLUMN IF EXISTS dataset_versions;

ALTER TABLE public.aoi_type
    DROP CONSTRAINT IF EXISTS ck_aoi_type_access_properties;

ALTER TABLE public.aoi_type
    DROP COLUMN IF EXISTS properties,
    DROP COLUMN IF EXISTS access_type,
    DROP COLUMN IF EXISTS read_perm,
    DROP COLUMN IF EXISTS owner,
    DROP COLUMN IF EXISTS filter_toggle;

DROP TABLE IF EXISTS public.aoi_access_type;

ALTER TABLE public.aoi_type
    DROP CONSTRAINT IF EXISTS uq_aoi_type_short_name;

ALTER TABLE public.aoi_type
    ALTER COLUMN short_name DROP NOT NULL;

DROP INDEX IF EXISTS public.idx_aoi_type_ext_id;


ALTER TABLE public.aoi
    ALTER COLUMN geometry SET NOT NULL;

ALTER TABLE public.aoi
    DROP COLUMN IF EXISTS ext_id;
