DROP VIEW IF EXISTS public.slick_plus_2;

DROP RULE IF EXISTS bypass_slick_to_aoi_insert ON public.slick_to_aoi;

DROP INDEX IF EXISTS public.idx_aoi_user_geometry;

ALTER TABLE public.aoi_user
    DROP COLUMN IF EXISTS geometry;

ALTER TABLE public.orchestrator_run
    DROP COLUMN IF EXISTS dataset_versions;

ALTER TABLE public.aoi_type
    DROP COLUMN IF EXISTS properties,
    DROP COLUMN IF EXISTS access_type,
    DROP COLUMN IF EXISTS read_perm,
    DROP COLUMN IF EXISTS owner,
    DROP COLUMN IF EXISTS filter_toggle;

DROP TABLE IF EXISTS public.aoi_access_type;

ALTER TABLE public.aoi
    DROP CONSTRAINT IF EXISTS uq_aoi_type_ext_id;

DROP INDEX IF EXISTS public.idx_aoi_type_ext_id;

ALTER TABLE public.aoi
    DROP COLUMN IF EXISTS ext_id;
