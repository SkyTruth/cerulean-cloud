ALTER TABLE public.aoi
    ADD COLUMN ext_id text;

UPDATE public.aoi a
SET ext_id = eez.mrgid::text
FROM public.aoi_eez eez
WHERE eez.aoi_id = a.id;

UPDATE public.aoi a
SET ext_id = iho.mrgid::text
FROM public.aoi_iho iho
WHERE iho.aoi_id = a.id;

UPDATE public.aoi a
SET ext_id = mpa.wdpaid::text
FROM public.aoi_mpa mpa
WHERE mpa.aoi_id = a.id;

ALTER TABLE public.aoi
    ADD CONSTRAINT uq_aoi_type_ext_id UNIQUE (type, ext_id);

CREATE INDEX idx_aoi_type_ext_id
    ON public.aoi (type, ext_id);

ALTER TABLE public.aoi_type
    ADD COLUMN geometry_source_uri text,
    ADD COLUMN pmtiles_uri text,
    ADD COLUMN dataset_version text,
    ADD COLUMN filter_toggle boolean,
    ADD COLUMN owner bigint REFERENCES public.users(id),
    ADD COLUMN read_perm bigint REFERENCES public.permission(id);

ALTER TABLE public.aoi_user
    ADD COLUMN geometry geography;

CREATE INDEX idx_aoi_user_geometry
    ON public.aoi_user
    USING gist (geometry);

CREATE OR REPLACE VIEW public.slick_plus_2 AS
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
    FROM public.slick
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
        AS slick_url,
    aois.aoi_ids
FROM base
JOIN public.orchestrator_run ON orchestrator_run.id = base.orchestrator_run
JOIN public.sentinel1_grd ON sentinel1_grd.id = orchestrator_run.sentinel1_grd
LEFT JOIN LATERAL (
    SELECT hs.cls
    FROM public.hitl_slick hs
    WHERE hs.slick = base.id
    ORDER BY hs.update_time DESC
    LIMIT 1
) AS hs ON TRUE
LEFT JOIN public.cls ON cls.id = hs.cls
LEFT JOIN LATERAL (
    SELECT
        array_agg(aoi.id) FILTER (WHERE aoi.type = 1) AS aoi_type_1_ids,
        array_agg(aoi.id) FILTER (WHERE aoi.type = 2) AS aoi_type_2_ids,
        array_agg(aoi.id) FILTER (WHERE aoi.type = 3) AS aoi_type_3_ids,
        (
            SELECT COALESCE(json_object_agg(aoi_ids.short_name, aoi_ids.ext_ids), '{}'::json)
            FROM (
                SELECT
                    aoi_type.short_name,
                    json_agg(aoi_by_type.ext_id ORDER BY aoi_by_type.ext_id) AS ext_ids
                FROM public.slick_to_aoi sta_by_type
                JOIN public.aoi aoi_by_type ON aoi_by_type.id = sta_by_type.aoi
                JOIN public.aoi_type ON aoi_type.id = aoi_by_type.type
                WHERE sta_by_type.slick = base.id
                  AND aoi_type.short_name IN ('EEZ', 'IHO', 'MPA')
                  AND aoi_by_type.ext_id IS NOT NULL
                GROUP BY aoi_type.short_name
            ) AS aoi_ids
        ) AS aoi_ids
    FROM public.slick_to_aoi sta
    JOIN public.aoi ON aoi.id = sta.aoi
    WHERE sta.slick = base.id
) AS aois ON TRUE
LEFT JOIN LATERAL (
    SELECT
        array_agg(src.ext_id) FILTER (WHERE src.type = 1) AS source_type_1_ids,
        array_agg(src.ext_id) FILTER (WHERE src.type = 2) AS source_type_2_ids,
        array_agg(src.ext_id) FILTER (WHERE src.type = 3) AS source_type_3_ids,
        MAX(sts.collated_score) AS max_source_collated_score
    FROM public.slick_to_source sts
    JOIN public.source src ON src.id = sts.source
    WHERE sts.slick = base.id
      AND sts.active = TRUE
) AS srcs ON TRUE
WHERE hs.cls IS NULL OR hs.cls NOT IN (SELECT id FROM not_oil_clses);
