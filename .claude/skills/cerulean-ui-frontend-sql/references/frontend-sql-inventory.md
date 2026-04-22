# Cerulean UI Frontend SQL Inventory

Stored snapshot added on 2026-03-11 from a user-provided frontend SQL dump.

Use this reference when answering `cerulean-ui` or frontend SQL questions from the Cerulean backend workspace.

## Index

- `pages/api/sources/[sourceId].ts`
- `pages/api/sources/profile/[id].ts`
- `utils/db/profile-query-builder.ts`
- `pages/api/sources/profile/aois/[id].ts`
- `pages/api/sources/profile/slicks/[id].ts`
- `pages/api/sources/profile/ranking/[id].ts`
- `pages/api/s1-imagery/[slickId].ts`
- `pages/api/slicks/[slickId].ts`
- `pages/api/slicks/geometry.ts`
- `utils/db/slick-query-builder.ts`
- `pages/api/admin/hitl/verification-requests.ts`
- `pages/api/hitl/verification-request/index.ts`
- `pages/api/hitl/verification-request/[slickId].ts`
- `pages/api/admin/slicks/sources/[sourceId].ts`
- `pages/api/admin/slicks/sources/hitl/[sourceId].ts`
- `pages/api/admin/slicks/hitl/[slickId].ts`
- `pages/api/config/index.ts`
- `pages/api/auth/[...all].ts`
- `libs/dom-i18n/server/dictionary-store.ts`

## `pages/api/sources/[sourceId].ts`

```sql
SELECT
  s.ext_id AS "extId",
  s.type::int AS "sourceTypeId",
  s.id::int AS "sourceId",
  sts.coincidence_score AS "coincidenceScore",
  sts.collated_score AS "collatedScore",
  sts.rank::int,
  jsonb_build_object(
    'verified', sts.hitl_verification,
    'userId', sts.hitl_user,
    'userEmail', u.email,
    'notes', sts.hitl_notes,
    'confidence', sts.hitl_confidence,
    'timestamp', sts.hitl_time
  ) AS hitl,
  slk.slick_timestamp AS "slickTimestamp",
  sts.geojson_fc::jsonb AS geometry
FROM slick_to_source sts
JOIN source s ON sts.source = s.id
JOIN slick slk ON slk.id = sts.slick
LEFT JOIN "users" u ON sts.hitl_user = u.id
WHERE sts.id = :slickToSourceId;

SELECT
  t.id,
  t.long_name AS "longName",
  t.short_name AS "shortName",
  t.public AS "isPublic",
  t.source_profile AS "shouldShowSourceProfile"
FROM source_to_tag stt
JOIN tag t ON t.id = stt.tag
WHERE stt.source_ext_id = :extId
  AND stt.source_type = :sourceTypeId;

SELECT flag
FROM source_vessel
WHERE source_id = :sourceId;

SELECT
  operator,
  sovereign,
  orig_yr AS "year",
  last_known_status AS status,
  ext_name AS name
FROM source_infra
WHERE source_id = :sourceId;

SELECT
  detection_probability AS "detectionProbability",
  scene_id AS "sceneId",
  length_m AS "lengthInMeters"
FROM source_dark
WHERE source_id = :sourceId;
```

## `pages/api/sources/profile/[id].ts`

```sql
SELECT
  sv.source_id::int AS "sourceId",
  sv.ext_name AS "shipName",
  sv.ext_shiptype AS "shipType",
  sv.flag,
  s.type::int AS "sourceType",
  'vessel' AS "sourceTypeName",
  s.ext_id::int AS "mmsi",
  tags_agg.tags
FROM source_vessel sv
JOIN source s ON s.id = sv.source_id
LEFT JOIN LATERAL (
  SELECT jsonb_agg(
    jsonb_build_object(
      'id', tag.id,
      'isValidForSourceProfile', tag.source_profile,
      'longName', tag.long_name,
      'shortName', tag.short_name
    )
  ) AS tags
  FROM source_to_tag stt
  LEFT JOIN tag ON stt.tag = tag.id
  WHERE stt.source_ext_id = s.ext_id
) tags_agg ON true
WHERE s.ext_id = :id;
```

## `utils/db/profile-query-builder.ts`

Shared fragment:

```sql
FROM slick_to_source sts
JOIN source s ON s.id = sts.source
JOIN slick slk ON slk.id = sts.slick
JOIN orchestrator_run or2 ON or2.id = slk.orchestrator_run
JOIN sentinel1_grd sg ON sg.id = or2.sentinel1_grd
LEFT JOIN source_to_tag stt ON stt.source_ext_id = s.ext_id AND stt.source_type = s.type
LEFT JOIN tag t ON t.id = stt.tag
LEFT JOIN hitl_slick hs ON hs.slick = slk.id
WHERE TRUE
  AND slk.machine_confidence >= :defaultMachineConfidence
  AND slk.active
  AND slk.cls <> 1
  AND (hs.cls IS NULL OR hs.cls <> 1)
  AND sts.active
  AND sts.hitl_verification IS NOT FALSE
  AND type = 1
  AND (sts.collated_score > :defaultSourceScoreMin OR sts.hitl_verification)
  AND (stt.tag IS NULL OR t.source_profile)
  /* optional */
  AND s.ext_id = :id::text
  /* optional */
  AND s.type = :type;
```

## `pages/api/sources/profile/aois/[id].ts`

```sql
WITH filtered_slicks AS (
  SELECT DISTINCT ON (sg.scene_id) slk.id
  FROM slick_to_source sts
  JOIN source s ON s.id = sts.source
  JOIN slick slk ON slk.id = sts.slick
  JOIN orchestrator_run or2 ON or2.id = slk.orchestrator_run
  JOIN sentinel1_grd sg ON sg.id = or2.sentinel1_grd
  LEFT JOIN source_to_tag stt ON stt.source_ext_id = s.ext_id AND stt.source_type = s.type
  LEFT JOIN tag t ON t.id = stt.tag
  LEFT JOIN hitl_slick hs ON hs.slick = slk.id
  WHERE TRUE
    AND slk.machine_confidence >= :defaultMachineConfidence
    AND slk.active
    AND slk.cls <> 1
    AND (hs.cls IS NULL OR hs.cls <> 1)
    AND sts.active
    AND sts.hitl_verification IS NOT FALSE
    AND type = 1
    AND (sts.collated_score > :defaultSourceScoreMin OR sts.hitl_verification)
    AND (stt.tag IS NULL OR t.source_profile)
    AND s.ext_id = :id::text
    AND s.type = :type
)
SELECT
  at.short_name AS "shortTypeName",
  aoi.type::int AS "typeId",
  at.long_name AS "typeName",
  aoi.id::int AS "aoiId",
  aoi.name,
  fs.id::int AS "slickId",
  CASE
    WHEN aoi.type = 1 THEN ae.mrgid
    WHEN aoi.type = 3 THEN am.wdpaid
    ELSE NULL
  END AS "providerId"
FROM filtered_slicks fs
LEFT JOIN slick_to_aoi sta ON sta.slick = fs.id
LEFT JOIN aoi ON aoi.id = sta.aoi
LEFT JOIN aoi_eez ae ON ae.aoi_id = aoi.id
LEFT JOIN aoi_mpa am ON am.aoi_id = aoi.id
LEFT JOIN aoi_type at ON at.id = aoi.type
WHERE aoi.type = 1 OR aoi.type = 3;
```

## `pages/api/sources/profile/slicks/[id].ts`

```sql
SELECT
  DISTINCT ON (sg.scene_id)
  slk.id::int AS "slickId",
  slk.slick_timestamp AS timestamp,
  slk.area::int,
  ST_AsGeoJSON(ST_Envelope(slk.geometry::geometry))::jsonb AS bbox
FROM slick_to_source sts
JOIN source s ON s.id = sts.source
JOIN slick slk ON slk.id = sts.slick
JOIN orchestrator_run or2 ON or2.id = slk.orchestrator_run
JOIN sentinel1_grd sg ON sg.id = or2.sentinel1_grd
LEFT JOIN source_to_tag stt ON stt.source_ext_id = s.ext_id AND stt.source_type = s.type
LEFT JOIN tag t ON t.id = stt.tag
LEFT JOIN hitl_slick hs ON hs.slick = slk.id
WHERE TRUE
  AND slk.machine_confidence >= :defaultMachineConfidence
  AND slk.active
  AND slk.cls <> 1
  AND (hs.cls IS NULL OR hs.cls <> 1)
  AND sts.active
  AND sts.hitl_verification IS NOT FALSE
  AND type = 1
  AND (sts.collated_score > :defaultSourceScoreMin OR sts.hitl_verification)
  AND (stt.tag IS NULL OR t.source_profile)
  AND s.ext_id = :id::text
  AND s.type = :type;
```

## `pages/api/sources/profile/ranking/[id].ts`

```sql
SELECT
  s.ext_id,
  s.type::int,
  COUNT(DISTINCT sg.scene_id)::int AS occurrence_count,
  SUM(slk.area)::int AS total_area,
  RANK() OVER (
    ORDER BY COUNT(DISTINCT sg.scene_id) DESC, SUM(slk.area) DESC
  )::int AS rank
FROM slick_to_source sts
JOIN source s ON s.id = sts.source
JOIN slick slk ON slk.id = sts.slick
JOIN orchestrator_run or2 ON or2.id = slk.orchestrator_run
JOIN sentinel1_grd sg ON sg.id = or2.sentinel1_grd
LEFT JOIN source_to_tag stt ON stt.source_ext_id = s.ext_id AND stt.source_type = s.type
LEFT JOIN tag t ON t.id = stt.tag
LEFT JOIN hitl_slick hs ON hs.slick = slk.id
WHERE TRUE
  AND slk.machine_confidence >= :defaultMachineConfidence
  AND slk.active
  AND slk.cls <> 1
  AND (hs.cls IS NULL OR hs.cls <> 1)
  AND sts.active
  AND sts.hitl_verification IS NOT FALSE
  AND type = 1
  AND (sts.collated_score > :defaultSourceScoreMin OR sts.hitl_verification)
  AND (stt.tag IS NULL OR t.source_profile)
GROUP BY s.ext_id, s.type
ORDER BY occurrence_count DESC, total_area DESC;

WITH ranking_query AS (
  SELECT
    s.ext_id,
    s.type::int,
    COUNT(DISTINCT sg.scene_id)::int AS occurrence_count,
    SUM(slk.area)::int AS total_area,
    RANK() OVER (
      ORDER BY COUNT(DISTINCT sg.scene_id) DESC, SUM(slk.area) DESC
    )::int AS rank
  FROM slick_to_source sts
  JOIN source s ON s.id = sts.source
  JOIN slick slk ON slk.id = sts.slick
  JOIN orchestrator_run or2 ON or2.id = slk.orchestrator_run
  JOIN sentinel1_grd sg ON sg.id = or2.sentinel1_grd
  LEFT JOIN source_to_tag stt ON stt.source_ext_id = s.ext_id AND stt.source_type = s.type
  LEFT JOIN tag t ON t.id = stt.tag
  LEFT JOIN hitl_slick hs ON hs.slick = slk.id
  WHERE TRUE
    AND slk.machine_confidence >= :defaultMachineConfidence
    AND slk.active
    AND slk.cls <> 1
    AND (hs.cls IS NULL OR hs.cls <> 1)
    AND sts.active
    AND sts.hitl_verification IS NOT FALSE
    AND type = 1
    AND (sts.collated_score > :defaultSourceScoreMin OR sts.hitl_verification)
    AND (stt.tag IS NULL OR t.source_profile)
  GROUP BY s.ext_id, s.type
  ORDER BY occurrence_count DESC, total_area DESC
),
total_sources AS (
  SELECT COUNT(*) AS total_sources FROM ranking_query
)
SELECT
  ts.total_sources::int AS "sourcesCount",
  rq.occurrence_count AS "slickCount",
  (rq.total_area / 1000000) AS "totalAreaKm2",
  rq.total_area AS "totalArea",
  rq.ext_id AS "extId",
  rq.type,
  rq.rank
FROM total_sources ts, ranking_query rq
WHERE rq.ext_id = :id
  AND rq.type = :type;
```

## `pages/api/s1-imagery/[slickId].ts`

```sql
SELECT REPLACE(sg.url, 'sceneid', 'scene_id') AS url
FROM slick sp
LEFT JOIN orchestrator_run orc ON orc.id = sp.orchestrator_run
JOIN sentinel1_grd sg ON sg.id = orc.sentinel1_grd
WHERE sp.id = $1;
```

## `pages/api/slicks/[slickId].ts`

```sql
SELECT
  ST_AsGeoJSON(g.*)::jsonb
    #- '{properties,centroid}'
    #- '{properties,s1_geometry}'
    #- '{properties,centerlines}' AS slick
FROM (
  SELECT
    sp.*,
    sg.scene_id AS s1_scene_id,
    COALESCE(jsonb_agg(sts.id) FILTER (WHERE sts.id IS NOT NULL), '[]'::jsonb) AS "sourceIds",
    COALESCE(jsonb_agg(sts.id) FILTER (WHERE sts.id IS NOT NULL AND sts.hitl_verification), '[]'::jsonb) AS "verifiedSourceIds",
    REPLACE(sg.url, 'sceneid', 'scene_id') AS s1_tile_url,
    ST_AsGeoJSON(ST_Transform(sp.centroid::geometry, 4326))::jsonb->'coordinates' AS lngLat,
    /* admin only */
    (
      SELECT jsonb_agg(uhs.*)
      FROM (
        SELECT
          hs.*,
          jsonb_build_object('name', u.name, 'email', u.email, 'id', u.id) AS user
        FROM hitl_slick hs
        JOIN users u ON u.id = hs.user
        WHERE sp.id = hs.slick
        ORDER BY hs.update_time DESC
      ) uhs
    ) AS hitl_reviews,
    (
      SELECT jsonb_agg(staa.*)
      FROM (
        SELECT sta.aoi AS id, a.name, a.type
        FROM slick_to_aoi sta
        JOIN aoi a ON a.id = sta.aoi
        WHERE sp.id = sta.slick
      ) staa
    ) AS aois
  FROM slick sp
  LEFT JOIN orchestrator_run orc ON orc.id = sp.orchestrator_run
  JOIN sentinel1_grd sg ON sg.id = orc.sentinel1_grd
  LEFT JOIN LATERAL (
    SELECT sts.id, sts.hitl_verification
    FROM slick_to_source sts
    WHERE sts.active
      AND sts.slick = :slickId
      /* optional */
      AND sts.collated_score >= :minSourceScore::float
      /* optional */
      AND sts.collated_score <= :maxSourceScore::float
    ORDER BY sts.collated_score DESC
    /* optional */
    LIMIT :sourceLimit
  ) sts ON true
  WHERE sp.id = :slickId
  GROUP BY sp.id, sg.scene_id, sg.url
) g;
```

## `pages/api/slicks/geometry.ts`

```sql
SELECT ST_AsGeoJSON(s.*) AS geojson
FROM (
  SELECT s.id, s.geometry
  FROM slick s
  WHERE s.id = ANY(:slickIds)
) s;
```

## `utils/db/slick-query-builder.ts`

```sql
-- geometry column chosen by code: centroid_3857 | geom_3857 | geom_3857_simplified
WITH source_candidates AS (
  SELECT
    sts.slick AS slick_id,
    ARRAY_AGG(DISTINCT s.type) AS source_types,
    MAX(sts.collated_score) AS max_collated_score
  FROM slick_to_source sts
  JOIN source s ON s.id = sts.source
  /* optional */
  LEFT JOIN source_to_tag stt ON stt.source_ext_id = s.ext_id AND s."type" = stt.source_type
  /* optional */
  LEFT JOIN tag t ON stt.tag = t.id
  /* optional */
  LEFT JOIN source_vessel sv ON sv.source_id = sts.source
  WHERE TRUE
    AND sts.active
    /* optional */
    AND sts.rank <= :sourceLimit
    /* optional */
    AND sts.collated_score >= :minCollatedScore
    /* optional */
    AND sts.collated_score <= :maxCollatedScore
    /* optional */
    AND t.short_name = :tag
    /* optional */
    AND sv.flag = ANY(:vesselFlagIds)
    /* optional */
    AND (s.ext_id = ANY(:mmsiIds) AND s.type = 1)
    /* optional */
    AND (s.ext_id = :structureId AND s.type = 2)
    /* optional */
    AND s.id = ANY(:adminSourceIds::int[])
    /* optional */
    AND sts.hitl_time IS NOT NULL
    /* optional */
    AND sts.hitl_time IS NULL
    /* optional */
    AND (sts.hitl_verification AND s.type = ANY(:sourceHITLClasses))
  GROUP BY sts.slick
),
/* optional */ tile_bounds(bounds) AS ((SELECT ST_TileEnvelope(:z, :x, :y))),
/* optional */ user_geometry(geom) AS ((SELECT ST_Transform(ST_MakeValid(ST_GeomFromGeoJSON(:geometryGeoJson)), 3857))),
slick_hitl_review AS (
  SELECT DISTINCT ON (hs.slick) hs.slick, hs.id, hs.cls
  FROM hitl_slick hs
  ORDER BY hs.slick, hs.update_time DESC
),
filtered_slicks AS (
  SELECT
    spp.id,
    spp.<geomCol> AS geometry,
    spp.area,
    spp.slick_timestamp,
    sc.max_collated_score
  FROM
    /* optional */ user_geometry ug,
    /* optional */ tile_bounds tb,
    slick spp
  LEFT JOIN source_candidates sc ON sc.slick_id = spp.id
  LEFT JOIN slick_hitl_review shr ON shr.slick = spp.id
  /* optional */ LEFT JOIN orchestrator_run o ON spp.orchestrator_run = o.id
  /* optional */ LEFT JOIN sentinel1_grd sg ON sg.id = o.sentinel1_grd
  /* optional */ LEFT JOIN slick_to_aoi sta ON sta.slick = spp.id
  /* optional */ LEFT JOIN hitl_request hr ON hr.slick = spp.id
  WHERE TRUE
    AND spp.active
    /* source/no-source branch chosen by code:
       AND sc.slick_id IS NULL
       AND sc.slick_id IS NOT NULL
       AND (sc.slick_id IS NULL OR sc.source_types && :slickSourceTypes::bigint[])
       AND sc.slick_id IS NOT NULL AND sc.source_types && :slickSourceTypes::bigint[]
    */
    /* optional */ AND hr.id IS NOT NULL AND spp.hitl_cls IS NULL
    /* optional */ AND spp.slick_timestamp >= :startDate::timestamptz
    /* optional */ AND spp.slick_timestamp <= :endDate::timestamptz
    /* optional */ AND spp.machine_confidence >= :machineConfidence
    /* optional */ AND spp.area >= :minAreaM2
    /* optional */ AND spp.area <= :maxAreaM2
    /* optional */ AND spp.cls = ANY(:slickClasses)
    /* optional */ AND sg.scene_id = ANY(:sceneIds)
    /* optional */ AND sta.aoi = :aoiId
    /* optional */ AND spp.id = ANY(:slickIds::int[])
    /* optional */ AND (spp.geom_3857 && ug.geom AND ST_Intersects(spp.geom_3857, ug.geom))
    /* optional */ AND ST_Intersects(
      spp.geom_3857,
      ST_Transform(ST_MakeEnvelope(:minLng, :minLat, :maxLng, :maxLat, 4326), 3857)
    )
    /* optional */ AND (shr.cls IS NULL OR shr.cls = ANY(:slickHITLClasses::int[]))
    /* optional */ AND tb.bounds && spp.geom_3857_simplified
    /* optional reviewed-state branch:
       AND shr.id IS NOT NULL
       AND shr.id IS NULL
    */
)
SELECT ST_AsMVT(t.*) AS tiles
FROM (
  SELECT
    fs.id,
    fs.slick_timestamp,
    ST_AsMVTGeom(
      fs.geometry,
      tb.bounds,
      extent => 4096,
      buffer => 256
    ) AS geometry
  FROM filtered_slicks fs, tile_bounds tb
) t;

WITH source_candidates AS (...), slick_hitl_review AS (...), filtered_slicks AS (...)
SELECT
  COUNT(*)::int AS count,
  SUM(area) / :metersToKmFactor AS "totalArea",
  ST_Transform(ST_SetSRID(ST_Extent(geometry), 3857), 4236)::box2d::text AS bb
FROM filtered_slicks;

WITH source_candidates AS (...), slick_hitl_review AS (...), filtered_slicks AS (...)
SELECT
  jsonb_build_object(
    'numberMatched', (SELECT COUNT(*) FROM filtered_slicks),
    'slicks',
    COALESCE(
      jsonb_agg(
        jsonb_build_object(
          'id', fs.id,
          'geometry', ST_AsGeoJSON(ST_Transform(fs.geometry, 4326))::jsonb,
          'timestamp', fs.slick_timestamp,
          'maxCollatedScore', fs.max_collated_score,
          'hitlCls', shr.cls
        )
      ),
      '[]'::jsonb
    )
  ) AS slicks
FROM (
  SELECT *
  FROM filtered_slicks
  ORDER BY <sortField> <ASC|DESC> NULLS LAST, id DESC
  /* optional */ LIMIT :limit
  /* optional */ OFFSET :offset
) fs
LEFT JOIN slick_hitl_review shr ON shr.slick = fs.id;

WITH source_candidates AS (...), slick_hitl_review AS (...), filtered_slicks AS (...)
SELECT
  jsonb_build_object(
    'type', 'FeatureCollection',
    'features', jsonb_agg(ST_AsGeoJSON(clustered.*, id_column => 'cluster')::jsonb)
  ) AS clusters
FROM (
  SELECT
    ST_Transform(ST_Centroid(ST_Collect(geometry))::geometry(Point, 3857), 4236) AS geometry,
    COUNT(*)::int AS count,
    cluster
  FROM (
    SELECT ST_ClusterDBSCAN(geometry, 10000, 2) OVER() AS cluster, geometry
    FROM filtered_slicks
  ) clustered
  GROUP BY cluster
) clustered;
```

## `pages/api/admin/hitl/verification-requests.ts`

```sql
WITH base_requests AS (
  SELECT
    hr.id,
    hr.slick,
    hr.user,
    u."firstName" AS user_first_name,
    u."lastName" AS user_last_name,
    u.email AS user_email,
    hr.date_requested,
    hs.update_time AS date_reviewed,
    hr.date_notified,
    s.slick_timestamp,
    CASE
      WHEN hs.cls = 1 THEN 'false_positive'
      WHEN hs.cls = 9 THEN 'ambiguous'
      WHEN hs.slick IS NOT NULL THEN 'verified'
      ELSE 'requested'
    END AS status
  FROM hitl_request hr
  LEFT JOIN LATERAL (
    SELECT hs.cls, hs.slick, hs.update_time
    FROM hitl_slick hs
    WHERE hs.slick = hr.slick
    ORDER BY hs.update_time DESC
    LIMIT 1
  ) hs ON hs.slick = hr.slick
  LEFT JOIN slick s ON hr.slick = s.id
  LEFT JOIN "users" u ON hr.user = u.id
),
source_agg AS (
  SELECT
    slick,
    COUNT(*) FILTER (WHERE hitl_verification = true) AS verified_count,
    COUNT(*) FILTER (WHERE hitl_verification IS NULL AND hitl_time IS NULL) AS not_reviewed_count,
    COUNT(*) FILTER (WHERE hitl_verification IS NULL AND hitl_time IS NOT NULL) AS ambiguous_count
  FROM slick_to_source
  WHERE slick IN (SELECT slick FROM base_requests)
  GROUP BY slick
)
SELECT
  br.*,
  CASE
    WHEN sa.verified_count > 0 THEN 'verified'
    WHEN sa.not_reviewed_count > 0 THEN 'requested'
    WHEN sa.ambiguous_count > 0 THEN 'ambiguous'
    ELSE 'false_positive'
  END AS source_status
FROM base_requests br
LEFT JOIN source_agg sa ON sa.slick = br.slick
ORDER BY br.date_requested DESC;
```

## `pages/api/hitl/verification-request/index.ts`

```sql
WITH base_requests AS (
  SELECT
    hr.id,
    hr.slick,
    hr.user,
    hr.date_requested,
    hr.escalation,
    hs.update_time AS date_reviewed,
    hr.date_notified,
    s.slick_timestamp,
    CASE
      WHEN hs.cls = 1 THEN 'false_positive'
      WHEN hs.cls = 9 THEN 'ambiguous'
      WHEN hs.slick IS NOT NULL THEN 'verified'
      ELSE 'requested'
    END AS status
  FROM hitl_request hr
  LEFT JOIN LATERAL (
    SELECT hs.cls, hs.slick, hs.update_time
    FROM hitl_slick hs
    WHERE hs.slick = hr.slick
    ORDER BY hs.update_time DESC
    LIMIT 1
  ) hs ON hs.slick = hr.slick
  LEFT JOIN slick s ON hr.slick = s.id
  WHERE hr.user = :userId::int
),
source_agg AS (
  SELECT
    slick,
    COUNT(*) FILTER (WHERE hitl_verification = true) AS verified_count,
    COUNT(*) FILTER (WHERE hitl_verification IS NULL AND hitl_time IS NULL) AS not_reviewed_count,
    COUNT(*) FILTER (WHERE hitl_verification IS NULL AND hitl_time IS NOT NULL) AS ambiguous_count
  FROM slick_to_source
  WHERE slick IN (SELECT slick FROM base_requests)
  GROUP BY slick
)
SELECT
  br.*,
  CASE
    WHEN sa.verified_count > 0 THEN 'verified'
    WHEN sa.not_reviewed_count > 0 THEN 'requested'
    WHEN sa.ambiguous_count > 0 THEN 'ambiguous'
    ELSE 'false_positive'
  END AS source_status
FROM base_requests br
LEFT JOIN source_agg sa ON sa.slick = br.slick
ORDER BY br.date_requested DESC;

SELECT COUNT(*) AS outstanding_count
FROM hitl_request hr
LEFT JOIN LATERAL (
  SELECT hs.cls, hs.slick
  FROM hitl_slick hs
  WHERE hs.slick = hr.slick
  ORDER BY hs.update_time DESC
  LIMIT 1
) hs ON hs.slick = hr.slick
WHERE hr.user = :userId::int
  AND hs.slick IS NULL;
```

## `pages/api/hitl/verification-request/[slickId].ts`

```sql
SELECT COUNT(*) AS count
FROM hitl_request hr
LEFT JOIN LATERAL (
  SELECT hs.cls, hs.slick
  FROM hitl_slick hs
  WHERE hs.slick = hr.slick
  ORDER BY hs.update_time DESC
  LIMIT 1
) hs ON hs.slick = hr.slick
WHERE hr.user = :userId::int
  AND hs.slick IS NULL;

SELECT id
FROM hitl_request
WHERE "user" = :userId::int
  AND slick = :slickId::int;

INSERT INTO hitl_request("user", slick, escalation)
VALUES (:userId::int, :slickId::int, :urgentNote);

SELECT
  CASE
    WHEN hs.cls = 1 THEN 'false_positive'
    WHEN hs.cls = 9 THEN 'ambiguous'
    WHEN hs.slick IS NOT NULL THEN 'verified'
    WHEN hr.slick IS NOT NULL THEN 'requested'
    ELSE 'unverified'
  END AS status
FROM slick s
LEFT JOIN hitl_slick hs ON hs.slick = s.id
LEFT JOIN hitl_request hr ON hr.slick = s.id
WHERE s.id = :slickId::int
ORDER BY hs.update_time DESC
LIMIT 1;
```

## `pages/api/admin/slicks/sources/[sourceId].ts`

```sql
UPDATE slick_to_source
SET hitl_verification =
  CASE
    WHEN source = :sourceId THEN :isSource
    ELSE false
  END,
  hitl_user = :userId,
  hitl_confidence = :confidence,
  hitl_time = :timestampIso
WHERE slick = :slickId;
```

## `pages/api/admin/slicks/sources/hitl/[sourceId].ts`

```sql
UPDATE slick_to_source
SET hitl_verification =
  CASE
    WHEN source = :sourceId THEN :isSource
    WHEN (:isSource::bool IS true OR :isSource::bool IS NULL)
      AND source != :sourceId
      AND hitl_verification IS NULL
      AND hitl_time IS NULL THEN false
    ELSE hitl_verification
  END,
  hitl_user = :userId,
  hitl_confidence =
    CASE
      WHEN :isSource::bool IS NOT false THEN :confidence
      WHEN :isSource::bool IS false AND source = :sourceId THEN :confidence
      ELSE hitl_confidence
    END,
  hitl_notes =
    CASE
      WHEN source = :sourceId THEN :notes::text
      ELSE hitl_notes
    END,
  hitl_time =
    CASE
      WHEN :isSource::bool IS false AND source != :sourceId THEN hitl_time
      ELSE NOW()
    END
WHERE slick = :slickId;
```

## `pages/api/admin/slicks/hitl/[slickId].ts`

```sql
INSERT INTO hitl_slick(id, slick, "user", cls, confidence, update_time, is_duplicate)
VALUES (
  nextval('hitl_slick_id_seq'::regclass),
  :slickId,
  :userId,
  :cls,
  :confidence,
  :timestampIso,
  :isDuplicate
);

UPDATE hitl_slick
SET slick = :slickId,
    "user" = :userId,
    cls = :cls,
    confidence = :confidence,
    update_time = :timestampIso,
    is_duplicate = :isDuplicate
WHERE "user" = :userId
  AND slick = :slickId
  AND update_time = (
    SELECT MAX(update_time)
    FROM hitl_slick
    WHERE "user" = :userId
      AND slick = :slickId
  );

SELECT
  ST_AsGeoJSON(slick.*)::jsonb
    #- '{properties,centroid}'
    #- '{properties,s1_geometry}' AS slick
FROM (
  SELECT
    sp.*,
    ST_Transform(sp.centroid::geometry, 4326)::jsonb->'coordinates' AS lngLat,
    (
      SELECT jsonb_agg(uhs.*)
      FROM (
        SELECT
          hs.*,
          jsonb_build_object('name', u.name, 'email', u.email, 'id', u.id) AS user
        FROM hitl_slick hs
        JOIN users u ON u.id = hs.user
        WHERE sp.id = hs.slick
        ORDER BY hs.update_time DESC
      ) uhs
    ) AS hitl_reviews
  FROM slick_plus sp
  WHERE sp.id = :slickId
) slick;
```

## `pages/api/config/index.ts`

```sql
SELECT a.id::int, a.name, a.type::int AS type_id, at.short_name AS type FROM aoi a JOIN aoi_type at ON at.id = a.type;
SELECT a.id::int, a.long_name, a.short_name, a.source_url, a.citation, a.update_time FROM aoi_type a;
SELECT s.id::int, s.long_name, s.short_name, s.citation, s.table_name FROM source_type s;
SELECT c.id::int, c.long_name, c.short_name, c.supercls FROM cls c;
SELECT MIN(s.slick_timestamp) AS min FROM slick s;
SELECT * FROM get_slick_subclses(:slickAnthroClass);
SELECT id::int, short_name, long_name, description, citation, source_profile FROM tag t WHERE t.public OR t.source_profile;
```

## `pages/api/auth/[...all].ts`

```sql
DELETE FROM hitl_request WHERE "user" = :userId;
DELETE FROM hitl_slick WHERE "user" = :userId;
DELETE FROM slick_to_source WHERE hitl_user = :userId;
```

## `libs/dom-i18n/server/dictionary-store.ts`

```sql
CREATE TABLE IF NOT EXISTS dom_i18n_dictionary (
  partition_key text NOT NULL PRIMARY KEY,
  key text NOT NULL,
  app_id text NOT NULL,
  locale text NOT NULL,
  scope text NOT NULL,
  tenant_id text NULL,
  source_text_canonical text NOT NULL,
  source_text_hash text NOT NULL,
  context_hash text NULL,
  context_mode text NOT NULL,
  translated_text text NOT NULL,
  quality text NOT NULL,
  model text NOT NULL,
  prompt_version text NOT NULL,
  glossary_version text NOT NULL,
  schema_version text NOT NULL,
  created_at timestamptz NOT NULL,
  updated_at timestamptz NOT NULL,
  hit_count integer NOT NULL DEFAULT 0,
  last_hit_at timestamptz NOT NULL,
  do_not_share boolean NOT NULL DEFAULT false,
  notes text NULL
);

SELECT *
FROM dom_i18n_dictionary
WHERE partition_key = $1
LIMIT 1;

INSERT INTO dom_i18n_dictionary (
  partition_key,
  key,
  app_id,
  locale,
  scope,
  tenant_id,
  source_text_canonical,
  source_text_hash,
  context_hash,
  context_mode,
  translated_text,
  quality,
  model,
  prompt_version,
  glossary_version,
  schema_version,
  created_at,
  updated_at,
  hit_count,
  last_hit_at,
  do_not_share,
  notes
) VALUES (
  $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15,
  $16, $17, $18, $19, $20, $21, $22
)
ON CONFLICT (partition_key)
DO UPDATE SET
  translated_text = excluded.translated_text,
  context_mode = excluded.context_mode,
  quality = excluded.quality,
  model = excluded.model,
  prompt_version = excluded.prompt_version,
  glossary_version = excluded.glossary_version,
  schema_version = excluded.schema_version,
  updated_at = excluded.updated_at,
  last_hit_at = excluded.last_hit_at,
  do_not_share = excluded.do_not_share,
  notes = excluded.notes;

UPDATE dom_i18n_dictionary
SET hit_count = hit_count + 1,
    last_hit_at = $2,
    updated_at = $2
WHERE partition_key = $1;
```
