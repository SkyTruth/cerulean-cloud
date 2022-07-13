-- Add many-many table connecting polygons to EEZs
CREATE SEQUENCE IF NOT EXISTS public.map_posi_poly_TO_eez_id_seq
    INCREMENT 1
    START 1
    MINVALUE 1
    MAXVALUE 2147483647
    CACHE 1;

CREATE TABLE IF NOT EXISTS public.map_posi_poly_TO_eez -- TODO add table to sqlalchemy, so it gets populated in real time
(
    id integer NOT NULL DEFAULT nextval('map_posi_poly_TO_eez_id_seq'::regclass),
    posi_poly__id integer,
    eez__id integer,
    CONSTRAINT map_posi_poly_pkey PRIMARY KEY (id),
    CONSTRAINT "eez-id_fkey" FOREIGN KEY (eez__id)
        REFERENCES public.eez (id) MATCH SIMPLE,
    CONSTRAINT "posi_poly-id_fkey" FOREIGN KEY (posi_poly__id)
        REFERENCES public.posi_poly (id) MATCH SIMPLE
);

INSERT INTO map_posi_poly_TO_eez (posi_poly__id, eez__id)
    SELECT a.id, b.id
    FROM posi_poly a
        JOIN eez b ON st_intersects(a.geometry, b.geometry);


-- Add static fields
ALTER TABLE public.posi_poly
    ADD COLUMN IF NOT EXISTS perimeter numeric,
    ADD COLUMN IF NOT EXISTS area numeric,
    ADD COLUMN IF NOT EXISTS centroid geometry,
    ADD COLUMN IF NOT EXISTS polsby_popper numeric,
    ADD COLUMN IF NOT EXISTS fill_factor numeric,
    ADD COLUMN IF NOT EXISTS pp_ff numeric;

-- Add indices to all fields used in Join, and (TODO commonly searched):
CREATE INDEX IF NOT EXISTS "fki_posi_poly__inference"
    ON public.posi_poly USING btree
    (inference__id ASC NULLS LAST);

CREATE INDEX IF NOT EXISTS "fki_inference__grd"
    ON public.inference USING btree
    (grd__id ASC NULLS LAST);

CREATE INDEX IF NOT EXISTS "fki_posi_poly__slick"
    ON public.posi_poly USING btree
    (slick__id ASC NULLS LAST);

CREATE INDEX IF NOT EXISTS "fki_map_posi_poly_TO_eez__eez"
    ON public.map_posi_poly_TO_eez USING btree
    (eez__id ASC NULLS LAST);

CREATE INDEX IF NOT EXISTS "fki_map_posi_poly_TO_eez__posi_poly"
    ON public.map_posi_poly_TO_eez USING btree
    (posi_poly__id ASC NULLS LAST);

CREATE INDEX IF NOT EXISTS "idx_pp_ff"
    ON public.posi_poly USING btree
    (pp_ff ASC NULLS LAST);

CREATE INDEX IF NOT EXISTS "idx_geometry_posi_poly"
    ON public.posi_poly USING GIST
    (geometry);

CREATE INDEX IF NOT EXISTS "idx_geometry_eez"
    ON public.eez USING GIST
    (geometry);

CREATE INDEX IF NOT EXISTS "idx_class_int"
    ON public.posi_poly USING btree
    (class_int ASC NULLS LAST);

-- Pre-calculate all the static fields

UPDATE posi_poly SET
    perimeter = st_perimeter(posi_poly.geometry),
    area = st_area(posi_poly.geometry),
    centroid = st_centroid(posi_poly.geometry)::geometry(Point);
UPDATE posi_poly SET
    polsby_popper = (posi_poly.perimeter * posi_poly.perimeter / posi_poly.area),
    fill_factor = (posi_poly.area / st_area(st_orientedenvelope(posi_poly.geometry::geometry)::geography));
UPDATE posi_poly SET
    pp_ff = (posi_poly.polsby_popper * posi_poly.fill_factor);
-- TODO add these fields to the EC2 code to make sure they populate in real time


