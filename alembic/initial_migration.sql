--
-- PostgreSQL database dump
--

-- Dumped from database version 11.13
-- Dumped by pg_dump version 13.4

-- Started on 2022-06-24 12:36:20 CEST

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- TOC entry 2 (class 3079 OID 28781)
-- Name: postgis; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION postgis;
CREATE EXTENSION IF NOT EXISTS postgis WITH SCHEMA public;


--
-- TOC entry 5400 (class 0 OID 0)
-- Dependencies: 2
-- Name: EXTENSION postgis; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON EXTENSION postgis IS 'PostGIS geometry, geography, and raster spatial types and functions';


SET default_tablespace = '';

--
-- TOC entry 212 (class 1259 OID 30359)
-- Name: coincident; Type: TABLE; Schema: public; Owner: -
--

CREATE SCHEMA public;

CREATE TABLE public.coincident (
    id integer NOT NULL,
    posi_poly__id integer,
    vessel__id integer,
    geometry public.geography,
    direct_hits integer,
    proximity numeric,
    score numeric,
    method integer,
    destination text,
    speed_avg numeric,
    status integer,
    port_last text,
    port_next text,
    cargo_type integer,
    cargo_amount numeric
);


--
-- TOC entry 213 (class 1259 OID 30365)
-- Name: coincident_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.coincident_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- TOC entry 5401 (class 0 OID 0)
-- Dependencies: 213
-- Name: coincident_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.coincident_id_seq OWNED BY public.coincident.id;


--
-- TOC entry 214 (class 1259 OID 30367)
-- Name: eez; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.eez (
    id integer NOT NULL,
    mrgid integer,
    geometry public.geography,
    geoname text,
    sovereigns text[],
    pol_type text
);


--
-- TOC entry 215 (class 1259 OID 30373)
-- Name: eez_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.eez_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- TOC entry 5402 (class 0 OID 0)
-- Dependencies: 215
-- Name: eez_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.eez_id_seq OWNED BY public.eez.id;


--
-- TOC entry 216 (class 1259 OID 30375)
-- Name: grd; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.grd (
    id integer NOT NULL,
    sns__id integer,
    pid text NOT NULL,
    uuid uuid,
    absoluteorbitnumber integer,
    mode text,
    polarization text,
    s3ingestion timestamp with time zone,
    scihubingestion timestamp with time zone,
    starttime timestamp with time zone,
    stoptime timestamp with time zone,
    geometry public.geography
);


--
-- TOC entry 217 (class 1259 OID 30381)
-- Name: grd_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.grd_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- TOC entry 5403 (class 0 OID 0)
-- Dependencies: 217
-- Name: grd_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.grd_id_seq OWNED BY public.grd.id;


--
-- TOC entry 218 (class 1259 OID 30383)
-- Name: inference; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.inference (
    id integer NOT NULL,
    grd__id integer NOT NULL,
    ocn__id integer,
    thresholds integer[],
    fine_pkl_idx integer,
    chip_size_orig integer,
    chip_size_reduced integer,
    overhang boolean,
    ml_pkls text[]
);


--
-- TOC entry 219 (class 1259 OID 30389)
-- Name: inference_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.inference_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- TOC entry 5404 (class 0 OID 0)
-- Dependencies: 219
-- Name: inference_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.inference_id_seq OWNED BY public.inference.id;


--
-- TOC entry 220 (class 1259 OID 30391)
-- Name: ocn; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.ocn (
    id integer NOT NULL,
    grd__id integer NOT NULL,
    pid text,
    uuid uuid,
    summary text,
    producttype text,
    filename text
);


--
-- TOC entry 221 (class 1259 OID 30397)
-- Name: ocn_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.ocn_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- TOC entry 5405 (class 0 OID 0)
-- Dependencies: 221
-- Name: ocn_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.ocn_id_seq OWNED BY public.ocn.id;


--
-- TOC entry 222 (class 1259 OID 30399)
-- Name: posi_poly; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.posi_poly (
    id integer NOT NULL,
    inference__id integer NOT NULL,
    geometry public.geography,
    slick__id integer,
    class_int integer
);


--
-- TOC entry 233 (class 1259 OID 44615)
-- Name: pg_featureserv; Type: VIEW; Schema: public; Owner: -
--

CREATE VIEW public.pg_featureserv AS
 SELECT a.id,
    (a.geometry)::public.geometry(Polygon,4326) AS geometry,
    c.starttime AS grd__starttime,
    c.pid AS grd__id
   FROM ((public.posi_poly a
     JOIN public.inference b ON ((b.id = a.inference__id)))
     JOIN public.grd c ON ((c.id = b.grd__id)));


--
-- TOC entry 223 (class 1259 OID 30409)
-- Name: posi_poly_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.posi_poly_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- TOC entry 5406 (class 0 OID 0)
-- Dependencies: 223
-- Name: posi_poly_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.posi_poly_id_seq OWNED BY public.posi_poly.id;


--
-- TOC entry 224 (class 1259 OID 30411)
-- Name: slick; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.slick (
    id integer NOT NULL,
    class_int integer
);


--
-- TOC entry 225 (class 1259 OID 30417)
-- Name: slick_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.slick_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- TOC entry 5407 (class 0 OID 0)
-- Dependencies: 225
-- Name: slick_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.slick_id_seq OWNED BY public.slick.id;


--
-- TOC entry 226 (class 1259 OID 30419)
-- Name: sns; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.sns (
    id integer NOT NULL,
    messageid text NOT NULL,
    subject text,
    "timestamp" timestamp with time zone
);


--
-- TOC entry 227 (class 1259 OID 30425)
-- Name: sns_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.sns_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- TOC entry 5408 (class 0 OID 0)
-- Dependencies: 227
-- Name: sns_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.sns_id_seq OWNED BY public.sns.id;


--
-- TOC entry 232 (class 1259 OID 44538)
-- Name: super2; Type: VIEW; Schema: public; Owner: -
--

CREATE VIEW public.super2 AS
 SELECT a.geometry AS posi_poly__poly,
    public.st_astext(d.geometry) AS grd__geometry,
    d.pid AS grd__pid,
    d.starttime AS grd__starttime,
    e.class_int AS slick__class_int,
    e.id AS slick__id,
    ((polycalculated.polyperimeter * polycalculated.polyperimeter) / polycalculated.polyarea) AS posi_poly__polsby_popper,
    (polycalculated.polyarea / public.st_area((public.st_orientedenvelope((a.geometry)::public.geometry))::public.geography)) AS posi_poly__fill_factor,
    ((polycalculated.polyperimeter * polycalculated.polyperimeter) / public.st_area((public.st_orientedenvelope((a.geometry)::public.geometry))::public.geography)) AS posi_poly__linearity,
    b.geoname AS eez__geoname,
    b.sovereigns AS eez__sovereigns,
    public.st_x((polycalculated.polycentroid)::public.geometry) AS posi_poly__longitude,
    public.st_y((polycalculated.polycentroid)::public.geometry) AS posi_poly__latitude,
    polycalculated.polyarea AS posi_poly__area,
    a.id AS posi_poly__id
   FROM ((((public.posi_poly a
     LEFT JOIN public.eez b ON (public.st_intersects(a.geometry, b.geometry)))
     JOIN public.inference c ON ((c.id = a.inference__id)))
     JOIN public.grd d ON ((d.id = c.grd__id)))
     JOIN public.slick e ON ((e.id = a.slick__id))),
    LATERAL ( SELECT public.st_perimeter(a.geometry) AS polyperimeter,
            public.st_area(a.geometry) AS polyarea,
            public.st_centroid(a.geometry) AS polycentroid) polycalculated
  ORDER BY ((polycalculated.polyperimeter * polycalculated.polyperimeter) / polycalculated.polyarea) DESC;


--
-- TOC entry 230 (class 1259 OID 36207)
-- Name: super_sans_eez; Type: VIEW; Schema: public; Owner: -
--

CREATE VIEW public.super_sans_eez AS
 SELECT a.geometry AS posi_poly__poly,
    e.class_int AS slick__class_int,
    e.id AS slick__id,
    d.pid AS grd__pid,
    ((public.st_perimeter(a.geometry) * public.st_perimeter(a.geometry)) / public.st_area(a.geometry)) AS posi_poly__polsby_popper,
    (public.st_area(a.geometry) / public.st_area((public.st_orientedenvelope((a.geometry)::public.geometry))::public.geography)) AS posi_poly__fill_factor,
    d.starttime AS grd__starttime,
    a.id AS posi_poly__id,
    public.st_x((public.st_centroid(a.geometry))::public.geometry) AS posi_poly__longitude,
    public.st_y((public.st_centroid(a.geometry))::public.geometry) AS posi_poly__latitude,
    public.st_area(a.geometry) AS posi_poly__area
   FROM (((public.posi_poly a
     JOIN public.inference c ON ((c.id = a.inference__id)))
     JOIN public.grd d ON ((d.id = c.grd__id)))
     JOIN public.slick e ON ((e.id = a.slick__id)))
  ORDER BY d.starttime DESC;


--
-- TOC entry 234 (class 1259 OID 44664)
-- Name: super_view_material; Type: MATERIALIZED VIEW; Schema: public; Owner: -
--

CREATE MATERIALIZED VIEW public.super_view_material AS
 SELECT (a.geometry)::public.geometry(Polygon,4326) AS posi_poly__poly,
    public.st_astext(d.geometry) AS grd__geometry,
    d.pid AS grd__pid,
    d.starttime AS grd__starttime,
    e.class_int AS slick__class_int,
    e.id AS slick__id,
    ((polycalculated.polyperimeter * polycalculated.polyperimeter) / polycalculated.polyarea) AS posi_poly__polsby_popper,
    (polycalculated.polyarea / public.st_area((public.st_orientedenvelope((a.geometry)::public.geometry))::public.geography)) AS posi_poly__fill_factor,
    b.geoname AS eez__geoname,
    b.sovereigns AS eez__sovereigns,
    public.st_x((polycalculated.polycentroid)::public.geometry) AS posi_poly__longitude,
    public.st_y((polycalculated.polycentroid)::public.geometry) AS posi_poly__latitude,
    polycalculated.polyarea AS posi_poly__area,
    a.id AS posi_poly__id
   FROM ((((public.posi_poly a
     LEFT JOIN public.eez b ON (public.st_intersects(a.geometry, b.geometry)))
     JOIN public.inference c ON ((c.id = a.inference__id)))
     JOIN public.grd d ON ((d.id = c.grd__id)))
     JOIN public.slick e ON ((e.id = a.slick__id))),
    LATERAL ( SELECT public.st_perimeter(a.geometry) AS polyperimeter,
            public.st_area(a.geometry) AS polyarea,
            public.st_centroid(a.geometry) AS polycentroid) polycalculated
  ORDER BY ((polycalculated.polyperimeter * polycalculated.polyperimeter) / polycalculated.polyarea) DESC
 LIMIT 10
  WITH NO DATA;


--
-- TOC entry 231 (class 1259 OID 36224)
-- Name: superview; Type: VIEW; Schema: public; Owner: -
--

CREATE VIEW public.superview AS
 SELECT a.geometry AS posi_poly__poly,
    public.st_astext(d.geometry) AS grd__geometry,
    d.pid AS grd__pid,
    d.starttime AS grd__starttime,
    e.class_int AS slick__class_int,
    e.id AS slick__id,
    ((polycalculated.polyperimeter * polycalculated.polyperimeter) / polycalculated.polyarea) AS posi_poly__polsby_popper,
    (polycalculated.polyarea / public.st_area((public.st_orientedenvelope((a.geometry)::public.geometry))::public.geography)) AS posi_poly__fill_factor,
    b.geoname AS eez__geoname,
    b.sovereigns AS eez__sovereigns,
    public.st_x((polycalculated.polycentroid)::public.geometry) AS posi_poly__longitude,
    public.st_y((polycalculated.polycentroid)::public.geometry) AS posi_poly__latitude,
    polycalculated.polyarea AS posi_poly__area,
    a.id AS posi_poly__id
   FROM ((((public.posi_poly a
     LEFT JOIN public.eez b ON (public.st_intersects(a.geometry, b.geometry)))
     JOIN public.inference c ON ((c.id = a.inference__id)))
     JOIN public.grd d ON ((d.id = c.grd__id)))
     JOIN public.slick e ON ((e.id = a.slick__id))),
    LATERAL ( SELECT public.st_perimeter(a.geometry) AS polyperimeter,
            public.st_area(a.geometry) AS polyarea,
            public.st_centroid(a.geometry) AS polycentroid) polycalculated
  ORDER BY ((polycalculated.polyperimeter * polycalculated.polyperimeter) / polycalculated.polyarea) DESC;


--
-- TOC entry 228 (class 1259 OID 30427)
-- Name: vessel; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.vessel (
    id integer NOT NULL,
    mmsi integer NOT NULL,
    name text,
    flag text,
    callsign text,
    imo text,
    shiptype text,
    length text
);


--
-- TOC entry 229 (class 1259 OID 30433)
-- Name: vessel_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.vessel_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- TOC entry 5409 (class 0 OID 0)
-- Dependencies: 229
-- Name: vessel_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.vessel_id_seq OWNED BY public.vessel.id;


--
-- TOC entry 5212 (class 2604 OID 30803)
-- Name: coincident id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.coincident ALTER COLUMN id SET DEFAULT nextval('public.coincident_id_seq'::regclass);


--
-- TOC entry 5213 (class 2604 OID 30804)
-- Name: eez id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.eez ALTER COLUMN id SET DEFAULT nextval('public.eez_id_seq'::regclass);


--
-- TOC entry 5214 (class 2604 OID 30805)
-- Name: grd id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.grd ALTER COLUMN id SET DEFAULT nextval('public.grd_id_seq'::regclass);


--
-- TOC entry 5215 (class 2604 OID 30806)
-- Name: inference id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.inference ALTER COLUMN id SET DEFAULT nextval('public.inference_id_seq'::regclass);


--
-- TOC entry 5216 (class 2604 OID 30807)
-- Name: ocn id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ocn ALTER COLUMN id SET DEFAULT nextval('public.ocn_id_seq'::regclass);


--
-- TOC entry 5217 (class 2604 OID 30808)
-- Name: posi_poly id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.posi_poly ALTER COLUMN id SET DEFAULT nextval('public.posi_poly_id_seq'::regclass);


--
-- TOC entry 5218 (class 2604 OID 30809)
-- Name: slick id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.slick ALTER COLUMN id SET DEFAULT nextval('public.slick_id_seq'::regclass);


--
-- TOC entry 5219 (class 2604 OID 30810)
-- Name: sns id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sns ALTER COLUMN id SET DEFAULT nextval('public.sns_id_seq'::regclass);


--
-- TOC entry 5220 (class 2604 OID 30811)
-- Name: vessel id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.vessel ALTER COLUMN id SET DEFAULT nextval('public.vessel_id_seq'::regclass);


--
-- TOC entry 5224 (class 2606 OID 30738)
-- Name: coincident coincident_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.coincident
    ADD CONSTRAINT coincident_pkey PRIMARY KEY (id);


--
-- TOC entry 5226 (class 2606 OID 30740)
-- Name: eez eez_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.eez
    ADD CONSTRAINT eez_pkey PRIMARY KEY (id);


--
-- TOC entry 5228 (class 2606 OID 30742)
-- Name: grd grd_pid_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.grd
    ADD CONSTRAINT grd_pid_key UNIQUE (pid);


--
-- TOC entry 5230 (class 2606 OID 30744)
-- Name: grd grd_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.grd
    ADD CONSTRAINT grd_pkey PRIMARY KEY (id);


--
-- TOC entry 5233 (class 2606 OID 30746)
-- Name: inference inference_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.inference
    ADD CONSTRAINT inference_pkey PRIMARY KEY (id);


--
-- TOC entry 5236 (class 2606 OID 30748)
-- Name: ocn ocn_pid_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ocn
    ADD CONSTRAINT ocn_pid_key UNIQUE (pid);


--
-- TOC entry 5238 (class 2606 OID 30750)
-- Name: ocn ocn_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ocn
    ADD CONSTRAINT ocn_pkey PRIMARY KEY (id);


--
-- TOC entry 5241 (class 2606 OID 30752)
-- Name: posi_poly posi_poly_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.posi_poly
    ADD CONSTRAINT posi_poly_pkey PRIMARY KEY (id);


--
-- TOC entry 5243 (class 2606 OID 30754)
-- Name: slick slick_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.slick
    ADD CONSTRAINT slick_pkey PRIMARY KEY (id);


--
-- TOC entry 5246 (class 2606 OID 30756)
-- Name: sns sns_messageid_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sns
    ADD CONSTRAINT sns_messageid_key UNIQUE (messageid);


--
-- TOC entry 5248 (class 2606 OID 30758)
-- Name: sns sns_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sns
    ADD CONSTRAINT sns_pkey PRIMARY KEY (id);


--
-- TOC entry 5250 (class 2606 OID 30760)
-- Name: vessel vessel_mmsi_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.vessel
    ADD CONSTRAINT vessel_mmsi_key UNIQUE (mmsi);


--
-- TOC entry 5252 (class 2606 OID 30762)
-- Name: vessel vessel_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.vessel
    ADD CONSTRAINT vessel_pkey PRIMARY KEY (id);


--
-- TOC entry 5239 (class 1259 OID 32018)
-- Name: fki_posi_poly_slick-id_fkey; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX "fki_posi_poly_slick-id_fkey" ON public.posi_poly USING btree (slick__id);


--
-- TOC entry 5231 (class 1259 OID 30763)
-- Name: idx-grd-pid; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX "idx-grd-pid" ON public.grd USING btree (pid);


--
-- TOC entry 5234 (class 1259 OID 30764)
-- Name: idx-ocn-pid; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX "idx-ocn-pid" ON public.ocn USING btree (pid);


--
-- TOC entry 5244 (class 1259 OID 30765)
-- Name: idx-sns-messageid; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX "idx-sns-messageid" ON public.sns USING btree (messageid);


--
-- TOC entry 5253 (class 2606 OID 30766)
-- Name: coincident coincident_posi_poly-id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.coincident
    ADD CONSTRAINT "coincident_posi_poly-id_fkey" FOREIGN KEY (posi_poly__id) REFERENCES public.posi_poly(id);


--
-- TOC entry 5254 (class 2606 OID 30771)
-- Name: coincident coincident_vessel-id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.coincident
    ADD CONSTRAINT "coincident_vessel-id_fkey" FOREIGN KEY (vessel__id) REFERENCES public.vessel(id);


--
-- TOC entry 5255 (class 2606 OID 30776)
-- Name: grd grd_sns-id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.grd
    ADD CONSTRAINT "grd_sns-id_fkey" FOREIGN KEY (sns__id) REFERENCES public.sns(id);


--
-- TOC entry 5256 (class 2606 OID 30781)
-- Name: inference inference_grd-id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.inference
    ADD CONSTRAINT "inference_grd-id_fkey" FOREIGN KEY (grd__id) REFERENCES public.grd(id);


--
-- TOC entry 5257 (class 2606 OID 30786)
-- Name: inference inference_ocn-id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.inference
    ADD CONSTRAINT "inference_ocn-id_fkey" FOREIGN KEY (ocn__id) REFERENCES public.ocn(id);


--
-- TOC entry 5258 (class 2606 OID 30791)
-- Name: ocn ocn_grd-id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ocn
    ADD CONSTRAINT "ocn_grd-id_fkey" FOREIGN KEY (grd__id) REFERENCES public.grd(id);


--
-- TOC entry 5259 (class 2606 OID 30796)
-- Name: posi_poly posi_poly_inference-id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.posi_poly
    ADD CONSTRAINT "posi_poly_inference-id_fkey" FOREIGN KEY (inference__id) REFERENCES public.inference(id);


--
-- TOC entry 5260 (class 2606 OID 32013)
-- Name: posi_poly posi_poly_slick-id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.posi_poly
    ADD CONSTRAINT "posi_poly_slick-id_fkey" FOREIGN KEY (slick__id) REFERENCES public.slick(id);


--
-- TOC entry 5399 (class 0 OID 0)
-- Dependencies: 4
-- Name: SCHEMA public; Type: ACL; Schema: -; Owner: -
--


REVOKE ALL ON SCHEMA public FROM PUBLIC;
GRANT ALL ON SCHEMA public TO postgres;
GRANT ALL ON SCHEMA public TO PUBLIC;