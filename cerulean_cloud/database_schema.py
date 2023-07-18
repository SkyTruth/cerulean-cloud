""""
1. Copy this comment
2. Run:
    sqlacodegen postgresql://user:password@localhost:5432/db --noviews --noindexes --noinflect > cerulean_cloud/database_schema.py
3. Add to every class:
    #noqa
4. Add:
    from sqlalchemy.orm.decl_api import DeclarativeMeta
5. Replace this definition:
    Base: DeclarativeMeta = declarative_base()
    metadata = Base.metadata
6. Paste this comment
"""
from geoalchemy2.types import Geography

# coding: utf-8
from sqlalchemy import (
    ARRAY,
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    Computed,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.orm.decl_api import DeclarativeMeta

Base: DeclarativeMeta = declarative_base()
metadata = Base.metadata


class AoiType(Base):  # noqa
    __tablename__ = "aoi_type"

    id = Column(
        BigInteger,
        primary_key=True,
        server_default=text("nextval('aoi_type_id_seq'::regclass)"),
    )
    table_name = Column(Text, nullable=False)
    long_name = Column(Text)
    short_name = Column(Text)
    source_url = Column(Text)
    citation = Column(Text)
    update_time = Column(DateTime, server_default=text("now()"))


class Class(Base):  # noqa
    __tablename__ = "class"

    id = Column(
        Integer,
        primary_key=True,
        server_default=text("nextval('class_id_seq'::regclass)"),
    )
    short_name = Column(Text)
    long_name = Column(Text)
    superclass = Column(ForeignKey("class.id"))

    parent = relationship("Class", remote_side=[id])


class Filter(Base):  # noqa
    __tablename__ = "filter"

    id = Column(
        BigInteger,
        primary_key=True,
        server_default=text("nextval('filter_id_seq'::regclass)"),
    )
    json = Column(JSON, nullable=False)
    hash = Column(Text)


class Frequency(Base):  # noqa
    __tablename__ = "frequency"

    id = Column(
        BigInteger,
        primary_key=True,
        server_default=text("nextval('frequency_id_seq'::regclass)"),
    )
    short_name = Column(Text, nullable=False, unique=True)
    long_name = Column(Text)


class InfraDistance(Base):  # noqa
    __tablename__ = "infra_distance"

    id = Column(
        Integer,
        primary_key=True,
        server_default=text("nextval('infra_distance_id_seq'::regclass)"),
    )
    name = Column(String(200), nullable=False)
    source = Column(Text, nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    meta = Column(JSONB(astext_type=Text()))
    geometry = Column(
        Geography("POLYGON", 4326, from_text="ST_GeogFromText", name="geography"),
        nullable=False,
    )
    url = Column(Text, nullable=False)


class Model(Base):  # noqa
    __tablename__ = "model"

    id = Column(
        Integer,
        primary_key=True,
        server_default=text("nextval('model_id_seq'::regclass)"),
    )
    name = Column(String(200), nullable=False)
    thresholds = Column(Integer)
    fine_pkl_idx = Column(Integer)
    chip_size_orig = Column(Integer)
    chip_size_reduced = Column(Integer)
    overhang = Column(Boolean)
    file_path = Column(Text, nullable=False)
    updated_time = Column(DateTime, nullable=False, server_default=text("now()"))


class Sentinel1Grd(Base):  # noqa
    __tablename__ = "sentinel1_grd"

    id = Column(
        BigInteger,
        primary_key=True,
        server_default=text("nextval('sentinel1_grd_id_seq'::regclass)"),
    )
    scene_id = Column(String(200), nullable=False, unique=True)
    absolute_orbit_number = Column(Integer)
    mode = Column(String(200))
    polarization = Column(String(200))
    scihub_ingestion_time = Column(DateTime)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    meta = Column(JSONB(astext_type=Text()))
    url = Column(Text, nullable=False)
    geometry = Column(
        Geography("POLYGON", 4326, from_text="ST_GeogFromText", name="geography"),
        nullable=False,
    )


class SourceType(Base):  # noqa
    __tablename__ = "source_type"

    id = Column(
        BigInteger,
        primary_key=True,
        server_default=text("nextval('source_type_id_seq'::regclass)"),
    )
    table_name = Column(Text)
    long_name = Column(Text)
    short_name = Column(Text)
    citation = Column(Text)


class SpatialRefSys(Base):  # noqa
    __tablename__ = "spatial_ref_sys"
    __table_args__ = (CheckConstraint("(srid > 0) AND (srid <= 998999)"),)

    srid = Column(Integer, primary_key=True)
    auth_name = Column(String(256))
    auth_srid = Column(Integer)
    srtext = Column(String(2048))
    proj4text = Column(String(2048))


class Trigger(Base):  # noqa
    __tablename__ = "trigger"

    id = Column(
        BigInteger,
        primary_key=True,
        server_default=text("nextval('trigger_id_seq'::regclass)"),
    )
    trigger_time = Column(DateTime, nullable=False, server_default=text("now()"))
    scene_count = Column(Integer)
    filtered_scene_count = Column(Integer)
    trigger_logs = Column(Text, nullable=False)
    trigger_type = Column(String(200), nullable=False)


class User(Base):  # noqa
    __tablename__ = "user"

    id = Column(
        BigInteger,
        primary_key=True,
        server_default=text("nextval('user_id_seq'::regclass)"),
    )
    email = Column(Text, nullable=False, unique=True)
    create_time = Column(DateTime, server_default=text("now()"))


class VesselDensity(Base):  # noqa
    __tablename__ = "vessel_density"

    id = Column(
        Integer,
        primary_key=True,
        server_default=text("nextval('vessel_density_id_seq'::regclass)"),
    )
    name = Column(String(200), nullable=False)
    source = Column(Text, nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    meta = Column(JSONB(astext_type=Text()))
    geometry = Column(
        Geography("POLYGON", 4326, from_text="ST_GeogFromText", name="geography"),
        nullable=False,
    )


class Aoi(Base):  # noqa
    __tablename__ = "aoi"

    id = Column(
        BigInteger,
        primary_key=True,
        server_default=text("nextval('aoi_id_seq'::regclass)"),
    )
    type = Column(ForeignKey("aoi_type.id"), nullable=False)
    name = Column(Text, nullable=False)
    geometry = Column(
        Geography("MULTIPOLYGON", 4326, from_text="ST_GeogFromText", name="geography"),
        nullable=False,
    )

    aoi_type = relationship("AoiType")


class AoiEez(Aoi):  # noqa
    __tablename__ = "aoi_eez"

    aoi_id = Column(ForeignKey("aoi.id"), primary_key=True)
    mrgid = Column(Integer)
    sovereigns = Column(ARRAY(Text()))


class AoiIho(Aoi):  # noqa
    __tablename__ = "aoi_iho"

    aoi_id = Column(ForeignKey("aoi.id"), primary_key=True)
    mrgid = Column(Integer)


class AoiMpa(Aoi):  # noqa
    __tablename__ = "aoi_mpa"

    aoi_id = Column(ForeignKey("aoi.id"), primary_key=True)
    wdpaid = Column(Integer)
    desig = Column(Text)
    desig_type = Column(Text)
    status_yr = Column(Integer)
    mang_auth = Column(Text)
    parent_iso = Column(Text)


class AoiUser(Aoi):  # noqa
    __tablename__ = "aoi_user"

    aoi_id = Column(ForeignKey("aoi.id"), primary_key=True)
    user = Column(ForeignKey("user.id"))
    create_time = Column(DateTime, server_default=text("now()"))

    user1 = relationship("User")


class ClassMap(Base):  # noqa
    __tablename__ = "class_map"
    __table_args__ = (UniqueConstraint("model", "inference_idx"),)

    id = Column(
        Integer,
        primary_key=True,
        server_default=text("nextval('class_map_id_seq'::regclass)"),
    )
    model = Column(ForeignKey("model.id"))
    inference_idx = Column(Integer)
    _class = Column("class", ForeignKey("class.id"))

    _class1 = relationship("Class")
    model1 = relationship("Model")


class MagicLink(Base):  # noqa
    __tablename__ = "magic_link"

    id = Column(
        BigInteger,
        primary_key=True,
        server_default=text("nextval('magic_link_id_seq'::regclass)"),
    )
    user = Column(ForeignKey("user.id"), nullable=False)
    token = Column(Text, nullable=False)
    expiration_time = Column(DateTime, nullable=False)
    is_used = Column(Boolean, nullable=False)
    create_time = Column(DateTime, server_default=text("now()"))
    update_time = Column(DateTime, server_default=text("now()"))

    user1 = relationship("User")


class OrchestratorRun(Base):  # noqa
    __tablename__ = "orchestrator_run"

    id = Column(
        BigInteger,
        primary_key=True,
        server_default=text("nextval('orchestrator_run_id_seq'::regclass)"),
    )
    inference_start_time = Column(DateTime, nullable=False)
    inference_end_time = Column(DateTime, nullable=False)
    base_tiles = Column(Integer)
    offset_tiles = Column(Integer)
    git_hash = Column(Text)
    git_tag = Column(String(200))
    zoom = Column(Integer)
    scale = Column(Integer)
    success = Column(Boolean)
    inference_run_logs = Column(Text, nullable=False)
    geometry = Column(
        Geography("POLYGON", 4326, from_text="ST_GeogFromText", name="geography"),
        nullable=False,
    )
    trigger = Column(ForeignKey("trigger.id"), nullable=False)
    model = Column(ForeignKey("model.id"), nullable=False)
    sentinel1_grd = Column(ForeignKey("sentinel1_grd.id"))
    vessel_density = Column(ForeignKey("vessel_density.id"))
    infra_distance = Column(ForeignKey("infra_distance.id"))

    infra_distance1 = relationship("InfraDistance")
    model1 = relationship("Model")
    sentinel1_grd1 = relationship("Sentinel1Grd")
    trigger1 = relationship("Trigger")
    vessel_density1 = relationship("VesselDensity")


class Source(Base):  # noqa
    __tablename__ = "source"

    id = Column(
        BigInteger,
        primary_key=True,
        server_default=text("nextval('source_id_seq'::regclass)"),
    )
    type = Column(ForeignKey("source_type.id"), nullable=False)
    st_name = Column(Text, nullable=False)

    source_type = relationship("SourceType")


class SourceInfra(Source):  # noqa
    __tablename__ = "source_infra"

    source_id = Column(ForeignKey("source.id"), primary_key=True)
    geometry = Column(
        Geography("POINT", 4326, from_text="ST_GeogFromText", name="geography"),
        nullable=False,
    )
    ext_id = Column(Text)
    ext_name = Column(Text)
    operator = Column(Text)
    sovereign = Column(Text)
    orig_yr = Column(DateTime)
    last_known_status = Column(Text)


class SourceVessel(Source):  # noqa
    __tablename__ = "source_vessel"

    source_id = Column(ForeignKey("source.id"), primary_key=True)
    ext_name = Column(Text)
    ext_shiptype = Column(Text)
    flag = Column(Text)


class Subscription(Base):  # noqa
    __tablename__ = "subscription"

    id = Column(
        BigInteger,
        primary_key=True,
        server_default=text("nextval('subscription_id_seq'::regclass)"),
    )
    user = Column(ForeignKey("user.id"), nullable=False)
    filter = Column(ForeignKey("filter.id"), nullable=False)
    frequency = Column(ForeignKey("frequency.id"), nullable=False)
    active = Column(Boolean)
    create_time = Column(DateTime, server_default=text("now()"))
    update_time = Column(DateTime, server_default=text("now()"))

    filter1 = relationship("Filter")
    frequency1 = relationship("Frequency")
    user1 = relationship("User")


class Slick(Base):  # noqa
    __tablename__ = "slick"

    id = Column(
        BigInteger,
        primary_key=True,
        server_default=text("nextval('slick_id_seq'::regclass)"),
    )
    geometry = Column(
        Geography("MULTIPOLYGON", 4326, from_text="ST_GeogFromText", name="geography"),
        nullable=False,
    )
    inference_idx = Column(Integer, nullable=False)
    slick_timestamp = Column(DateTime, nullable=False)
    hitl_class = Column(BigInteger)
    machine_confidence = Column(Float(53))
    length = Column(
        Float(53),
        Computed(
            "GREATEST(st_distance(st_pointn(st_orientedenvelope((geometry)::geometry), 1), st_pointn(st_orientedenvelope((geometry)::geometry), 2)), st_distance(st_pointn(st_orientedenvelope((geometry)::geometry), 2), st_pointn(st_orientedenvelope((geometry)::geometry), 3)))",
            persisted=True,
        ),
    )
    area = Column(Float(53), Computed("st_area(geometry)", persisted=True))
    perimeter = Column(Float(53), Computed("st_perimeter(geometry)", persisted=True))
    centroid = Column(
        Geography("POINT", 4326, from_text="ST_GeogFromText", name="geography"),
        Computed("st_centroid(geometry)", persisted=True),
    )
    polsby_popper = Column(
        Float(53),
        Computed(
            "((((4)::double precision * pi()) * st_area(geometry)) / (st_perimeter(geometry) ^ (2)::double precision))",
            persisted=True,
        ),
    )
    fill_factor = Column(
        Float(53),
        Computed(
            "(st_area(geometry) / st_area((st_orientedenvelope((geometry)::geometry))::geography))",
            persisted=True,
        ),
    )
    create_time = Column(DateTime, nullable=False, server_default=text("now()"))
    active = Column(Boolean, nullable=False)
    orchestrator_run = Column(ForeignKey("orchestrator_run.id"), nullable=False)

    orchestrator_run1 = relationship("OrchestratorRun")


class SlickToAoi(Base):  # noqa
    __tablename__ = "slick_to_aoi"

    id = Column(
        BigInteger,
        primary_key=True,
        server_default=text("nextval('slick_to_aoi_id_seq'::regclass)"),
    )
    slick = Column(ForeignKey("slick.id"), nullable=False)
    aoi = Column(ForeignKey("aoi.id"), nullable=False)

    aoi1 = relationship("Aoi")
    slick1 = relationship("Slick")


class SlickToSource(Base):  # noqa
    __tablename__ = "slick_to_source"

    id = Column(
        BigInteger,
        primary_key=True,
        server_default=text("nextval('slick_to_source_id_seq'::regclass)"),
    )
    slick = Column(ForeignKey("slick.id"), nullable=False)
    source = Column(ForeignKey("source.id"), nullable=False)
    machine_confidence = Column(Float(53))
    rank = Column(BigInteger)
    hitl_confirmed = Column(Boolean)
    geojson_fc = Column(JSON, nullable=False)
    geometry = Column(
        Geography("LINESTRING", 4326, from_text="ST_GeogFromText", name="geography"),
        nullable=False,
    )
    create_time = Column(DateTime, nullable=False, server_default=text("now()"))

    slick1 = relationship("Slick")
    source1 = relationship("Source")
