""""
1. Copy this comment
2. Run:
    sqlacodegen $DB_URL --noviews --noindexes --noinflect > cerulean_cloud/database_schema.py
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
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.orm.decl_api import DeclarativeMeta

# Base = declarative_base()
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


class InfraDistance(Base):
    """This is a dummy docstring."""

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


class Model(Base):
    """This is a dummy docstring."""

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


class Sentinel1Grd(Base):
    """This is a dummy docstring."""

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


class SlickClass(Base):
    """This is a dummy docstring."""

    __tablename__ = "slick_class"

    id = Column(
        Integer,
        primary_key=True,
        server_default=text("nextval('slick_class_id_seq'::regclass)"),
    )
    value = Column(Integer)
    name = Column(String(200))
    notes = Column(Text)
    slick_class = Column(ARRAY(Integer()))
    create_time = Column(DateTime, nullable=False, server_default=text("now()"))
    active = Column(Boolean, nullable=False)


class SourceClass(Base):
    """This is a dummy docstring."""

    id = Column(
        BigInteger,
        primary_key=True,
        server_default=text("nextval('slick_source_id_seq'::regclass)"),
    )
    name = Column(String(200))
    notes = Column(Text)
    slick_source = Column(ARRAY(BigInteger()))
    create_time = Column(DateTime, nullable=False, server_default=text("now()"))
    active = Column(Boolean, nullable=False)
    geometry = Column(
        Geography(srid=4326, from_text="ST_GeogFromText", name="geography")
    )
    type = Column(String(200))
    parent = Column(ForeignKey("source_class.id"))

    parent1 = relationship("SourceClass", remote_side=[id])


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


class SlickSource(Base):
    """This is a dummy docstring."""

    __tablename__ = "slick_source"

    id = Column(
        BigInteger,
        primary_key=True,
        server_default=text("nextval('slick_source_id_seq'::regclass)"),
    )
    name = Column(String(200))
    source_class = Column(ForeignKey("source_class.id"), nullable=False)
    reference = Column(Text)
    create_time = Column(DateTime, nullable=False, server_default=text("now()"))

    source_class1 = relationship("SourceClass")


class Slick(Base):
    """This is a dummy docstring."""

    __tablename__ = "slick"

    id = Column(
        BigInteger,
        primary_key=True,
        server_default=text("nextval('slick_id_seq'::regclass)"),
    )
    slick_timestamp = Column(DateTime, nullable=False)
    geometry = Column(
        Geography("MULTIPOLYGON", 4326, from_text="ST_GeogFromText", name="geography"),
        nullable=False,
    )
    machine_confidence = Column(Float(53))
    human_confidence = Column(Float(53))
    area = Column(Float(53), Computed("st_area(geometry)", persisted=True))
    perimeter = Column(Float(53), Computed("st_perimeter(geometry)", persisted=True))
    centroid = Column(
        Geography("POINT", 4326, from_text="ST_GeogFromText", name="geography"),
        Computed("st_centroid(geometry)", persisted=True),
    )
    polsby_popper = Column(
        Float(53),
        Computed(
            "((st_perimeter(geometry) * st_perimeter(geometry)) / st_area(geometry))",
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
    validated = Column(Boolean, nullable=False)
    slick = Column(ARRAY(BigInteger()))
    notes = Column(Text)
    meta = Column(JSONB(astext_type=Text()))
    orchestrator_run = Column(ForeignKey("orchestrator_run.id"), nullable=False)
    slick_class = Column(ForeignKey("slick_class.id"), nullable=False)

    orchestrator_run1 = relationship("OrchestratorRun")
    slick_class1 = relationship("SlickClass")


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


class SlickToSlickSource(Base):  # noqa
    __tablename__ = "slick_to_slick_source"

    id = Column(
        BigInteger,
        primary_key=True,
        server_default=text("nextval('slick_to_slick_source_id_seq'::regclass)"),
    )
    slick = Column(ForeignKey("slick.id"), nullable=False)
    slick_source = Column(ForeignKey("slick_source.id"), nullable=False)
    machine_confidence = Column(Float(53))
    rank = Column(BigInteger)
    hitl_coincident = Column(Boolean)
    geojson_fc = Column(JSON, nullable=False)
    geometry = Column(
        Geography("LINESTRING", 4326, from_text="ST_GeogFromText", name="geography"),
        nullable=False,
    )
    create_time = Column(DateTime, nullable=False, server_default=text("now()"))

    slick1 = relationship("Slick")
    slick_source1 = relationship("SlickSource")
