# type: ignore
""""
Generated with
sqlacodegen $DB_URL --noviews --noindexes --noinflect > cerulean_cloud/database_schema.py

"""
from geoalchemy2.types import Geography
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
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()
metadata = Base.metadata


class Eez(Base):  # noqa
    __tablename__ = "eez"

    id = Column(
        BigInteger,
        primary_key=True,
        server_default=text("nextval('eez_id_seq'::regclass)"),
    )
    mrgid = Column(Integer)
    geoname = Column(Text)
    sovereigns = Column(ARRAY(Text()))
    geometry = Column(
        Geography("MULTIPOLYGON", 4326, from_text="ST_GeogFromText", name="geography"),
        nullable=False,
    )


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
    meta = Column(JSON)
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
    meta = Column(JSON)
    url = Column(Text, nullable=False)
    geometry = Column(
        Geography("POLYGON", 4326, from_text="ST_GeogFromText", name="geography"),
        nullable=False,
    )


class SlickClass(Base):  # noqa
    __tablename__ = "slick_class"

    id = Column(
        Integer,
        primary_key=True,
        server_default=text("nextval('slick_class_id_seq'::regclass)"),
    )
    name = Column(String(200))
    notes = Column(Text)
    slick_class = Column(ARRAY(Integer()))
    create_time = Column(DateTime, nullable=False, server_default=text("now()"))
    active = Column(Boolean, nullable=False)


class SlickSource(Base):  # noqa
    __tablename__ = "slick_source"

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
    meta = Column(JSON)
    geometry = Column(
        Geography("POLYGON", 4326, from_text="ST_GeogFromText", name="geography"),
        nullable=False,
    )


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


class Slick(Base):  # noqa
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
    meta = Column(JSON)
    orchestrator_run = Column(ForeignKey("orchestrator_run.id"), nullable=False)
    slick_class = Column(ForeignKey("slick_class.id"), nullable=False)

    orchestrator_run1 = relationship("OrchestratorRun")
    slick_class1 = relationship("SlickClass")


class SlickToEez(Base):  # noqa
    __tablename__ = "slick_to_eez"

    id = Column(
        BigInteger,
        primary_key=True,
        server_default=text("nextval('slick_to_eez_id_seq'::regclass)"),
    )
    slick = Column(ForeignKey("slick.id"), nullable=False)
    eez = Column(ForeignKey("eez.id"), nullable=False)

    eez1 = relationship("Eez")
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
    human_confidence = Column(Float(53))
    machine_confidence = Column(Float(53))

    slick1 = relationship("Slick")
    slick_source1 = relationship("SlickSource")
