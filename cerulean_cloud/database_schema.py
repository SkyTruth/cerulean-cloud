"""
0. Make any changes you want to EVERYWHERE ELSE that has #EditTheDatabase, but NOT here
1. Copy this comment
2. Run:
    Build the database locally using the readme
    sqlacodegen postgresql://user:password@localhost:5432/db --noviews --noindexes --noinflect > cerulean_cloud/database_schema.py
3. Add to every class:
    #noqa
4. Add:
    from sqlalchemy.orm.decl_api import DeclarativeMeta
5. Replace this definition:
    Base: DeclarativeMeta = declarative_base()
    metadata = Base.metadata
6.  Add the following to source_to_tag (There's no way to have sqlacodegen output this relationship without a DB FK):
        from sqlalchemy import and_
        from sqlalchemy.orm import foreign, relationship
        source_ext = relationship(
            "Source",
            primaryjoin=lambda: and_(
                foreign(SourceToTag.source_ext_id) == Source.ext_id,
                foreign(SourceToTag.source_type)   == Source.type,
            ),
            foreign_keys=lambda: [SourceToTag.source_ext_id, SourceToTag.source_type],
        )
7. Aoi.geometry is a deprecated nullable compatibility column. Do not use it in
    live paths. In AoiUser, keep the child-table "geometry" column mapped as
    aoi_user_geometry to avoid colliding with inherited Aoi.geometry.
8. Paste this comment
"""

from geoalchemy2.types import Geography, Geometry

# coding: utf-8
from sqlalchemy import (
    ARRAY,
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    Computed,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    SmallInteger,
    String,
    Table,
    Text,
    UniqueConstraint,
    and_,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import deferred, foreign, relationship
from sqlalchemy.orm.decl_api import DeclarativeMeta

Base: DeclarativeMeta = declarative_base()
metadata = Base.metadata


class AoiAccessType(Base):  # noqa
    __tablename__ = "aoi_access_type"

    id = Column(Integer, primary_key=True)
    short_name = Column(Text, nullable=False, unique=True)
    prop_keys = Column(ARRAY(Text()), nullable=False)


class AoiType(Base):  # noqa
    __tablename__ = "aoi_type"

    id = Column(
        BigInteger,
        primary_key=True,
        server_default=text("nextval('aoi_type_id_seq'::regclass)"),
    )
    table_name = Column(Text, nullable=False)
    long_name = Column(Text)
    short_name = Column(Text, nullable=False, unique=True)
    source_url = Column(Text)
    citation = Column(Text)
    update_time = Column(DateTime, server_default=text("now()"))
    filter_toggle = deferred(Column(Boolean, server_default=text("NULL")))
    owner = deferred(Column(ForeignKey("users.id"), server_default=text("NULL")))
    read_perm = deferred(
        Column(ForeignKey("permission.id"), server_default=text("NULL"))
    )
    access_type = deferred(
        Column(ForeignKey("aoi_access_type.short_name"), server_default=text("NULL"))
    )
    properties = deferred(Column(JSONB, server_default=text("NULL")))

    aoi_access_type = relationship("AoiAccessType", viewonly=True)
    users = relationship("Users", viewonly=True)
    permission = relationship("Permission", viewonly=True)


class Cls(Base):  # noqa
    __tablename__ = "cls"

    id = Column(
        Integer,
        primary_key=True,
        server_default=text("nextval('cls_id_seq'::regclass)"),
    )
    short_name = Column(Text, unique=True)
    long_name = Column(Text)
    supercls = Column(ForeignKey("cls.id"))
    description = Column(Text)

    parent = relationship("Cls", remote_side=[id])


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


class Layer(Base):  # noqa
    __tablename__ = "layer"

    id = Column(
        Integer,
        primary_key=True,
        server_default=text("nextval('layer_id_seq'::regclass)"),
    )
    short_name = Column(Text, nullable=False, unique=True)
    long_name = Column(Text)
    citation = Column(Text)
    source_url = Column(Text)
    notes = Column(Text)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    json = Column(JSON)
    update_time = Column(DateTime, server_default=text("now()"))


class Model(Base):  # noqa
    __tablename__ = "model"

    id = Column(
        Integer,
        primary_key=True,
        server_default=text("nextval('model_id_seq'::regclass)"),
    )
    type = Column(Text, nullable=False)
    file_path = Column(Text, nullable=False)
    layers = Column(ARRAY(Text()), nullable=False)
    cls_map = Column(JSON, nullable=False)
    name = Column(Text)
    tile_width_m = Column(Integer, nullable=False)
    tile_width_px = Column(Integer, nullable=False)
    zoom_level = Column(
        Integer,
        Computed(
            "(round(log((2)::numeric, (40075000.0 / (tile_width_m)::numeric))) - (1)::numeric)",
            persisted=True,
        ),
    )
    scale = Column(
        Integer, Computed("round(((tile_width_px)::numeric / 256.0))", persisted=True)
    )
    epochs = Column(Integer)
    thresholds = Column(JSON, nullable=False)
    backbone_size = Column(Integer)
    pixel_f1 = Column(Float(53))
    instance_f1 = Column(Float(53))
    update_time = Column(DateTime, nullable=False, server_default=text("now()"))


class Permission(Base):  # noqa
    __tablename__ = "permission"

    id = Column(
        BigInteger,
        primary_key=True,
        server_default=text("nextval('permission_id_seq'::regclass)"),
    )
    short_name = Column(Text, nullable=False, unique=True)
    long_name = Column(Text, nullable=False)


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
    ext_id_name = Column(Text)


class SpatialRefSys(Base):  # noqa
    __tablename__ = "spatial_ref_sys"
    __table_args__ = (CheckConstraint("(srid > 0) AND (srid <= 998999)"),)

    srid = Column(Integer, primary_key=True)
    auth_name = Column(String(256))
    auth_srid = Column(Integer)
    srtext = Column(String(2048))
    proj4text = Column(String(2048))


class SupportedLocale(Base):  # noqa
    __tablename__ = "supported_locale"
    __table_args__ = (
        CheckConstraint("(NOT is_default) OR (fallback_code IS NULL)"),
        CheckConstraint("(fallback_code IS NULL) OR (fallback_code <> code)"),
        CheckConstraint("code = lower(code)"),
        CheckConstraint("text_direction = ANY (ARRAY['ltr'::text, 'rtl'::text])"),
    )

    code = Column(Text, primary_key=True)
    english_name = Column(Text, nullable=False)
    native_name = Column(Text, nullable=False)
    text_direction = Column(Text, nullable=False, server_default=text("'ltr'::text"))
    fallback_code = Column(ForeignKey("supported_locale.code"))
    is_default = Column(Boolean, nullable=False, server_default=text("false"))
    is_active = Column(Boolean, nullable=False, server_default=text("true"))
    sort_order = Column(Integer, nullable=False)
    notes = Column(Text)

    parent = relationship("SupportedLocale", remote_side=[code])


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


class Users(Base):  # noqa
    __tablename__ = "users"

    id = Column(
        BigInteger,
        primary_key=True,
        server_default=text("nextval('users_id_seq'::regclass)"),
    )
    firstName = Column(Text)
    lastName = Column(Text)
    name = Column(Text)
    email = Column(Text, nullable=False, unique=True)
    emailVerified = Column(Boolean)
    image = Column(Text)
    role = Column(Text)
    organization = Column(Text)
    organizationType = Column(JSONB)
    location = Column(Text)
    emailConsent = Column(Boolean)
    banned = Column(Boolean)
    banReason = Column(Text)
    banExpires = Column(DateTime)
    createdAt = Column(DateTime, server_default=text("now()"))
    updatedAt = Column(DateTime, server_default=text("now()"))


class Verifications(Base):  # noqa
    __tablename__ = "verifications"

    id = Column(
        BigInteger,
        primary_key=True,
        server_default=text("nextval('verifications_id_seq'::regclass)"),
    )
    identifier = Column(Text, nullable=False)
    value = Column(Text, nullable=False)
    expiresAt = Column(DateTime)
    createdAt = Column(DateTime, server_default=text("now()"))
    updatedAt = Column(DateTime, server_default=text("now()"))


class Accounts(Base):  # noqa
    __tablename__ = "accounts"

    id = Column(
        BigInteger,
        primary_key=True,
        server_default=text("nextval('accounts_id_seq'::regclass)"),
    )
    userId = Column(ForeignKey("users.id"), nullable=False)
    providerId = Column(Text, nullable=False)
    accountId = Column(Text, nullable=False)
    refreshToken = Column(Text)
    accessToken = Column(Text)
    accessTokenExpiresAt = Column(DateTime)
    idToken = Column(Text)
    scope = Column(Text)
    createdAt = Column(DateTime, server_default=text("now()"))
    updatedAt = Column(DateTime, server_default=text("now()"))

    users = relationship("Users")


class Aoi(Base):  # noqa
    __tablename__ = "aoi"

    id = Column(
        BigInteger,
        primary_key=True,
        server_default=text("nextval('aoi_id_seq'::regclass)"),
    )
    type = Column(ForeignKey("aoi_type.id"), nullable=False)
    name = Column(Text, nullable=False)
    ext_id = deferred(Column(Text, server_default=text("NULL")))
    geometry = deferred(
        Column(
            Geography(
                "MULTIPOLYGON",
                4326,
                from_text="ST_GeogFromText",
                name="geography",
            ),
            server_default=text("NULL"),
        )
    )

    aoi_type = relationship("AoiType")
    slick = relationship("Slick", secondary="slick_to_aoi")


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
    user = Column(ForeignKey("users.id"))
    create_time = Column(DateTime, server_default=text("now()"))
    aoi_user_geometry = deferred(
        Column(
            "geometry",
            Geography(srid=4326, from_text="ST_GeogFromText", name="geography"),
            server_default=text("NULL"),
        )
    )

    users = relationship("Users")


class AoiTypeI18n(Base):  # noqa
    __tablename__ = "aoi_type_i18n"
    __table_args__ = (
        CheckConstraint("num_nonnulls(long_name, citation) > 0"),
        CheckConstraint(
            "quality = ANY (ARRAY['human'::text, 'machine'::text, 'machine_reviewed'::text])"
        ),
        CheckConstraint(
            "status = ANY (ARRAY['draft'::text, 'reviewed'::text, 'published'::text])"
        ),
    )

    aoi_type_id = Column(
        ForeignKey("aoi_type.id", ondelete="CASCADE"), primary_key=True, nullable=False
    )
    locale = Column(
        ForeignKey("supported_locale.code"), primary_key=True, nullable=False
    )
    long_name = Column(Text)
    citation = Column(Text)
    status = Column(Text, nullable=False, server_default=text("'published'::text"))
    quality = Column(Text, nullable=False, server_default=text("'human'::text"))
    source_checksum = Column(Text, nullable=False)
    updated_by = Column(ForeignKey("users.id", ondelete="SET NULL"))
    created_at = Column(DateTime(True), nullable=False, server_default=text("now()"))
    updated_at = Column(DateTime(True), nullable=False, server_default=text("now()"))

    aoi_type = relationship("AoiType")
    supported_locale = relationship("SupportedLocale")
    users = relationship("Users")


class ClsI18n(Base):  # noqa
    __tablename__ = "cls_i18n"
    __table_args__ = (
        CheckConstraint("num_nonnulls(long_name, description) > 0"),
        CheckConstraint(
            "quality = ANY (ARRAY['human'::text, 'machine'::text, 'machine_reviewed'::text])"
        ),
        CheckConstraint(
            "status = ANY (ARRAY['draft'::text, 'reviewed'::text, 'published'::text])"
        ),
    )

    cls_id = Column(
        ForeignKey("cls.id", ondelete="CASCADE"), primary_key=True, nullable=False
    )
    locale = Column(
        ForeignKey("supported_locale.code"), primary_key=True, nullable=False
    )
    long_name = Column(Text)
    description = Column(Text)
    status = Column(Text, nullable=False, server_default=text("'published'::text"))
    quality = Column(Text, nullable=False, server_default=text("'human'::text"))
    source_checksum = Column(Text, nullable=False)
    updated_by = Column(ForeignKey("users.id", ondelete="SET NULL"))
    created_at = Column(DateTime(True), nullable=False, server_default=text("now()"))
    updated_at = Column(DateTime(True), nullable=False, server_default=text("now()"))

    cls = relationship("Cls")
    supported_locale = relationship("SupportedLocale")
    users = relationship("Users")


class FrequencyI18n(Base):
    __tablename__ = "frequency_i18n"
    __table_args__ = (
        CheckConstraint("long_name IS NOT NULL"),
        CheckConstraint(
            "quality = ANY (ARRAY['human'::text, 'machine'::text, 'machine_reviewed'::text])"
        ),
        CheckConstraint(
            "status = ANY (ARRAY['draft'::text, 'reviewed'::text, 'published'::text])"
        ),
    )

    frequency_id = Column(
        ForeignKey("frequency.id", ondelete="CASCADE"), primary_key=True, nullable=False
    )
    locale = Column(
        ForeignKey("supported_locale.code"), primary_key=True, nullable=False
    )
    long_name = Column(Text)
    status = Column(Text, nullable=False, server_default=text("'published'::text"))
    quality = Column(Text, nullable=False, server_default=text("'human'::text"))
    source_checksum = Column(Text, nullable=False)
    updated_by = Column(ForeignKey("users.id", ondelete="SET NULL"))
    created_at = Column(DateTime(True), nullable=False, server_default=text("now()"))
    updated_at = Column(DateTime(True), nullable=False, server_default=text("now()"))

    frequency = relationship("Frequency")
    supported_locale = relationship("SupportedLocale")
    users = relationship("Users")


class LayerI18n(Base):  # noqa
    __tablename__ = "layer_i18n"
    __table_args__ = (
        CheckConstraint("num_nonnulls(long_name, notes, citation) > 0"),
        CheckConstraint(
            "quality = ANY (ARRAY['human'::text, 'machine'::text, 'machine_reviewed'::text])"
        ),
        CheckConstraint(
            "status = ANY (ARRAY['draft'::text, 'reviewed'::text, 'published'::text])"
        ),
    )

    layer_id = Column(
        ForeignKey("layer.id", ondelete="CASCADE"), primary_key=True, nullable=False
    )
    locale = Column(
        ForeignKey("supported_locale.code"), primary_key=True, nullable=False
    )
    long_name = Column(Text)
    notes = Column(Text)
    citation = Column(Text)
    status = Column(Text, nullable=False, server_default=text("'published'::text"))
    quality = Column(Text, nullable=False, server_default=text("'human'::text"))
    source_checksum = Column(Text, nullable=False)
    updated_by = Column(ForeignKey("users.id", ondelete="SET NULL"))
    created_at = Column(DateTime(True), nullable=False, server_default=text("now()"))
    updated_at = Column(DateTime(True), nullable=False, server_default=text("now()"))

    layer = relationship("Layer")
    supported_locale = relationship("SupportedLocale")
    users = relationship("Users")


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
    sea_ice_date = Column(Date)
    dataset_versions = Column(JSONB)
    success = Column(Boolean)
    inference_run_logs = Column(Text, nullable=False)
    geometry = Column(
        Geography("POLYGON", 4326, from_text="ST_GeogFromText", name="geography"),
        nullable=False,
    )
    trigger = Column(ForeignKey("trigger.id"), nullable=False)
    model = Column(ForeignKey("model.id"), nullable=False)
    sentinel1_grd = Column(ForeignKey("sentinel1_grd.id"))

    model1 = relationship("Model")
    sentinel1_grd1 = relationship("Sentinel1Grd")
    trigger1 = relationship("Trigger")


class PermissionI18n(Base):  # noqa
    __tablename__ = "permission_i18n"
    __table_args__ = (
        CheckConstraint("long_name IS NOT NULL"),
        CheckConstraint(
            "quality = ANY (ARRAY['human'::text, 'machine'::text, 'machine_reviewed'::text])"
        ),
        CheckConstraint(
            "status = ANY (ARRAY['draft'::text, 'reviewed'::text, 'published'::text])"
        ),
    )

    permission_id = Column(
        ForeignKey("permission.id", ondelete="CASCADE"),
        primary_key=True,
        nullable=False,
    )
    locale = Column(
        ForeignKey("supported_locale.code"), primary_key=True, nullable=False
    )
    long_name = Column(Text)
    status = Column(Text, nullable=False, server_default=text("'published'::text"))
    quality = Column(Text, nullable=False, server_default=text("'human'::text"))
    source_checksum = Column(Text, nullable=False)
    updated_by = Column(ForeignKey("users.id", ondelete="SET NULL"))
    created_at = Column(DateTime(True), nullable=False, server_default=text("now()"))
    updated_at = Column(DateTime(True), nullable=False, server_default=text("now()"))

    supported_locale = relationship("SupportedLocale")
    permission = relationship("Permission")
    users = relationship("Users")


class Sessions(Base):  # noqa
    __tablename__ = "sessions"

    id = Column(
        BigInteger,
        primary_key=True,
        server_default=text("nextval('sessions_id_seq'::regclass)"),
    )
    userId = Column(ForeignKey("users.id"), nullable=False)
    expiresAt = Column(DateTime, nullable=False)
    token = Column(Text, nullable=False)
    createdAt = Column(DateTime, server_default=text("now()"))
    updatedAt = Column(DateTime, server_default=text("now()"))
    impersonatedBy = Column(Text)
    ipAddress = Column(Text)
    userAgent = Column(Text)

    users = relationship("Users")


class Source(Base):  # noqa
    __tablename__ = "source"
    __table_args__ = (UniqueConstraint("ext_id", "type"),)

    id = Column(
        BigInteger,
        primary_key=True,
        server_default=text("nextval('source_id_seq'::regclass)"),
    )
    type = Column(ForeignKey("source_type.id"), nullable=False)
    ext_id = Column(Text, nullable=False)

    source_type = relationship("SourceType")


class SourceDark(Source):  # noqa
    __tablename__ = "source_dark"

    source_id = Column(ForeignKey("source.id"), primary_key=True)
    geometry = Column(
        Geography("POINT", 4326, from_text="ST_GeogFromText", name="geography"),
        nullable=False,
    )
    scene_id = Column(Text)
    length_m = Column(Float(53))
    detection_probability = Column(Float(53))


class SourceInfra(Source):  # noqa
    __tablename__ = "source_infra"

    source_id = Column(ForeignKey("source.id"), primary_key=True)
    geometry = Column(
        Geography("POINT", 4326, from_text="ST_GeogFromText", name="geography"),
        nullable=False,
    )
    ext_name = Column(Text)
    operator = Column(Text)
    sovereign = Column(Text)
    orig_yr = Column(DateTime)
    last_known_status = Column(Text)
    first_detection = Column(DateTime)
    last_detection = Column(DateTime)
    mmsi = Column(Text)


class SourceNatural(Source):  # noqa
    __tablename__ = "source_natural"

    source_id = Column(ForeignKey("source.id"), primary_key=True)
    geometry = Column(
        Geography("POINT", 4326, from_text="ST_GeogFromText", name="geography"),
        nullable=False,
    )


class SourceVessel(Source):  # noqa
    __tablename__ = "source_vessel"

    source_id = Column(ForeignKey("source.id"), primary_key=True)
    ext_name = Column(Text)
    ext_shiptype = Column(Text)
    flag = Column(Text)


class SourceTypeI18n(Base):  # noqa
    __tablename__ = "source_type_i18n"
    __table_args__ = (
        CheckConstraint("num_nonnulls(long_name, citation) > 0"),
        CheckConstraint(
            "quality = ANY (ARRAY['human'::text, 'machine'::text, 'machine_reviewed'::text])"
        ),
        CheckConstraint(
            "status = ANY (ARRAY['draft'::text, 'reviewed'::text, 'published'::text])"
        ),
    )

    source_type_id = Column(
        ForeignKey("source_type.id", ondelete="CASCADE"),
        primary_key=True,
        nullable=False,
    )
    locale = Column(
        ForeignKey("supported_locale.code"), primary_key=True, nullable=False
    )
    long_name = Column(Text)
    citation = Column(Text)
    status = Column(Text, nullable=False, server_default=text("'published'::text"))
    quality = Column(Text, nullable=False, server_default=text("'human'::text"))
    source_checksum = Column(Text, nullable=False)
    updated_by = Column(ForeignKey("users.id", ondelete="SET NULL"))
    created_at = Column(DateTime(True), nullable=False, server_default=text("now()"))
    updated_at = Column(DateTime(True), nullable=False, server_default=text("now()"))

    supported_locale = relationship("SupportedLocale")
    source_type = relationship("SourceType")
    users = relationship("Users")


class Subscription(Base):  # noqa
    __tablename__ = "subscription"

    id = Column(
        BigInteger,
        primary_key=True,
        server_default=text("nextval('subscription_id_seq'::regclass)"),
    )
    user = Column(ForeignKey("users.id"), nullable=False)
    filter = Column(ForeignKey("filter.id"), nullable=False)
    frequency = Column(ForeignKey("frequency.id"), nullable=False)
    active = Column(Boolean)
    create_time = Column(DateTime, server_default=text("now()"))
    update_time = Column(DateTime, server_default=text("now()"))

    filter1 = relationship("Filter")
    frequency1 = relationship("Frequency")
    users = relationship("Users")


class Tag(Base):  # noqa
    __tablename__ = "tag"

    id = Column(
        BigInteger,
        primary_key=True,
        server_default=text("nextval('tag_id_seq'::regclass)"),
    )
    short_name = Column(Text, nullable=False, unique=True)
    long_name = Column(Text, nullable=False)
    description = Column(Text)
    citation = Column(Text)
    owner = Column(ForeignKey("users.id"))
    read_perm = Column(ForeignKey("permission.id"))
    write_perm = Column(ForeignKey("permission.id"))
    public = Column(Boolean, nullable=False)
    source_profile = Column(Boolean, nullable=False)

    users = relationship("Users")
    permission = relationship(
        "Permission", primaryjoin="Tag.read_perm == Permission.id"
    )
    permission1 = relationship(
        "Permission", primaryjoin="Tag.write_perm == Permission.id"
    )


t_aoi_chunks = Table(
    "aoi_chunks",
    metadata,
    Column(
        "id",
        ForeignKey("aoi.id", ondelete="CASCADE", deferrable=True, initially="DEFERRED"),
    ),
    Column(
        "geometry",
        Geometry("POLYGON", 4326, from_text="ST_GeomFromEWKT", name="geometry"),
        nullable=False,
    ),
)


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
    active = Column(Boolean, nullable=False)
    orchestrator_run = Column(ForeignKey("orchestrator_run.id"), nullable=False)
    create_time = Column(DateTime, nullable=False, server_default=text("now()"))
    inference_idx = Column(Integer, nullable=False)
    cls = Column(Integer)
    hitl_cls = Column(ForeignKey("cls.id"))
    machine_confidence = Column(Float(53))
    precursor_slicks = Column(ARRAY(BigInteger()))
    notes = Column(Text)
    centerlines = Column(JSON)
    aspect_ratio_factor = Column(Float(53))
    length = Column(Float(53))
    area = Column(Float(53))
    perimeter = Column(Float(53))
    centroid = Column(
        Geography("POINT", 4326, from_text="ST_GeogFromText", name="geography")
    )
    polsby_popper = Column(Float(53))
    fill_factor = Column(Float(53))
    geom_3857_simplified = Column(
        Geometry(srid=3857, from_text="ST_GeomFromEWKT", name="geometry"),
        Computed(
            "st_simplifypreservetopology(st_transform((geometry)::geometry, 3857), (100)::double precision)",
            persisted=True,
        ),
    )
    centroid_3857 = Column(
        Geometry("POINT", 3857, from_text="ST_GeomFromEWKT", name="geometry"),
        Computed("st_transform((centroid)::geometry, 3857)", persisted=True),
    )
    geom_3857 = Column(
        Geometry(srid=3857, from_text="ST_GeomFromEWKT", name="geometry"),
        Computed("st_transform((geometry)::geometry, 3857)", persisted=True),
    )
    geometry_count = Column(SmallInteger)
    largest_area = Column(Float(53))
    median_area = Column(Float(53))
    geometric_slick_potential = Column(Float(53))

    cls1 = relationship("Cls")
    orchestrator_run1 = relationship("OrchestratorRun")


class SourceToTag(Base):  # noqa
    __tablename__ = "source_to_tag"

    source_ext_id = Column(Text, primary_key=True, nullable=False)
    source_type = Column(BigInteger, primary_key=True, nullable=False)
    tag = Column(ForeignKey("tag.id"), primary_key=True, nullable=False)
    create_time = Column(DateTime, nullable=False, server_default=text("now()"))

    source_ext = relationship(
        "Source",
        primaryjoin=lambda: and_(
            foreign(SourceToTag.source_ext_id) == Source.ext_id,
            foreign(SourceToTag.source_type) == Source.type,
        ),
        foreign_keys=lambda: [SourceToTag.source_ext_id, SourceToTag.source_type],
    )
    tag1 = relationship("Tag")


class TagI18n(Base):  # noqa
    __tablename__ = "tag_i18n"
    __table_args__ = (
        CheckConstraint("num_nonnulls(long_name, description, citation) > 0"),
        CheckConstraint(
            "quality = ANY (ARRAY['human'::text, 'machine'::text, 'machine_reviewed'::text])"
        ),
        CheckConstraint(
            "status = ANY (ARRAY['draft'::text, 'reviewed'::text, 'published'::text])"
        ),
    )

    tag_id = Column(
        ForeignKey("tag.id", ondelete="CASCADE"), primary_key=True, nullable=False
    )
    locale = Column(
        ForeignKey("supported_locale.code"), primary_key=True, nullable=False
    )
    long_name = Column(Text)
    description = Column(Text)
    citation = Column(Text)
    status = Column(Text, nullable=False, server_default=text("'published'::text"))
    quality = Column(Text, nullable=False, server_default=text("'human'::text"))
    source_checksum = Column(Text, nullable=False)
    updated_by = Column(ForeignKey("users.id", ondelete="SET NULL"))
    created_at = Column(DateTime(True), nullable=False, server_default=text("now()"))
    updated_at = Column(DateTime(True), nullable=False, server_default=text("now()"))

    supported_locale = relationship("SupportedLocale")
    tag = relationship("Tag")
    users = relationship("Users")


class HitlRequest(Base):  # noqa
    __tablename__ = "hitl_request"
    __table_args__ = (UniqueConstraint("slick", "user"),)

    id = Column(
        BigInteger,
        primary_key=True,
        server_default=text("nextval('hitl_request_id_seq'::regclass)"),
    )
    slick = Column(ForeignKey("slick.id"), nullable=False)
    user = Column(ForeignKey("users.id"), nullable=False)
    date_requested = Column(DateTime, server_default=text("now()"))
    date_notified = Column(DateTime)
    escalation = Column(Text)

    slick1 = relationship("Slick")
    users = relationship("Users")


class HitlSlick(Base):  # noqa
    __tablename__ = "hitl_slick"

    id = Column(
        BigInteger,
        primary_key=True,
        server_default=text("nextval('hitl_slick_id_seq'::regclass)"),
    )
    slick = Column(ForeignKey("slick.id"), nullable=False)
    user = Column(ForeignKey("users.id"), nullable=False)
    cls = Column(ForeignKey("cls.id"), nullable=False)
    confidence = Column(Float(53))
    update_time = Column(DateTime, nullable=False, server_default=text("now()"))
    is_duplicate = Column(Boolean)

    cls1 = relationship("Cls")
    slick1 = relationship("Slick")
    users = relationship("Users")


t_slick_to_aoi = Table(
    "slick_to_aoi",
    metadata,
    Column(
        "slick",
        ForeignKey(
            "slick.id", ondelete="CASCADE", deferrable=True, initially="DEFERRED"
        ),
        primary_key=True,
        nullable=False,
    ),
    Column(
        "aoi",
        ForeignKey("aoi.id", ondelete="CASCADE", deferrable=True, initially="DEFERRED"),
        primary_key=True,
        nullable=False,
    ),
)


class SlickToSource(Base):  # noqa
    __tablename__ = "slick_to_source"
    __table_args__ = (UniqueConstraint("slick", "source"),)

    id = Column(
        BigInteger,
        primary_key=True,
        server_default=text("nextval('slick_to_source_id_seq'::regclass)"),
    )
    slick = Column(ForeignKey("slick.id"), nullable=False)
    source = Column(ForeignKey("source.id"), nullable=False)
    active = Column(Boolean, nullable=False)
    git_hash = Column(Text)
    git_tag = Column(Text)
    coincidence_score = Column(Float(53))
    collated_score = Column(Float(53))
    rank = Column(BigInteger)
    geojson_fc = Column(JSON, nullable=False)
    geometry = Column(
        Geography(srid=4326, from_text="ST_GeogFromText", name="geography"),
        nullable=False,
    )
    create_time = Column(DateTime, nullable=False, server_default=text("now()"))
    hitl_verification = Column(Boolean)
    hitl_confidence = Column(Float(53))
    hitl_user = Column(ForeignKey("users.id"))
    hitl_time = Column(DateTime)
    hitl_notes = Column(Text)

    users = relationship("Users")
    slick1 = relationship("Slick")
    source1 = relationship("Source")
