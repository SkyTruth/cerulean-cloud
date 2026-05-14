"""
0. Make any changes you want to EVERYWHERE ELSE that has #EditTheDatabase, but NOT here
1. Copy this comment
2. Run:
    Build the database locally using the readme
    sqlacodegen postgresql://user:password@localhost:5432/db --generator declarative --noviews --options noindexes --outfile cerulean_cloud/database_schema.py
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
7. Paste this comment
"""

import datetime
from typing import Any, Optional

from geoalchemy2.types import Geography, Geometry
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
    Double,
    ForeignKeyConstraint,
    Integer,
    PrimaryKeyConstraint,
    String,
    Table,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class AoiAccessType(Base):
    __tablename__ = "aoi_access_type"
    __table_args__ = (
        PrimaryKeyConstraint("id", name="aoi_access_type_pkey"),
        UniqueConstraint("short_name", name="aoi_access_type_short_name_key"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    short_name: Mapped[str] = mapped_column(Text, nullable=False)
    prop_keys: Mapped[list[str]] = mapped_column(ARRAY(Text()), nullable=False)

    aoi_type: Mapped[list["AoiType"]] = relationship(
        "AoiType", back_populates="aoi_access_type"
    )


class Cls(Base):
    __tablename__ = "cls"
    __table_args__ = (
        ForeignKeyConstraint(["supercls"], ["cls.id"], name="cls_supercls_fkey"),
        PrimaryKeyConstraint("id", name="cls_pkey"),
        UniqueConstraint("short_name", name="cls_short_name_key"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    short_name: Mapped[Optional[str]] = mapped_column(Text)
    long_name: Mapped[Optional[str]] = mapped_column(Text)
    supercls: Mapped[Optional[int]] = mapped_column(BigInteger)
    description: Mapped[Optional[str]] = mapped_column(Text)

    cls: Mapped[Optional["Cls"]] = relationship(
        "Cls", remote_side=[id], back_populates="cls_reverse"
    )
    cls_reverse: Mapped[list["Cls"]] = relationship(
        "Cls", remote_side=[supercls], back_populates="cls"
    )
    cls_i18n: Mapped[list["ClsI18n"]] = relationship("ClsI18n", back_populates="cls")
    slick: Mapped[list["Slick"]] = relationship("Slick", back_populates="cls_")
    hitl_slick: Mapped[list["HitlSlick"]] = relationship(
        "HitlSlick", back_populates="cls_"
    )


class Filter(Base):
    __tablename__ = "filter"
    __table_args__ = (PrimaryKeyConstraint("id", name="filter_pkey"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    json: Mapped[dict] = mapped_column(JSON, nullable=False)
    hash: Mapped[Optional[str]] = mapped_column(Text)

    subscription: Mapped[list["Subscription"]] = relationship(
        "Subscription", back_populates="filter_"
    )


class Frequency(Base):
    __tablename__ = "frequency"
    __table_args__ = (
        PrimaryKeyConstraint("id", name="frequency_pkey"),
        UniqueConstraint("short_name", name="frequency_short_name_key"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    short_name: Mapped[str] = mapped_column(Text, nullable=False)
    long_name: Mapped[Optional[str]] = mapped_column(Text)

    frequency_i18n: Mapped[list["FrequencyI18n"]] = relationship(
        "FrequencyI18n", back_populates="frequency"
    )
    subscription: Mapped[list["Subscription"]] = relationship(
        "Subscription", back_populates="frequency_"
    )


class Layer(Base):
    __tablename__ = "layer"
    __table_args__ = (
        PrimaryKeyConstraint("id", name="layer_pkey"),
        UniqueConstraint("short_name", name="layer_short_name_key"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    short_name: Mapped[str] = mapped_column(Text, nullable=False)
    long_name: Mapped[Optional[str]] = mapped_column(Text)
    citation: Mapped[Optional[str]] = mapped_column(Text)
    source_url: Mapped[Optional[str]] = mapped_column(Text)
    notes: Mapped[Optional[str]] = mapped_column(Text)
    start_time: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime)
    end_time: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime)
    json: Mapped[Optional[dict]] = mapped_column(JSON)
    update_time: Mapped[Optional[datetime.datetime]] = mapped_column(
        DateTime, server_default=text("now()")
    )

    layer_i18n: Mapped[list["LayerI18n"]] = relationship(
        "LayerI18n", back_populates="layer"
    )


class Model(Base):
    __tablename__ = "model"
    __table_args__ = (PrimaryKeyConstraint("id", name="model_pkey"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    type: Mapped[str] = mapped_column(Text, nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    layers: Mapped[list[str]] = mapped_column(ARRAY(Text()), nullable=False)
    cls_map: Mapped[dict] = mapped_column(JSON, nullable=False)
    tile_width_m: Mapped[int] = mapped_column(Integer, nullable=False)
    tile_width_px: Mapped[int] = mapped_column(Integer, nullable=False)
    thresholds: Mapped[dict] = mapped_column(JSON, nullable=False)
    update_time: Mapped[datetime.datetime] = mapped_column(
        DateTime, nullable=False, server_default=text("now()")
    )
    name: Mapped[Optional[str]] = mapped_column(Text)
    zoom_level: Mapped[Optional[int]] = mapped_column(
        Integer,
        Computed(
            "(round(log((2)::numeric, (40075000.0 / (tile_width_m)::numeric))) - (1)::numeric)",
            persisted=True,
        ),
    )
    scale: Mapped[Optional[int]] = mapped_column(
        Integer, Computed("round(((tile_width_px)::numeric / 256.0))", persisted=True)
    )
    epochs: Mapped[Optional[int]] = mapped_column(Integer)
    backbone_size: Mapped[Optional[int]] = mapped_column(Integer)
    pixel_f1: Mapped[Optional[float]] = mapped_column(Double(53))
    instance_f1: Mapped[Optional[float]] = mapped_column(Double(53))

    orchestrator_run: Mapped[list["OrchestratorRun"]] = relationship(
        "OrchestratorRun", back_populates="model_"
    )


class Permission(Base):
    __tablename__ = "permission"
    __table_args__ = (
        PrimaryKeyConstraint("id", name="permission_pkey"),
        UniqueConstraint("short_name", name="permission_short_name_key"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    short_name: Mapped[str] = mapped_column(Text, nullable=False)
    long_name: Mapped[str] = mapped_column(Text, nullable=False)

    aoi_type: Mapped[list["AoiType"]] = relationship(
        "AoiType", back_populates="permission"
    )
    permission_i18n: Mapped[list["PermissionI18n"]] = relationship(
        "PermissionI18n", back_populates="permission"
    )
    tag_read_perm: Mapped[list["Tag"]] = relationship(
        "Tag", foreign_keys="[Tag.read_perm]", back_populates="permission"
    )
    tag_write_perm: Mapped[list["Tag"]] = relationship(
        "Tag", foreign_keys="[Tag.write_perm]", back_populates="permission_"
    )


class Sentinel1Grd(Base):
    __tablename__ = "sentinel1_grd"
    __table_args__ = (
        PrimaryKeyConstraint("id", name="sentinel1_grd_pkey"),
        UniqueConstraint("scene_id", name="sentinel1_grd_scene_id_key"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    scene_id: Mapped[str] = mapped_column(String(200), nullable=False)
    start_time: Mapped[datetime.datetime] = mapped_column(DateTime, nullable=False)
    end_time: Mapped[datetime.datetime] = mapped_column(DateTime, nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    geometry: Mapped[Any] = mapped_column(
        Geography(
            "POLYGON",
            4326,
            2,
            from_text="ST_GeogFromText",
            name="geography",
            nullable=False,
        ),
        nullable=False,
    )
    absolute_orbit_number: Mapped[Optional[int]] = mapped_column(Integer)
    mode: Mapped[Optional[str]] = mapped_column(String(200))
    polarization: Mapped[Optional[str]] = mapped_column(String(200))
    scihub_ingestion_time: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime)

    orchestrator_run: Mapped[list["OrchestratorRun"]] = relationship(
        "OrchestratorRun", back_populates="sentinel1_grd_"
    )


class SourceType(Base):
    __tablename__ = "source_type"
    __table_args__ = (PrimaryKeyConstraint("id", name="source_type_pkey"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    table_name: Mapped[Optional[str]] = mapped_column(Text)
    long_name: Mapped[Optional[str]] = mapped_column(Text)
    short_name: Mapped[Optional[str]] = mapped_column(Text)
    citation: Mapped[Optional[str]] = mapped_column(Text)
    ext_id_name: Mapped[Optional[str]] = mapped_column(Text)

    source: Mapped[list["Source"]] = relationship(
        "Source", back_populates="source_type"
    )
    source_type_i18n: Mapped[list["SourceTypeI18n"]] = relationship(
        "SourceTypeI18n", back_populates="source_type"
    )


class SpatialRefSys(Base):
    __tablename__ = "spatial_ref_sys"
    __table_args__ = (
        CheckConstraint(
            "srid > 0 AND srid <= 998999", name="spatial_ref_sys_srid_check"
        ),
        PrimaryKeyConstraint("srid", name="spatial_ref_sys_pkey"),
    )

    srid: Mapped[int] = mapped_column(Integer, primary_key=True)
    auth_name: Mapped[Optional[str]] = mapped_column(String(256))
    auth_srid: Mapped[Optional[int]] = mapped_column(Integer)
    srtext: Mapped[Optional[str]] = mapped_column(String(2048))
    proj4text: Mapped[Optional[str]] = mapped_column(String(2048))


class SupportedLocale(Base):
    __tablename__ = "supported_locale"
    __table_args__ = (
        CheckConstraint(
            "NOT is_default OR fallback_code IS NULL",
            name="ck_supported_locale_default_fallback",
        ),
        CheckConstraint("code = lower(code)", name="ck_supported_locale_code_lower"),
        CheckConstraint(
            "fallback_code IS NULL OR fallback_code <> code",
            name="ck_supported_locale_fallback_self",
        ),
        CheckConstraint(
            "text_direction = ANY (ARRAY['ltr'::text, 'rtl'::text])",
            name="ck_supported_locale_direction",
        ),
        ForeignKeyConstraint(
            ["fallback_code"],
            ["supported_locale.code"],
            name="supported_locale_fallback_code_fkey",
        ),
        PrimaryKeyConstraint("code", name="supported_locale_pkey"),
    )

    code: Mapped[str] = mapped_column(Text, primary_key=True)
    english_name: Mapped[str] = mapped_column(Text, nullable=False)
    native_name: Mapped[str] = mapped_column(Text, nullable=False)
    text_direction: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'ltr'::text")
    )
    is_default: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("true")
    )
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False)
    fallback_code: Mapped[Optional[str]] = mapped_column(Text)
    notes: Mapped[Optional[str]] = mapped_column(Text)

    supported_locale: Mapped[Optional["SupportedLocale"]] = relationship(
        "SupportedLocale", remote_side=[code], back_populates="supported_locale_reverse"
    )
    supported_locale_reverse: Mapped[list["SupportedLocale"]] = relationship(
        "SupportedLocale",
        remote_side=[fallback_code],
        back_populates="supported_locale",
    )
    cls_i18n: Mapped[list["ClsI18n"]] = relationship(
        "ClsI18n", back_populates="supported_locale"
    )
    frequency_i18n: Mapped[list["FrequencyI18n"]] = relationship(
        "FrequencyI18n", back_populates="supported_locale"
    )
    layer_i18n: Mapped[list["LayerI18n"]] = relationship(
        "LayerI18n", back_populates="supported_locale"
    )
    permission_i18n: Mapped[list["PermissionI18n"]] = relationship(
        "PermissionI18n", back_populates="supported_locale"
    )
    source_type_i18n: Mapped[list["SourceTypeI18n"]] = relationship(
        "SourceTypeI18n", back_populates="supported_locale"
    )
    aoi_type_i18n: Mapped[list["AoiTypeI18n"]] = relationship(
        "AoiTypeI18n", back_populates="supported_locale"
    )
    tag_i18n: Mapped[list["TagI18n"]] = relationship(
        "TagI18n", back_populates="supported_locale"
    )
    aoi_i18n: Mapped[list["AoiI18n"]] = relationship(
        "AoiI18n", back_populates="supported_locale"
    )


class Trigger(Base):
    __tablename__ = "trigger"
    __table_args__ = (PrimaryKeyConstraint("id", name="trigger_pkey"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    trigger_time: Mapped[datetime.datetime] = mapped_column(
        DateTime, nullable=False, server_default=text("now()")
    )
    trigger_logs: Mapped[str] = mapped_column(Text, nullable=False)
    trigger_type: Mapped[str] = mapped_column(String(200), nullable=False)
    scene_count: Mapped[Optional[int]] = mapped_column(Integer)
    filtered_scene_count: Mapped[Optional[int]] = mapped_column(Integer)

    orchestrator_run: Mapped[list["OrchestratorRun"]] = relationship(
        "OrchestratorRun", back_populates="trigger_"
    )


class Users(Base):
    __tablename__ = "users"
    __table_args__ = (
        PrimaryKeyConstraint("id", name="users_pkey"),
        UniqueConstraint("email", name="users_email_key"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    email: Mapped[str] = mapped_column(Text, nullable=False)
    firstName: Mapped[Optional[str]] = mapped_column(Text)
    lastName: Mapped[Optional[str]] = mapped_column(Text)
    name: Mapped[Optional[str]] = mapped_column(Text)
    emailVerified: Mapped[Optional[bool]] = mapped_column(Boolean)
    image: Mapped[Optional[str]] = mapped_column(Text)
    role: Mapped[Optional[str]] = mapped_column(Text)
    organization: Mapped[Optional[str]] = mapped_column(Text)
    organizationType: Mapped[Optional[dict]] = mapped_column(JSONB)
    location: Mapped[Optional[str]] = mapped_column(Text)
    emailConsent: Mapped[Optional[bool]] = mapped_column(Boolean)
    banned: Mapped[Optional[bool]] = mapped_column(Boolean)
    banReason: Mapped[Optional[str]] = mapped_column(Text)
    banExpires: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime)
    createdAt: Mapped[Optional[datetime.datetime]] = mapped_column(
        DateTime, server_default=text("now()")
    )
    updatedAt: Mapped[Optional[datetime.datetime]] = mapped_column(
        DateTime, server_default=text("now()")
    )

    accounts: Mapped[list["Accounts"]] = relationship(
        "Accounts", back_populates="users"
    )
    aoi_type: Mapped[list["AoiType"]] = relationship("AoiType", back_populates="users")
    cls_i18n: Mapped[list["ClsI18n"]] = relationship("ClsI18n", back_populates="users")
    frequency_i18n: Mapped[list["FrequencyI18n"]] = relationship(
        "FrequencyI18n", back_populates="users"
    )
    layer_i18n: Mapped[list["LayerI18n"]] = relationship(
        "LayerI18n", back_populates="users"
    )
    permission_i18n: Mapped[list["PermissionI18n"]] = relationship(
        "PermissionI18n", back_populates="users"
    )
    sessions: Mapped[list["Sessions"]] = relationship(
        "Sessions", back_populates="users"
    )
    source_type_i18n: Mapped[list["SourceTypeI18n"]] = relationship(
        "SourceTypeI18n", back_populates="users"
    )
    subscription: Mapped[list["Subscription"]] = relationship(
        "Subscription", back_populates="users"
    )
    tag: Mapped[list["Tag"]] = relationship("Tag", back_populates="users")
    aoi_type_i18n: Mapped[list["AoiTypeI18n"]] = relationship(
        "AoiTypeI18n", back_populates="users"
    )
    tag_i18n: Mapped[list["TagI18n"]] = relationship("TagI18n", back_populates="users")
    aoi_i18n: Mapped[list["AoiI18n"]] = relationship("AoiI18n", back_populates="users")
    aoi_user: Mapped[list["AoiUser"]] = relationship("AoiUser", back_populates="users")
    hitl_request: Mapped[list["HitlRequest"]] = relationship(
        "HitlRequest", back_populates="users"
    )
    hitl_slick: Mapped[list["HitlSlick"]] = relationship(
        "HitlSlick", back_populates="users"
    )
    slick_to_source: Mapped[list["SlickToSource"]] = relationship(
        "SlickToSource", back_populates="users"
    )


class Verifications(Base):
    __tablename__ = "verifications"
    __table_args__ = (PrimaryKeyConstraint("id", name="verifications_pkey"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    identifier: Mapped[str] = mapped_column(Text, nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    expiresAt: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime)
    createdAt: Mapped[Optional[datetime.datetime]] = mapped_column(
        DateTime, server_default=text("now()")
    )
    updatedAt: Mapped[Optional[datetime.datetime]] = mapped_column(
        DateTime, server_default=text("now()")
    )


class Accounts(Base):
    __tablename__ = "accounts"
    __table_args__ = (
        ForeignKeyConstraint(["userId"], ["users.id"], name="accounts_userId_fkey"),
        PrimaryKeyConstraint("id", name="accounts_pkey"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    userId: Mapped[int] = mapped_column(BigInteger, nullable=False)
    providerId: Mapped[str] = mapped_column(Text, nullable=False)
    accountId: Mapped[str] = mapped_column(Text, nullable=False)
    refreshToken: Mapped[Optional[str]] = mapped_column(Text)
    accessToken: Mapped[Optional[str]] = mapped_column(Text)
    accessTokenExpiresAt: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime)
    idToken: Mapped[Optional[str]] = mapped_column(Text)
    scope: Mapped[Optional[str]] = mapped_column(Text)
    createdAt: Mapped[Optional[datetime.datetime]] = mapped_column(
        DateTime, server_default=text("now()")
    )
    updatedAt: Mapped[Optional[datetime.datetime]] = mapped_column(
        DateTime, server_default=text("now()")
    )

    users: Mapped["Users"] = relationship("Users", back_populates="accounts")


class AoiType(Base):
    __tablename__ = "aoi_type"
    __table_args__ = (
        CheckConstraint(
            "access_type IS NULL OR properties IS NOT NULL AND jsonb_typeof(properties) = 'object'::text AND\nCASE\n    WHEN NOT properties ? 'slick_to_aoi_buffer_m'::text THEN true\n    WHEN jsonb_typeof(properties -> 'slick_to_aoi_buffer_m'::text) = 'null'::text THEN true\n    WHEN jsonb_typeof(properties -> 'slick_to_aoi_buffer_m'::text) = 'number'::text THEN true\n    ELSE false\nEND AND (access_type = 'SHARED_DATASET'::text AND NULLIF(properties ->> 'asset_slug'::text, ''::text) IS NOT NULL AND NULLIF(properties ->> 'ext_id_field'::text, ''::text) IS NOT NULL OR access_type = 'DB_LOCAL'::text AND NULLIF(properties ->> 'table_name'::text, ''::text) IS NOT NULL AND NULLIF(properties ->> 'geog_col'::text, ''::text) IS NOT NULL AND NULLIF(properties ->> 'ext_id_col'::text, ''::text) IS NOT NULL OR access_type = 'DB_REMOTE'::text AND NULLIF(properties ->> 'db_conn_secret_name'::text, ''::text) IS NOT NULL AND NULLIF(properties ->> 'table_name'::text, ''::text) IS NOT NULL AND NULLIF(properties ->> 'geog_col'::text, ''::text) IS NOT NULL AND NULLIF(properties ->> 'ext_id_col'::text, ''::text) IS NOT NULL)",
            name="ck_aoi_type_access_properties",
        ),
        ForeignKeyConstraint(
            ["access_type"],
            ["aoi_access_type.short_name"],
            name="fk_aoi_type_access_type_aoi_access_type",
        ),
        ForeignKeyConstraint(["owner"], ["users.id"], name="fk_aoi_type_owner_users"),
        ForeignKeyConstraint(
            ["read_perm"], ["permission.id"], name="fk_aoi_type_read_perm_permission"
        ),
        PrimaryKeyConstraint("id", name="aoi_type_pkey"),
        UniqueConstraint("short_name", name="uq_aoi_type_short_name"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    short_name: Mapped[str] = mapped_column(Text, nullable=False)
    slick_to_aoi_enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("true")
    )
    table_name: Mapped[Optional[str]] = mapped_column(Text)
    long_name: Mapped[Optional[str]] = mapped_column(Text)
    source_url: Mapped[Optional[str]] = mapped_column(Text)
    citation: Mapped[Optional[str]] = mapped_column(Text)
    update_time: Mapped[Optional[datetime.datetime]] = mapped_column(
        DateTime, server_default=text("now()")
    )
    filter_toggle: Mapped[Optional[bool]] = mapped_column(Boolean)
    owner: Mapped[Optional[int]] = mapped_column(BigInteger)
    read_perm: Mapped[Optional[int]] = mapped_column(BigInteger)
    access_type: Mapped[Optional[str]] = mapped_column(Text)
    properties: Mapped[Optional[dict]] = mapped_column(JSONB)

    aoi_access_type: Mapped[Optional["AoiAccessType"]] = relationship(
        "AoiAccessType", back_populates="aoi_type"
    )
    users: Mapped[Optional["Users"]] = relationship("Users", back_populates="aoi_type")
    permission: Mapped[Optional["Permission"]] = relationship(
        "Permission", back_populates="aoi_type"
    )
    aoi: Mapped[list["Aoi"]] = relationship("Aoi", back_populates="aoi_type")
    aoi_type_i18n: Mapped[list["AoiTypeI18n"]] = relationship(
        "AoiTypeI18n", back_populates="aoi_type"
    )


class ClsI18n(Base):
    __tablename__ = "cls_i18n"
    __table_args__ = (
        CheckConstraint(
            "num_nonnulls(long_name, description) > 0",
            name="ck_cls_i18n_has_translation",
        ),
        CheckConstraint(
            "quality = ANY (ARRAY['human'::text, 'machine'::text, 'machine_reviewed'::text])",
            name="ck_cls_i18n_quality",
        ),
        CheckConstraint(
            "status = ANY (ARRAY['draft'::text, 'reviewed'::text, 'published'::text])",
            name="ck_cls_i18n_status",
        ),
        ForeignKeyConstraint(
            ["cls_id"], ["cls.id"], ondelete="CASCADE", name="cls_i18n_cls_id_fkey"
        ),
        ForeignKeyConstraint(
            ["locale"], ["supported_locale.code"], name="cls_i18n_locale_fkey"
        ),
        ForeignKeyConstraint(
            ["updated_by"],
            ["users.id"],
            ondelete="SET NULL",
            name="cls_i18n_updated_by_fkey",
        ),
        PrimaryKeyConstraint("cls_id", "locale", name="cls_i18n_pkey"),
    )

    cls_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    locale: Mapped[str] = mapped_column(Text, primary_key=True)
    status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'published'::text")
    )
    quality: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'human'::text")
    )
    source_checksum: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(True), nullable=False, server_default=text("now()")
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(True), nullable=False, server_default=text("now()")
    )
    long_name: Mapped[Optional[str]] = mapped_column(Text)
    description: Mapped[Optional[str]] = mapped_column(Text)
    updated_by: Mapped[Optional[int]] = mapped_column(BigInteger)

    cls: Mapped["Cls"] = relationship("Cls", back_populates="cls_i18n")
    supported_locale: Mapped["SupportedLocale"] = relationship(
        "SupportedLocale", back_populates="cls_i18n"
    )
    users: Mapped[Optional["Users"]] = relationship("Users", back_populates="cls_i18n")


class FrequencyI18n(Base):
    __tablename__ = "frequency_i18n"
    __table_args__ = (
        CheckConstraint(
            "long_name IS NOT NULL", name="ck_frequency_i18n_has_translation"
        ),
        CheckConstraint(
            "quality = ANY (ARRAY['human'::text, 'machine'::text, 'machine_reviewed'::text])",
            name="ck_frequency_i18n_quality",
        ),
        CheckConstraint(
            "status = ANY (ARRAY['draft'::text, 'reviewed'::text, 'published'::text])",
            name="ck_frequency_i18n_status",
        ),
        ForeignKeyConstraint(
            ["frequency_id"],
            ["frequency.id"],
            ondelete="CASCADE",
            name="frequency_i18n_frequency_id_fkey",
        ),
        ForeignKeyConstraint(
            ["locale"], ["supported_locale.code"], name="frequency_i18n_locale_fkey"
        ),
        ForeignKeyConstraint(
            ["updated_by"],
            ["users.id"],
            ondelete="SET NULL",
            name="frequency_i18n_updated_by_fkey",
        ),
        PrimaryKeyConstraint("frequency_id", "locale", name="frequency_i18n_pkey"),
    )

    frequency_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    locale: Mapped[str] = mapped_column(Text, primary_key=True)
    status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'published'::text")
    )
    quality: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'human'::text")
    )
    source_checksum: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(True), nullable=False, server_default=text("now()")
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(True), nullable=False, server_default=text("now()")
    )
    long_name: Mapped[Optional[str]] = mapped_column(Text)
    updated_by: Mapped[Optional[int]] = mapped_column(BigInteger)

    frequency: Mapped["Frequency"] = relationship(
        "Frequency", back_populates="frequency_i18n"
    )
    supported_locale: Mapped["SupportedLocale"] = relationship(
        "SupportedLocale", back_populates="frequency_i18n"
    )
    users: Mapped[Optional["Users"]] = relationship(
        "Users", back_populates="frequency_i18n"
    )


class LayerI18n(Base):
    __tablename__ = "layer_i18n"
    __table_args__ = (
        CheckConstraint(
            "num_nonnulls(long_name, notes, citation) > 0",
            name="ck_layer_i18n_has_translation",
        ),
        CheckConstraint(
            "quality = ANY (ARRAY['human'::text, 'machine'::text, 'machine_reviewed'::text])",
            name="ck_layer_i18n_quality",
        ),
        CheckConstraint(
            "status = ANY (ARRAY['draft'::text, 'reviewed'::text, 'published'::text])",
            name="ck_layer_i18n_status",
        ),
        ForeignKeyConstraint(
            ["layer_id"],
            ["layer.id"],
            ondelete="CASCADE",
            name="layer_i18n_layer_id_fkey",
        ),
        ForeignKeyConstraint(
            ["locale"], ["supported_locale.code"], name="layer_i18n_locale_fkey"
        ),
        ForeignKeyConstraint(
            ["updated_by"],
            ["users.id"],
            ondelete="SET NULL",
            name="layer_i18n_updated_by_fkey",
        ),
        PrimaryKeyConstraint("layer_id", "locale", name="layer_i18n_pkey"),
    )

    layer_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    locale: Mapped[str] = mapped_column(Text, primary_key=True)
    status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'published'::text")
    )
    quality: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'human'::text")
    )
    source_checksum: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(True), nullable=False, server_default=text("now()")
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(True), nullable=False, server_default=text("now()")
    )
    long_name: Mapped[Optional[str]] = mapped_column(Text)
    notes: Mapped[Optional[str]] = mapped_column(Text)
    citation: Mapped[Optional[str]] = mapped_column(Text)
    updated_by: Mapped[Optional[int]] = mapped_column(BigInteger)

    layer: Mapped["Layer"] = relationship("Layer", back_populates="layer_i18n")
    supported_locale: Mapped["SupportedLocale"] = relationship(
        "SupportedLocale", back_populates="layer_i18n"
    )
    users: Mapped[Optional["Users"]] = relationship(
        "Users", back_populates="layer_i18n"
    )


class OrchestratorRun(Base):
    __tablename__ = "orchestrator_run"
    __table_args__ = (
        ForeignKeyConstraint(
            ["model"], ["model.id"], name="orchestrator_run_model_fkey"
        ),
        ForeignKeyConstraint(
            ["sentinel1_grd"],
            ["sentinel1_grd.id"],
            name="orchestrator_run_sentinel1_grd_fkey",
        ),
        ForeignKeyConstraint(
            ["trigger"], ["trigger.id"], name="orchestrator_run_trigger_fkey"
        ),
        PrimaryKeyConstraint("id", name="orchestrator_run_pkey"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    inference_start_time: Mapped[datetime.datetime] = mapped_column(
        DateTime, nullable=False
    )
    inference_end_time: Mapped[datetime.datetime] = mapped_column(
        DateTime, nullable=False
    )
    inference_run_logs: Mapped[str] = mapped_column(Text, nullable=False)
    geometry: Mapped[Any] = mapped_column(
        Geography(
            "POLYGON",
            4326,
            2,
            from_text="ST_GeogFromText",
            name="geography",
            nullable=False,
        ),
        nullable=False,
    )
    trigger: Mapped[int] = mapped_column(BigInteger, nullable=False)
    model: Mapped[int] = mapped_column(Integer, nullable=False)
    base_tiles: Mapped[Optional[int]] = mapped_column(Integer)
    offset_tiles: Mapped[Optional[int]] = mapped_column(Integer)
    git_hash: Mapped[Optional[str]] = mapped_column(Text)
    git_tag: Mapped[Optional[str]] = mapped_column(String(200))
    zoom: Mapped[Optional[int]] = mapped_column(Integer)
    scale: Mapped[Optional[int]] = mapped_column(Integer)
    success: Mapped[Optional[bool]] = mapped_column(Boolean)
    sentinel1_grd: Mapped[Optional[int]] = mapped_column(BigInteger)
    sea_ice_date: Mapped[Optional[datetime.date]] = mapped_column(Date)
    dataset_versions: Mapped[Optional[dict]] = mapped_column(JSONB)

    model_: Mapped["Model"] = relationship("Model", back_populates="orchestrator_run")
    sentinel1_grd_: Mapped[Optional["Sentinel1Grd"]] = relationship(
        "Sentinel1Grd", back_populates="orchestrator_run"
    )
    trigger_: Mapped["Trigger"] = relationship(
        "Trigger", back_populates="orchestrator_run"
    )
    slick: Mapped[list["Slick"]] = relationship(
        "Slick", back_populates="orchestrator_run_"
    )


class PermissionI18n(Base):
    __tablename__ = "permission_i18n"
    __table_args__ = (
        CheckConstraint(
            "long_name IS NOT NULL", name="ck_permission_i18n_has_translation"
        ),
        CheckConstraint(
            "quality = ANY (ARRAY['human'::text, 'machine'::text, 'machine_reviewed'::text])",
            name="ck_permission_i18n_quality",
        ),
        CheckConstraint(
            "status = ANY (ARRAY['draft'::text, 'reviewed'::text, 'published'::text])",
            name="ck_permission_i18n_status",
        ),
        ForeignKeyConstraint(
            ["locale"], ["supported_locale.code"], name="permission_i18n_locale_fkey"
        ),
        ForeignKeyConstraint(
            ["permission_id"],
            ["permission.id"],
            ondelete="CASCADE",
            name="permission_i18n_permission_id_fkey",
        ),
        ForeignKeyConstraint(
            ["updated_by"],
            ["users.id"],
            ondelete="SET NULL",
            name="permission_i18n_updated_by_fkey",
        ),
        PrimaryKeyConstraint("permission_id", "locale", name="permission_i18n_pkey"),
    )

    permission_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    locale: Mapped[str] = mapped_column(Text, primary_key=True)
    status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'published'::text")
    )
    quality: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'human'::text")
    )
    source_checksum: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(True), nullable=False, server_default=text("now()")
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(True), nullable=False, server_default=text("now()")
    )
    long_name: Mapped[Optional[str]] = mapped_column(Text)
    updated_by: Mapped[Optional[int]] = mapped_column(BigInteger)

    supported_locale: Mapped["SupportedLocale"] = relationship(
        "SupportedLocale", back_populates="permission_i18n"
    )
    permission: Mapped["Permission"] = relationship(
        "Permission", back_populates="permission_i18n"
    )
    users: Mapped[Optional["Users"]] = relationship(
        "Users", back_populates="permission_i18n"
    )


class Sessions(Base):
    __tablename__ = "sessions"
    __table_args__ = (
        ForeignKeyConstraint(["userId"], ["users.id"], name="sessions_userId_fkey"),
        PrimaryKeyConstraint("id", name="sessions_pkey"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    userId: Mapped[int] = mapped_column(BigInteger, nullable=False)
    expiresAt: Mapped[datetime.datetime] = mapped_column(DateTime, nullable=False)
    token: Mapped[str] = mapped_column(Text, nullable=False)
    createdAt: Mapped[Optional[datetime.datetime]] = mapped_column(
        DateTime, server_default=text("now()")
    )
    updatedAt: Mapped[Optional[datetime.datetime]] = mapped_column(
        DateTime, server_default=text("now()")
    )
    impersonatedBy: Mapped[Optional[str]] = mapped_column(Text)
    ipAddress: Mapped[Optional[str]] = mapped_column(Text)
    userAgent: Mapped[Optional[str]] = mapped_column(Text)

    users: Mapped["Users"] = relationship("Users", back_populates="sessions")


class Source(Base):
    __tablename__ = "source"
    __table_args__ = (
        ForeignKeyConstraint(["type"], ["source_type.id"], name="source_type_fkey"),
        PrimaryKeyConstraint("id", name="source_pkey"),
        UniqueConstraint("ext_id", "type", name="uq_source_extid_type"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    type: Mapped[int] = mapped_column(BigInteger, nullable=False)
    ext_id: Mapped[str] = mapped_column(Text, nullable=False)

    source_type: Mapped["SourceType"] = relationship(
        "SourceType", back_populates="source"
    )
    slick_to_source: Mapped[list["SlickToSource"]] = relationship(
        "SlickToSource", back_populates="source_"
    )


class SourceTypeI18n(Base):
    __tablename__ = "source_type_i18n"
    __table_args__ = (
        CheckConstraint(
            "num_nonnulls(long_name, citation) > 0",
            name="ck_source_type_i18n_has_translation",
        ),
        CheckConstraint(
            "quality = ANY (ARRAY['human'::text, 'machine'::text, 'machine_reviewed'::text])",
            name="ck_source_type_i18n_quality",
        ),
        CheckConstraint(
            "status = ANY (ARRAY['draft'::text, 'reviewed'::text, 'published'::text])",
            name="ck_source_type_i18n_status",
        ),
        ForeignKeyConstraint(
            ["locale"], ["supported_locale.code"], name="source_type_i18n_locale_fkey"
        ),
        ForeignKeyConstraint(
            ["source_type_id"],
            ["source_type.id"],
            ondelete="CASCADE",
            name="source_type_i18n_source_type_id_fkey",
        ),
        ForeignKeyConstraint(
            ["updated_by"],
            ["users.id"],
            ondelete="SET NULL",
            name="source_type_i18n_updated_by_fkey",
        ),
        PrimaryKeyConstraint("source_type_id", "locale", name="source_type_i18n_pkey"),
    )

    source_type_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    locale: Mapped[str] = mapped_column(Text, primary_key=True)
    status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'published'::text")
    )
    quality: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'human'::text")
    )
    source_checksum: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(True), nullable=False, server_default=text("now()")
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(True), nullable=False, server_default=text("now()")
    )
    long_name: Mapped[Optional[str]] = mapped_column(Text)
    citation: Mapped[Optional[str]] = mapped_column(Text)
    updated_by: Mapped[Optional[int]] = mapped_column(BigInteger)

    supported_locale: Mapped["SupportedLocale"] = relationship(
        "SupportedLocale", back_populates="source_type_i18n"
    )
    source_type: Mapped["SourceType"] = relationship(
        "SourceType", back_populates="source_type_i18n"
    )
    users: Mapped[Optional["Users"]] = relationship(
        "Users", back_populates="source_type_i18n"
    )


class Subscription(Base):
    __tablename__ = "subscription"
    __table_args__ = (
        ForeignKeyConstraint(
            ["filter"], ["filter.id"], name="subscription_filter_fkey"
        ),
        ForeignKeyConstraint(
            ["frequency"], ["frequency.id"], name="subscription_frequency_fkey"
        ),
        ForeignKeyConstraint(["user"], ["users.id"], name="subscription_user_fkey"),
        PrimaryKeyConstraint("id", name="subscription_pkey"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    user: Mapped[int] = mapped_column(BigInteger, nullable=False)
    filter: Mapped[int] = mapped_column(BigInteger, nullable=False)
    frequency: Mapped[int] = mapped_column(BigInteger, nullable=False)
    active: Mapped[Optional[bool]] = mapped_column(Boolean)
    create_time: Mapped[Optional[datetime.datetime]] = mapped_column(
        DateTime, server_default=text("now()")
    )
    update_time: Mapped[Optional[datetime.datetime]] = mapped_column(
        DateTime, server_default=text("now()")
    )

    filter_: Mapped["Filter"] = relationship("Filter", back_populates="subscription")
    frequency_: Mapped["Frequency"] = relationship(
        "Frequency", back_populates="subscription"
    )
    users: Mapped["Users"] = relationship("Users", back_populates="subscription")


class Tag(Base):
    __tablename__ = "tag"
    __table_args__ = (
        ForeignKeyConstraint(["owner"], ["users.id"], name="tag_owner_fkey"),
        ForeignKeyConstraint(
            ["read_perm"], ["permission.id"], name="tag_read_perm_fkey"
        ),
        ForeignKeyConstraint(
            ["write_perm"], ["permission.id"], name="tag_write_perm_fkey"
        ),
        PrimaryKeyConstraint("id", name="tag_pkey"),
        UniqueConstraint("short_name", name="tag_short_name_key"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    short_name: Mapped[str] = mapped_column(Text, nullable=False)
    long_name: Mapped[str] = mapped_column(Text, nullable=False)
    public: Mapped[bool] = mapped_column(Boolean, nullable=False)
    source_profile: Mapped[bool] = mapped_column(Boolean, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    citation: Mapped[Optional[str]] = mapped_column(Text)
    owner: Mapped[Optional[int]] = mapped_column(BigInteger)
    read_perm: Mapped[Optional[int]] = mapped_column(BigInteger)
    write_perm: Mapped[Optional[int]] = mapped_column(BigInteger)

    users: Mapped[Optional["Users"]] = relationship("Users", back_populates="tag")
    permission: Mapped[Optional["Permission"]] = relationship(
        "Permission", foreign_keys=[read_perm], back_populates="tag_read_perm"
    )
    permission_: Mapped[Optional["Permission"]] = relationship(
        "Permission", foreign_keys=[write_perm], back_populates="tag_write_perm"
    )
    source_to_tag: Mapped[list["SourceToTag"]] = relationship(
        "SourceToTag", back_populates="tag_"
    )
    tag_i18n: Mapped[list["TagI18n"]] = relationship("TagI18n", back_populates="tag")


class Aoi(Base):
    __tablename__ = "aoi"
    __table_args__ = (
        ForeignKeyConstraint(["type"], ["aoi_type.id"], name="aoi_type_fkey"),
        PrimaryKeyConstraint("id", name="aoi_pkey"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    type: Mapped[int] = mapped_column(BigInteger, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    geometry: Mapped[Optional[Any]] = mapped_column(
        Geography(
            "MULTIPOLYGON", 4326, 2, from_text="ST_GeogFromText", name="geography"
        )
    )
    ext_id: Mapped[Optional[str]] = mapped_column(Text)

    aoi_type: Mapped["AoiType"] = relationship("AoiType", back_populates="aoi")
    slick: Mapped[list["Slick"]] = relationship(
        "Slick", secondary="slick_to_aoi", back_populates="aoi"
    )
    aoi_i18n: Mapped[list["AoiI18n"]] = relationship("AoiI18n", back_populates="aoi")


class AoiTypeI18n(Base):
    __tablename__ = "aoi_type_i18n"
    __table_args__ = (
        CheckConstraint(
            "num_nonnulls(long_name, citation) > 0",
            name="ck_aoi_type_i18n_has_translation",
        ),
        CheckConstraint(
            "quality = ANY (ARRAY['human'::text, 'machine'::text, 'machine_reviewed'::text])",
            name="ck_aoi_type_i18n_quality",
        ),
        CheckConstraint(
            "status = ANY (ARRAY['draft'::text, 'reviewed'::text, 'published'::text])",
            name="ck_aoi_type_i18n_status",
        ),
        ForeignKeyConstraint(
            ["aoi_type_id"],
            ["aoi_type.id"],
            ondelete="CASCADE",
            name="aoi_type_i18n_aoi_type_id_fkey",
        ),
        ForeignKeyConstraint(
            ["locale"], ["supported_locale.code"], name="aoi_type_i18n_locale_fkey"
        ),
        ForeignKeyConstraint(
            ["updated_by"],
            ["users.id"],
            ondelete="SET NULL",
            name="aoi_type_i18n_updated_by_fkey",
        ),
        PrimaryKeyConstraint("aoi_type_id", "locale", name="aoi_type_i18n_pkey"),
    )

    aoi_type_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    locale: Mapped[str] = mapped_column(Text, primary_key=True)
    status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'published'::text")
    )
    quality: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'human'::text")
    )
    source_checksum: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(True), nullable=False, server_default=text("now()")
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(True), nullable=False, server_default=text("now()")
    )
    long_name: Mapped[Optional[str]] = mapped_column(Text)
    citation: Mapped[Optional[str]] = mapped_column(Text)
    updated_by: Mapped[Optional[int]] = mapped_column(BigInteger)

    aoi_type: Mapped["AoiType"] = relationship(
        "AoiType", back_populates="aoi_type_i18n"
    )
    supported_locale: Mapped["SupportedLocale"] = relationship(
        "SupportedLocale", back_populates="aoi_type_i18n"
    )
    users: Mapped[Optional["Users"]] = relationship(
        "Users", back_populates="aoi_type_i18n"
    )


class Slick(Base):
    __tablename__ = "slick"
    __table_args__ = (
        ForeignKeyConstraint(["hitl_cls"], ["cls.id"], name="slick_hitl_cls_fkey"),
        ForeignKeyConstraint(
            ["orchestrator_run"],
            ["orchestrator_run.id"],
            name="slick_orchestrator_run_fkey",
        ),
        PrimaryKeyConstraint("id", name="slick_pkey"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    slick_timestamp: Mapped[datetime.datetime] = mapped_column(DateTime, nullable=False)
    geometry: Mapped[Any] = mapped_column(
        Geography(
            "MULTIPOLYGON",
            4326,
            2,
            from_text="ST_GeogFromText",
            name="geography",
            nullable=False,
        ),
        nullable=False,
    )
    active: Mapped[bool] = mapped_column(Boolean, nullable=False)
    orchestrator_run: Mapped[int] = mapped_column(BigInteger, nullable=False)
    create_time: Mapped[datetime.datetime] = mapped_column(
        DateTime, nullable=False, server_default=text("now()")
    )
    inference_idx: Mapped[int] = mapped_column(Integer, nullable=False)
    cls: Mapped[int] = mapped_column(Integer, nullable=False)
    hitl_cls: Mapped[Optional[int]] = mapped_column(BigInteger)
    machine_confidence: Mapped[Optional[float]] = mapped_column(Double(53))
    precursor_slicks: Mapped[Optional[list[int]]] = mapped_column(ARRAY(BigInteger()))
    notes: Mapped[Optional[str]] = mapped_column(Text)
    centerlines: Mapped[Optional[dict]] = mapped_column(JSON)
    aspect_ratio_factor: Mapped[Optional[float]] = mapped_column(Double(53))
    length: Mapped[Optional[float]] = mapped_column(Double(53))
    area: Mapped[Optional[float]] = mapped_column(Double(53))
    perimeter: Mapped[Optional[float]] = mapped_column(Double(53))
    centroid: Mapped[Optional[Any]] = mapped_column(
        Geography("POINT", 4326, 2, from_text="ST_GeogFromText", name="geography")
    )
    polsby_popper: Mapped[Optional[float]] = mapped_column(Double(53))
    fill_factor: Mapped[Optional[float]] = mapped_column(Double(53))
    geometry_count: Mapped[Optional[int]] = mapped_column(Integer)
    largest_area: Mapped[Optional[float]] = mapped_column(Double(53))
    median_area: Mapped[Optional[float]] = mapped_column(Double(53))
    geometric_slick_potential: Mapped[Optional[float]] = mapped_column(Double(53))
    geom_3857_simplified: Mapped[Optional[Any]] = mapped_column(
        Geometry(srid=3857, dimension=2, from_text="ST_GeomFromEWKT", name="geometry"),
        Computed(
            "st_simplifypreservetopology(st_transform((geometry)::geometry, 3857), (100)::double precision)",
            persisted=True,
        ),
    )
    centroid_3857: Mapped[Optional[Any]] = mapped_column(
        Geometry("POINT", 3857, 2, from_text="ST_GeomFromEWKT", name="geometry"),
        Computed("st_transform((centroid)::geometry, 3857)", persisted=True),
    )
    geom_3857: Mapped[Optional[Any]] = mapped_column(
        Geometry(srid=3857, dimension=2, from_text="ST_GeomFromEWKT", name="geometry"),
        Computed("st_transform((geometry)::geometry, 3857)", persisted=True),
    )

    aoi: Mapped[list["Aoi"]] = relationship(
        "Aoi", secondary="slick_to_aoi", back_populates="slick"
    )
    cls_: Mapped[Optional["Cls"]] = relationship("Cls", back_populates="slick")
    orchestrator_run_: Mapped["OrchestratorRun"] = relationship(
        "OrchestratorRun", back_populates="slick"
    )
    hitl_request: Mapped[list["HitlRequest"]] = relationship(
        "HitlRequest", back_populates="slick_"
    )
    hitl_slick: Mapped[list["HitlSlick"]] = relationship(
        "HitlSlick", back_populates="slick_"
    )
    slick_to_source: Mapped[list["SlickToSource"]] = relationship(
        "SlickToSource", back_populates="slick_"
    )


class SourceDark(Source):
    __tablename__ = "source_dark"
    __table_args__ = (
        ForeignKeyConstraint(
            ["source_id"], ["source.id"], name="source_dark_source_id_fkey"
        ),
        PrimaryKeyConstraint("source_id", name="source_dark_pkey"),
    )

    source_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    geometry: Mapped[Any] = mapped_column(
        Geography(
            "POINT",
            4326,
            2,
            from_text="ST_GeogFromText",
            name="geography",
            nullable=False,
        ),
        nullable=False,
    )
    scene_id: Mapped[Optional[str]] = mapped_column(Text)
    length_m: Mapped[Optional[float]] = mapped_column(Double(53))
    detection_probability: Mapped[Optional[float]] = mapped_column(Double(53))


class SourceInfra(Source):
    __tablename__ = "source_infra"
    __table_args__ = (
        ForeignKeyConstraint(
            ["source_id"], ["source.id"], name="source_infra_source_id_fkey"
        ),
        PrimaryKeyConstraint("source_id", name="source_infra_pkey"),
    )

    source_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    geometry: Mapped[Any] = mapped_column(
        Geography(
            "POINT",
            4326,
            2,
            from_text="ST_GeogFromText",
            name="geography",
            nullable=False,
        ),
        nullable=False,
    )
    ext_name: Mapped[Optional[str]] = mapped_column(Text)
    operator: Mapped[Optional[str]] = mapped_column(Text)
    sovereign: Mapped[Optional[str]] = mapped_column(Text)
    orig_yr: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime)
    last_known_status: Mapped[Optional[str]] = mapped_column(Text)
    first_detection: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime)
    last_detection: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime)
    mmsi: Mapped[Optional[str]] = mapped_column(Text)


class SourceNatural(Source):
    __tablename__ = "source_natural"
    __table_args__ = (
        ForeignKeyConstraint(
            ["source_id"], ["source.id"], name="source_natural_source_id_fkey"
        ),
        PrimaryKeyConstraint("source_id", name="source_natural_pkey"),
    )

    source_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    geometry: Mapped[Any] = mapped_column(
        Geography(
            "POINT",
            4326,
            2,
            from_text="ST_GeogFromText",
            name="geography",
            nullable=False,
        ),
        nullable=False,
    )


class SourceToTag(Base):
    __tablename__ = "source_to_tag"
    __table_args__ = (
        ForeignKeyConstraint(["tag"], ["tag.id"], name="source_to_tag_tag_fkey"),
        PrimaryKeyConstraint(
            "source_ext_id", "source_type", "tag", name="source_to_tag_pkey"
        ),
    )

    source_ext_id: Mapped[str] = mapped_column(Text, primary_key=True)
    source_type: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    tag: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    create_time: Mapped[datetime.datetime] = mapped_column(
        DateTime, nullable=False, server_default=text("now()")
    )

    tag_: Mapped["Tag"] = relationship("Tag", back_populates="source_to_tag")


class SourceVessel(Source):
    __tablename__ = "source_vessel"
    __table_args__ = (
        ForeignKeyConstraint(
            ["source_id"], ["source.id"], name="source_vessel_source_id_fkey"
        ),
        PrimaryKeyConstraint("source_id", name="source_vessel_pkey"),
    )

    source_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    ext_name: Mapped[Optional[str]] = mapped_column(Text)
    ext_shiptype: Mapped[Optional[str]] = mapped_column(Text)
    flag: Mapped[Optional[str]] = mapped_column(Text)


class TagI18n(Base):
    __tablename__ = "tag_i18n"
    __table_args__ = (
        CheckConstraint(
            "num_nonnulls(long_name, description, citation) > 0",
            name="ck_tag_i18n_has_translation",
        ),
        CheckConstraint(
            "quality = ANY (ARRAY['human'::text, 'machine'::text, 'machine_reviewed'::text])",
            name="ck_tag_i18n_quality",
        ),
        CheckConstraint(
            "status = ANY (ARRAY['draft'::text, 'reviewed'::text, 'published'::text])",
            name="ck_tag_i18n_status",
        ),
        ForeignKeyConstraint(
            ["locale"], ["supported_locale.code"], name="tag_i18n_locale_fkey"
        ),
        ForeignKeyConstraint(
            ["tag_id"], ["tag.id"], ondelete="CASCADE", name="tag_i18n_tag_id_fkey"
        ),
        ForeignKeyConstraint(
            ["updated_by"],
            ["users.id"],
            ondelete="SET NULL",
            name="tag_i18n_updated_by_fkey",
        ),
        PrimaryKeyConstraint("tag_id", "locale", name="tag_i18n_pkey"),
    )

    tag_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    locale: Mapped[str] = mapped_column(Text, primary_key=True)
    status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'published'::text")
    )
    quality: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'human'::text")
    )
    source_checksum: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(True), nullable=False, server_default=text("now()")
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(True), nullable=False, server_default=text("now()")
    )
    long_name: Mapped[Optional[str]] = mapped_column(Text)
    description: Mapped[Optional[str]] = mapped_column(Text)
    citation: Mapped[Optional[str]] = mapped_column(Text)
    updated_by: Mapped[Optional[int]] = mapped_column(BigInteger)

    supported_locale: Mapped["SupportedLocale"] = relationship(
        "SupportedLocale", back_populates="tag_i18n"
    )
    tag: Mapped["Tag"] = relationship("Tag", back_populates="tag_i18n")
    users: Mapped[Optional["Users"]] = relationship("Users", back_populates="tag_i18n")


t_aoi_chunks = Table(
    "aoi_chunks",
    Base.metadata,
    Column("id", BigInteger),
    Column(
        "geometry",
        Geometry(
            "POLYGON",
            4326,
            2,
            from_text="ST_GeomFromEWKT",
            name="geometry",
            nullable=False,
        ),
        nullable=False,
    ),
    ForeignKeyConstraint(
        ["id"],
        ["aoi.id"],
        ondelete="CASCADE",
        deferrable=True,
        initially="DEFERRED",
        name="aoi_chunks_id_fkey",
    ),
)


class AoiEez(Aoi):
    __tablename__ = "aoi_eez"
    __table_args__ = (
        ForeignKeyConstraint(["aoi_id"], ["aoi.id"], name="aoi_eez_aoi_id_fkey"),
        PrimaryKeyConstraint("aoi_id", name="aoi_eez_pkey"),
    )

    aoi_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    mrgid: Mapped[Optional[int]] = mapped_column(Integer)
    sovereigns: Mapped[Optional[list[str]]] = mapped_column(ARRAY(Text()))


class AoiI18n(Base):
    __tablename__ = "aoi_i18n"
    __table_args__ = (
        CheckConstraint("name <> ''::text", name="ck_aoi_i18n_name_not_empty"),
        CheckConstraint(
            "quality = ANY (ARRAY['human'::text, 'machine'::text, 'machine_reviewed'::text])",
            name="ck_aoi_i18n_quality",
        ),
        CheckConstraint(
            "status = ANY (ARRAY['draft'::text, 'reviewed'::text, 'published'::text])",
            name="ck_aoi_i18n_status",
        ),
        ForeignKeyConstraint(
            ["aoi_id"], ["aoi.id"], ondelete="CASCADE", name="aoi_i18n_aoi_id_fkey"
        ),
        ForeignKeyConstraint(
            ["locale"], ["supported_locale.code"], name="aoi_i18n_locale_fkey"
        ),
        ForeignKeyConstraint(
            ["updated_by"],
            ["users.id"],
            ondelete="SET NULL",
            name="aoi_i18n_updated_by_fkey",
        ),
        PrimaryKeyConstraint("aoi_id", "locale", name="aoi_i18n_pkey"),
    )

    aoi_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    locale: Mapped[str] = mapped_column(Text, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'published'::text")
    )
    quality: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'human'::text")
    )
    source_checksum: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(True), nullable=False, server_default=text("now()")
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(True), nullable=False, server_default=text("now()")
    )
    updated_by: Mapped[Optional[int]] = mapped_column(BigInteger)

    aoi: Mapped["Aoi"] = relationship("Aoi", back_populates="aoi_i18n")
    supported_locale: Mapped["SupportedLocale"] = relationship(
        "SupportedLocale", back_populates="aoi_i18n"
    )
    users: Mapped[Optional["Users"]] = relationship("Users", back_populates="aoi_i18n")


class AoiIho(Aoi):
    __tablename__ = "aoi_iho"
    __table_args__ = (
        ForeignKeyConstraint(["aoi_id"], ["aoi.id"], name="aoi_iho_aoi_id_fkey"),
        PrimaryKeyConstraint("aoi_id", name="aoi_iho_pkey"),
    )

    aoi_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    mrgid: Mapped[Optional[int]] = mapped_column(Integer)


class AoiMpa(Aoi):
    __tablename__ = "aoi_mpa"
    __table_args__ = (
        ForeignKeyConstraint(["aoi_id"], ["aoi.id"], name="aoi_mpa_aoi_id_fkey"),
        PrimaryKeyConstraint("aoi_id", name="aoi_mpa_pkey"),
    )

    aoi_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    wdpaid: Mapped[Optional[int]] = mapped_column(Integer)
    desig: Mapped[Optional[str]] = mapped_column(Text)
    desig_type: Mapped[Optional[str]] = mapped_column(Text)
    status_yr: Mapped[Optional[int]] = mapped_column(Integer)
    mang_auth: Mapped[Optional[str]] = mapped_column(Text)
    parent_iso: Mapped[Optional[str]] = mapped_column(Text)


class AoiUser(Aoi):
    __tablename__ = "aoi_user"
    __table_args__ = (
        ForeignKeyConstraint(["aoi_id"], ["aoi.id"], name="aoi_user_aoi_id_fkey"),
        ForeignKeyConstraint(["user"], ["users.id"], name="aoi_user_user_fkey"),
        PrimaryKeyConstraint("aoi_id", name="aoi_user_pkey"),
    )

    aoi_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    user: Mapped[Optional[int]] = mapped_column(BigInteger)
    create_time: Mapped[Optional[datetime.datetime]] = mapped_column(
        DateTime, server_default=text("now()")
    )
    geometry: Mapped[Optional[Any]] = mapped_column(
        Geography(dimension=2, from_text="ST_GeogFromText", name="geography")
    )

    users: Mapped[Optional["Users"]] = relationship("Users", back_populates="aoi_user")


class HitlRequest(Base):
    __tablename__ = "hitl_request"
    __table_args__ = (
        ForeignKeyConstraint(["slick"], ["slick.id"], name="hitl_request_slick_fkey"),
        ForeignKeyConstraint(["user"], ["users.id"], name="hitl_request_user_fkey"),
        PrimaryKeyConstraint("id", name="hitl_request_pkey"),
        UniqueConstraint("slick", "user", name="uq_hitl_request_slick_user"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    slick: Mapped[int] = mapped_column(BigInteger, nullable=False)
    user: Mapped[int] = mapped_column(BigInteger, nullable=False)
    date_requested: Mapped[Optional[datetime.datetime]] = mapped_column(
        DateTime, server_default=text("now()")
    )
    date_notified: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime)
    escalation: Mapped[Optional[str]] = mapped_column(Text)

    slick_: Mapped["Slick"] = relationship("Slick", back_populates="hitl_request")
    users: Mapped["Users"] = relationship("Users", back_populates="hitl_request")


class HitlSlick(Base):
    __tablename__ = "hitl_slick"
    __table_args__ = (
        ForeignKeyConstraint(["cls"], ["cls.id"], name="hitl_slick_cls_fkey"),
        ForeignKeyConstraint(["slick"], ["slick.id"], name="hitl_slick_slick_fkey"),
        ForeignKeyConstraint(["user"], ["users.id"], name="hitl_slick_user_fkey"),
        PrimaryKeyConstraint("id", name="hitl_slick_pkey"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    slick: Mapped[int] = mapped_column(BigInteger, nullable=False)
    user: Mapped[int] = mapped_column(BigInteger, nullable=False)
    cls: Mapped[int] = mapped_column(BigInteger, nullable=False)
    update_time: Mapped[datetime.datetime] = mapped_column(
        DateTime, nullable=False, server_default=text("now()")
    )
    confidence: Mapped[Optional[float]] = mapped_column(Double(53))
    is_duplicate: Mapped[Optional[bool]] = mapped_column(Boolean)

    cls_: Mapped["Cls"] = relationship("Cls", back_populates="hitl_slick")
    slick_: Mapped["Slick"] = relationship("Slick", back_populates="hitl_slick")
    users: Mapped["Users"] = relationship("Users", back_populates="hitl_slick")


t_slick_to_aoi = Table(
    "slick_to_aoi",
    Base.metadata,
    Column("slick", BigInteger, primary_key=True),
    Column("aoi", BigInteger, primary_key=True),
    ForeignKeyConstraint(
        ["aoi"],
        ["aoi.id"],
        ondelete="CASCADE",
        deferrable=True,
        initially="DEFERRED",
        name="slick_to_aoi_aoi_fkey",
    ),
    ForeignKeyConstraint(
        ["slick"],
        ["slick.id"],
        ondelete="CASCADE",
        deferrable=True,
        initially="DEFERRED",
        name="slick_to_aoi_slick_fkey",
    ),
    PrimaryKeyConstraint("slick", "aoi", name="slick_to_aoi_pkey"),
)


class SlickToSource(Base):
    __tablename__ = "slick_to_source"
    __table_args__ = (
        ForeignKeyConstraint(
            ["hitl_user"], ["users.id"], name="slick_to_source_hitl_user_fkey"
        ),
        ForeignKeyConstraint(
            ["slick"], ["slick.id"], name="slick_to_source_slick_fkey"
        ),
        ForeignKeyConstraint(
            ["source"], ["source.id"], name="slick_to_source_source_fkey"
        ),
        PrimaryKeyConstraint("id", name="slick_to_source_pkey"),
        UniqueConstraint("slick", "source", name="uq_slick_to_source_slick_source"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    slick: Mapped[int] = mapped_column(BigInteger, nullable=False)
    source: Mapped[int] = mapped_column(BigInteger, nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False)
    geojson_fc: Mapped[dict] = mapped_column(JSON, nullable=False)
    geometry: Mapped[Any] = mapped_column(
        Geography(
            srid=4326,
            dimension=2,
            from_text="ST_GeogFromText",
            name="geography",
            nullable=False,
        ),
        nullable=False,
    )
    create_time: Mapped[datetime.datetime] = mapped_column(
        DateTime, nullable=False, server_default=text("now()")
    )
    git_hash: Mapped[Optional[str]] = mapped_column(Text)
    git_tag: Mapped[Optional[str]] = mapped_column(Text)
    coincidence_score: Mapped[Optional[float]] = mapped_column(Double(53))
    collated_score: Mapped[Optional[float]] = mapped_column(Double(53))
    rank: Mapped[Optional[int]] = mapped_column(BigInteger)
    hitl_verification: Mapped[Optional[bool]] = mapped_column(Boolean)
    hitl_confidence: Mapped[Optional[float]] = mapped_column(Double(53))
    hitl_user: Mapped[Optional[int]] = mapped_column(BigInteger)
    hitl_time: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime)
    hitl_notes: Mapped[Optional[str]] = mapped_column(Text)

    users: Mapped[Optional["Users"]] = relationship(
        "Users", back_populates="slick_to_source"
    )
    slick_: Mapped["Slick"] = relationship("Slick", back_populates="slick_to_source")
    source_: Mapped["Source"] = relationship("Source", back_populates="slick_to_source")
