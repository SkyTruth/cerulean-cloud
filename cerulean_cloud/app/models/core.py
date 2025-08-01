"""
Core (non-spatial) tables that do not rely on PostGIS types.
Edit these first; rely on Alembic autogenerate for migrations.
"""

from sqlalchemy import (
    ARRAY,
    JSON,
    BigInteger,
    Boolean,
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
from sqlalchemy.orm import relationship

from .base import Base


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


class Permission(Base):  # noqa
    __tablename__ = "permission"

    id = Column(
        BigInteger,
        primary_key=True,
        server_default=text("nextval('permission_id_seq'::regclass)"),
    )
    short_name = Column(Text, nullable=False, unique=True)
    long_name = Column(Text, nullable=False)


class Users(Base):  # noqa
    __tablename__ = "users"

    id = Column(
        BigInteger,
        primary_key=True,
        server_default=text("nextval('users_id_seq'::regclass)"),
    )
    name = Column(Text)
    email = Column(Text)
    emailVerified = Column(DateTime)
    image = Column(Text)
    role = Column(Text)


class Accounts(Base):  # noqa
    __tablename__ = "accounts"

    id = Column(
        BigInteger,
        primary_key=True,
        server_default=text("nextval('accounts_id_seq'::regclass)"),
    )
    userId = Column(ForeignKey("users.id"), nullable=False)
    type = Column(Text, nullable=False)
    provider = Column(Text, nullable=False)
    providerAccountId = Column(Text, nullable=False)
    refresh_token = Column(Text)
    access_token = Column(Text)
    expires_at = Column(BigInteger)
    id_token = Column(Text)
    scope = Column(Text)
    session_state = Column(Text)
    token_type = Column(Text)

    users = relationship("Users")


class Sessions(Base):  # noqa
    __tablename__ = "sessions"

    id = Column(
        BigInteger,
        primary_key=True,
        server_default=text("nextval('sessions_id_seq'::regclass)"),
    )
    userId = Column(ForeignKey("users.id"), nullable=False)
    expires = Column(DateTime, nullable=False)
    sessionToken = Column(Text, nullable=False)

    users = relationship("Users")


class VerificationToken(Base):  # noqa
    __tablename__ = "verification_token"

    identifier = Column(Text, primary_key=True, nullable=False)
    token = Column(Text, primary_key=True, nullable=False)
    expires = Column(DateTime, nullable=False)


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
    read_permission = relationship(
        "Permission", primaryjoin="Tag.read_perm == Permission.id"
    )
    write_permission = relationship(
        "Permission", primaryjoin="Tag.write_perm == Permission.id"
    )


class Cls(Base):  # noqa
    __tablename__ = "cls"

    id = Column(
        Integer,
        primary_key=True,
        server_default=text("nextval('cls_id_seq'::regclass)"),
    )
    short_name = Column(Text, unique=True)
    long_name = Column(Text)
    description = Column(Text)
    supercls = Column(ForeignKey("cls.id"))

    parent = relationship("Cls", remote_side=[id])


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
    pixel_f1 = Column(Float)
    instance_f1 = Column(Float)
    update_time = Column(DateTime, nullable=False, server_default=text("now()"))


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

    filter_rel = relationship("Filter")
    frequency_rel = relationship("Frequency")
    users = relationship("Users")
