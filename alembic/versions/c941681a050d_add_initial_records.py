"""Add initial records

Revision ID: c941681a050d
Revises: 39277f6278f4
Create Date: 2022-07-06 12:49:46.037868

"""

import json
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision = "c941681a050d"
down_revision = "39277f6278f4"
branch_labels = None
depends_on = None


CLS_DELETE_GROUPS = [
    ["COIN_VESSEL"],
    ["OLD_VESSEL", "REC_VESSEL"],
    ["INFRA", "VESSEL"],
    ["BACKGROUND", "ANTHRO", "NATURAL", "AMBIGUOUS"],
]
MODEL_FILE_PATHS = [
    (
        "experiments/2023_10_05_02_22_46_4cls_rnxt101_pr512_px1024_680min_"
        "maskrcnn_wd01/scripting_cpu_model.pt"
    ),
    (
        "experiments/2024_09_04_21_34_24_4cls_resnet34_pr512_px1024_"
        "100epochs_unet/tracing_cpu_model.pt"
    ),
]
LAYER_SHORT_NAMES = ["VV", "INFRA", "VESSEL", "ALL_255", "ALL_ZEROS"]
AOI_TYPE_SHORT_NAMES = ["EEZ", "IHO", "MPA", "USER"]
SOURCE_TYPE_SHORT_NAMES = ["VESSEL", "INFRA", "DARK", "NATURAL"]
PERMISSION_SHORT_NAMES = ["own", "org", "any"]
TAG_SHORT_NAMES = ["fxo", "obs", "lng", "exc"]
FREQUENCY_SHORT_NAMES = ["REALTIME", "DAILY", "WEEKLY", "MONTHLY"]
DUMMY_USER_EMAIL = "dummy@dummy.dummy"


def _seed_table(table_name: str, columns: list[tuple[str, object]]):
    return sa.table(table_name, *(sa.column(name, type_) for name, type_ in columns))


def _offline_safe_rows(
    columns: list[tuple[str, object]], rows: list[dict]
) -> list[dict]:
    type_by_column = dict(columns)
    safe_rows = []

    for row in rows:
        safe_row = {}
        for key, value in row.items():
            if value is not None and isinstance(
                type_by_column[key], (sa.JSON, postgresql.JSONB)
            ):
                safe_row[key] = op.inline_literal(json.dumps(value))
            else:
                safe_row[key] = value
        safe_rows.append(safe_row)

    return safe_rows


def _bulk_insert(
    table_name: str, columns: list[tuple[str, object]], rows: list[dict]
) -> None:
    if op.get_context().as_sql:
        rows = _offline_safe_rows(columns, rows)

    op.bulk_insert(_seed_table(table_name, columns), rows, multiinsert=False)


def _table_with_column(table_name: str, column_name: str):
    return sa.table(table_name, sa.column(column_name))


def _delete_by_column(
    bind,
    table_name: str,
    column_name: str,
    values: list[str],
    reference_checks: list[tuple[str, str]] | None = None,
) -> None:
    table_ = sa.table(table_name, sa.column("id"), sa.column(column_name))
    statement = sa.delete(table_).where(table_.c[column_name].in_(values))

    for reference_table_name, reference_column_name in reference_checks or []:
        reference_table = _table_with_column(
            reference_table_name, reference_column_name
        )
        statement = statement.where(
            ~sa.exists().where(reference_table.c[reference_column_name] == table_.c.id)
        )

    bind.execute(statement)


def _delete_tags(bind) -> None:
    _delete_by_column(
        bind,
        "tag",
        "short_name",
        TAG_SHORT_NAMES,
        reference_checks=[("source_to_tag", "tag")],
    )


def _delete_users(bind) -> None:
    _delete_by_column(
        bind,
        "users",
        "email",
        [DUMMY_USER_EMAIL],
        reference_checks=[
            ("accounts", "userId"),
            ("sessions", "userId"),
            ("subscription", "user"),
            ("aoi_user", "user"),
            ("hitl_request", "user"),
            ("hitl_slick", "user"),
            ("tag", "owner"),
        ],
    )


def _delete_permissions(bind) -> None:
    permission = sa.table("permission", sa.column("id"), sa.column("short_name"))
    tag = sa.table("tag", sa.column("read_perm"), sa.column("write_perm"))
    bind.execute(
        sa.delete(permission).where(
            permission.c.short_name.in_(PERMISSION_SHORT_NAMES),
            ~sa.exists().where(
                sa.or_(
                    tag.c.read_perm == permission.c.id,
                    tag.c.write_perm == permission.c.id,
                )
            ),
        )
    )


def _delete_cls_group(bind, short_names: list[str]) -> None:
    cls = sa.table(
        "cls", sa.column("id"), sa.column("short_name"), sa.column("supercls")
    )
    cls_child = cls.alias("cls_child")
    slick = sa.table("slick", sa.column("cls"), sa.column("hitl_cls"))
    hitl_slick = _table_with_column("hitl_slick", "cls")

    bind.execute(
        sa.delete(cls).where(
            cls.c.short_name.in_(short_names),
            ~sa.exists().where(cls_child.c.supercls == cls.c.id),
            ~sa.exists().where(
                sa.or_(slick.c.cls == cls.c.id, slick.c.hitl_cls == cls.c.id)
            ),
            ~sa.exists().where(hitl_slick.c.cls == cls.c.id),
        )
    )


def _insert_tags() -> None:
    bind = op.get_bind()
    tag_insert = sa.text("""
        INSERT INTO tag (
            short_name,
            long_name,
            description,
            citation,
            owner,
            read_perm,
            write_perm,
            public,
            source_profile
        )
        SELECT
            :short_name,
            :long_name,
            :description,
            :citation,
            owner_user.id,
            read_permission.id,
            write_permission.id,
            :public,
            :source_profile
        FROM users AS owner_user
        CROSS JOIN permission AS read_permission
        CROSS JOIN permission AS write_permission
        WHERE owner_user.email = :owner_email
          AND read_permission.short_name = :read_perm
          AND write_permission.short_name = :write_perm
        """)

    for row in [
        {
            "short_name": "fxo",
            "long_name": "FxO",
            "description": "Vessels that have been identified as FPSOs or FSOs",
            "citation": "SkyTruth: fxo_masterlist_uncompressed_v1_20241029.csv",
            "read_perm": "any",
            "write_perm": "org",
            "public": True,
            "source_profile": True,
        },
        {
            "short_name": "obs",
            "long_name": "Obsolete",
            "description": "Sources that should be referenced by other records instead",
            "citation": None,
            "read_perm": "org",
            "write_perm": "org",
            "public": False,
            "source_profile": False,
        },
        {
            "short_name": "lng",
            "long_name": "LNG",
            "description": "Vessels that are suspected to be LNG carriers",
            "citation": "Global Fishing Watch",
            "read_perm": "any",
            "write_perm": "org",
            "public": True,
            "source_profile": True,
        },
        {
            "short_name": "exc",
            "long_name": "Excluded",
            "description": "Sources that should be excluded from ASA",
            "citation": "SkyTruth",
            "read_perm": "org",
            "write_perm": "org",
            "public": False,
            "source_profile": False,
        },
    ]:
        bind.execute(tag_insert, {"owner_email": DUMMY_USER_EMAIL, **row})


def _insert_cls_rows() -> None:
    bind = op.get_bind()
    cls_insert = sa.text("""
        INSERT INTO cls (short_name, long_name, supercls, description)
        VALUES (:short_name, :long_name, :supercls, :description)
        """)
    supercls_select = sa.text("SELECT id FROM cls WHERE short_name = :short_name")

    for row in [
        {
            "short_name": "BACKGROUND",
            "long_name": "Background",
            "description": (
                "Detections that are unlikely to be slicks (e.g. wind shadow, "
                "weather, ice, visual artifacts, over land, internal waves, etc.)"
            ),
        },
        {
            "short_name": "ANTHRO",
            "long_name": "Anthropogenic",
            "description": (
                "Detections that appear to be from anthropogenic sources, such as "
                "infrastructure, vessels"
            ),
        },
        {
            "short_name": "NATURAL",
            "long_name": "Natural",
            "description": (
                "Detections that appear to be from natural sources, such as oil seeps"
            ),
        },
        {
            "short_name": "INFRA",
            "long_name": "Infrastructure",
            "supercls_short_name": "ANTHRO",
            "description": (
                "Detections that appear to be from infrastructure, such as oil "
                "platforms or other man-made structures"
            ),
        },
        {
            "short_name": "VESSEL",
            "long_name": "Vessel",
            "supercls_short_name": "ANTHRO",
            "description": (
                "Detections that appear to be from vessels, such as ships, boats, "
                "or other watercraft"
            ),
        },
        {
            "short_name": "OLD_VESSEL",
            "long_name": "Vessel, old",
            "supercls_short_name": "VESSEL",
            "description": (
                "Detections that appear to be from vessels, but are old, so the slick "
                "is difficult to identify and the responsible party is very unlikely "
                "to be determined"
            ),
        },
        {
            "short_name": "REC_VESSEL",
            "long_name": "Vessel, recent",
            "supercls_short_name": "VESSEL",
            "description": (
                "Detections that appear to be from vessels, recent but not visible "
                "in the imagery, it may be possible to determine the responsible "
                "party, but it is unlikely"
            ),
        },
        {
            "short_name": "COIN_VESSEL",
            "long_name": "Vessel, coincident",
            "supercls_short_name": "REC_VESSEL",
            "description": (
                "Detections that appear to be from vessels, and are coincident with "
                "a vessel that is visible in the imagery, so the responsible party "
                "is highly likely to be determined"
            ),
        },
        {
            "short_name": "AMBIGUOUS",
            "long_name": "Ambiguous",
            "description": (
                "Detections that are ambiguous, it is not clear after human review "
                "if the detection is oil or some other slick (e.g. wind shadow, "
                "precipitation, etc.)"
            ),
        },
    ]:
        supercls_short_name = row.pop("supercls_short_name", None)
        supercls = None
        if supercls_short_name is not None:
            supercls = bind.execute(
                supercls_select, {"short_name": supercls_short_name}
            ).scalar_one()

        bind.execute(cls_insert, {**row, "supercls": supercls})


def upgrade() -> None:
    """add initial rows"""
    # EditTheDatabase
    _insert_cls_rows()

    _bulk_insert(
        "model",
        [
            ("type", sa.Text()),
            ("file_path", sa.Text()),
            ("layers", postgresql.ARRAY(sa.Text())),
            ("cls_map", sa.JSON()),
            ("name", sa.Text()),
            ("tile_width_m", sa.Integer()),
            ("tile_width_px", sa.Integer()),
            ("epochs", sa.Integer()),
            ("thresholds", sa.JSON()),
            ("backbone_size", sa.Integer()),
            ("pixel_f1", sa.Float()),
            ("instance_f1", sa.Float()),
        ],
        [
            {
                "type": "MASKRCNN",
                "file_path": MODEL_FILE_PATHS[0],
                "layers": ["VV", "ALL_255", "VESSEL"],
                "cls_map": {
                    0: "BACKGROUND",
                    1: "INFRA",
                    2: "NATURAL",
                    3: "VESSEL",
                },
                "name": "ResNext 101 hires56",
                "tile_width_m": 40844,
                "tile_width_px": 512,
                "epochs": 122,
                "thresholds": {
                    "poly_nms_thresh": 0.2,
                    "pixel_nms_thresh": 0.4,
                    "bbox_score_thresh": 0.3,
                    "poly_score_thresh": 0.1,
                    "pixel_score_thresh": 0.5,
                    "groundtruth_dice_thresh": 0.0,
                },
                "backbone_size": 101,
                "pixel_f1": 0.461,
                "instance_f1": 0.47,
            },
            {
                "type": "FASTAIUNET",
                "file_path": MODEL_FILE_PATHS[1],
                "layers": ["VV"],
                "cls_map": {
                    0: "BACKGROUND",
                    1: "INFRA",
                    2: "NATURAL",
                    3: "VESSEL",
                },
                "name": "ResNet34 46.6%",
                "tile_width_m": 40844,
                "tile_width_px": 512,
                "epochs": 500,
                "thresholds": {
                    "poly_nms_thresh": 0.2,
                    "pixel_nms_thresh": 0.0,
                    "bbox_score_thresh": 0.0001,
                    "poly_score_thresh": 0.5,
                    "pixel_score_thresh": 0.9,
                    "groundtruth_dice_thresh": 0.0,
                },
                "backbone_size": 34,
                "pixel_f1": 0.532,
            },
        ],
    )

    _bulk_insert(
        "layer",
        [
            ("short_name", sa.Text()),
            ("long_name", sa.Text()),
            ("citation", sa.Text()),
            ("source_url", sa.Text()),
            ("notes", sa.Text()),
        ],
        [
            {
                "short_name": "VV",
                "long_name": "S1 VV",
                "citation": (
                    "Copernicus Sentinel data, processed by ESA, accessed via AWS "
                    "Open Data Registry."
                ),
                "source_url": "https://registry.opendata.aws/sentinel-1/",
            },
            {
                "short_name": "INFRA",
                "long_name": "Infrastructure Distance",
                "citation": (
                    "Generated by SkyTruth, using GFW's Infrastructure Dataset "
                    "(pre-release)"
                ),
                "source_url": (
                    "https://storage.googleapis.com/ceruleanml/aux_datasets/"
                    "infra_locations_01_cogeo.tiff"
                ),
            },
            {
                "short_name": "VESSEL",
                "long_name": "Vessel Density",
                "citation": (
                    "Global Maritime Traffic Density Service (GTMDS) retrieved from "
                    "GlobalMaritimeTraffic.org, a service of MapLarge 2021"
                ),
                "source_url": "https://gmtds.maplarge.com/public/ext/GMTDS/Main",
                "notes": (
                    "Typically uses the previous month's density map. If unavailable "
                    "will default to previous year."
                ),
            },
            {
                "short_name": "ALL_255",
                "long_name": "All Pixels Value=255",
                "citation": "",
                "source_url": "",
                "notes": "Can be used for ablation or to replace unwanted layers.",
            },
            {
                "short_name": "ALL_ZEROS",
                "long_name": "All Pixels Value=0",
                "citation": "",
                "source_url": "",
                "notes": "Can be used for ablation or to replace unwanted layers.",
            },
        ],
    )

    _bulk_insert(
        "aoi_type",
        [
            ("table_name", sa.Text()),
            ("long_name", sa.Text()),
            ("short_name", sa.Text()),
            ("source_url", sa.Text()),
            ("citation", sa.Text()),
            ("update_time", sa.DateTime()),
        ],
        [
            {
                "table_name": "aoi_eez",
                "long_name": "Exclusive Economic Zone",
                "short_name": "EEZ",
                "source_url": "https://www.marineregions.org/eez.php",
                "citation": (
                    "Flanders Marine Institute (2019). Maritime Boundaries "
                    "Geodatabase, version 11. Available online at "
                    "https://www.marineregions.org/. https://doi.org/10.14284/382."
                ),
                "update_time": datetime.now(),
            },
            {
                "table_name": "aoi_iho",
                "long_name": "IHO Sea Areas",
                "short_name": "IHO",
                "source_url": "https://www.marineregions.org/sources.php#iho",
                "citation": (
                    "Flanders Marine Institute (2018). IHO Sea Areas, version 3. "
                    "Available online at https://www.marineregions.org/. "
                    "https://doi.org/10.14284/323."
                ),
                "update_time": datetime.now(),
            },
            {
                "table_name": "aoi_mpa",
                "long_name": "Marine Protected Area",
                "short_name": "MPA",
                "source_url": (
                    "https://www.protectedplanet.net/en/thematic-areas/"
                    "marine-protected-areas"
                ),
                "citation": (
                    "UNEP-WCMC and IUCN (2023), Protected Planet: The World "
                    "Database on Protected Areas (WDPA) and World Database on Other "
                    "Effective Area-based Conservation Measures (WD-OECM) [Online], "
                    "July 2023, Cambridge, UK: UNEP-WCMC and IUCN. Available at: "
                    "www.protectedplanet.net."
                ),
                "update_time": datetime.now(),
            },
            {
                "table_name": "aoi_user",
                "long_name": "User-generated",
                "short_name": "USER",
                "update_time": datetime.now(),
            },
        ],
    )

    _bulk_insert(
        "source_type",
        [
            ("table_name", sa.Text()),
            ("long_name", sa.Text()),
            ("short_name", sa.Text()),
            ("ext_id_name", sa.Text()),
            ("citation", sa.Text()),
        ],
        [
            {
                "table_name": "source_vessel",
                "long_name": "Vessel Source",
                "short_name": "VESSEL",
                "ext_id_name": "mmsi",
                "citation": "AIS from GFW",
            },
            {
                "table_name": "source_infra",
                "long_name": "Infrastructure Source",
                "short_name": "INFRA",
                "ext_id_name": "structure_id",
                "citation": "S1 Detections from GFW",
            },
            {
                "table_name": "source_dark",
                "long_name": "Dark Vessel Source",
                "short_name": "DARK",
                "ext_id_name": "dark_id",
                "citation": "S1 Detections from GFW",
            },
            {
                "table_name": "source_natural",
                "long_name": "Natural Seep Source",
                "short_name": "NATURAL",
                "ext_id_name": "seep_id",
                "citation": "SkyTruth",
            },
        ],
    )

    _bulk_insert(
        "permission",
        [("short_name", sa.Text()), ("long_name", sa.Text())],
        [
            {"short_name": "own", "long_name": "Owner Only"},
            {"short_name": "org", "long_name": "Organization Only"},
            {"short_name": "any", "long_name": "Any User"},
        ],
    )

    _bulk_insert(
        "users",
        [
            ("firstName", sa.Text()),
            ("lastName", sa.Text()),
            ("email", sa.Text()),
            ("role", sa.Text()),
            ("organization", sa.Text()),
            ("organizationType", postgresql.JSONB()),
            ("location", sa.Text()),
        ],
        [
            {
                "firstName": "dummy",
                "lastName": "dummy",
                "email": DUMMY_USER_EMAIL,
                "role": "dummy",
                "organization": "dummy",
                "organizationType": ["dummy"],
                "location": "dummy",
            }
        ],
    )

    _insert_tags()

    _bulk_insert(
        "frequency",
        [("short_name", sa.Text()), ("long_name", sa.Text())],
        [
            {"short_name": "REALTIME", "long_name": "Near real-time alerts"},
            {"short_name": "DAILY", "long_name": "Daily digest"},
            {"short_name": "WEEKLY", "long_name": "Weekly digest"},
            {"short_name": "MONTHLY", "long_name": "Monthly digest"},
        ],
    )


def downgrade() -> None:
    """drop initial rows"""
    bind = op.get_bind()

    _delete_by_column(
        bind,
        "model",
        "file_path",
        MODEL_FILE_PATHS,
        reference_checks=[("orchestrator_run", "model")],
    )
    _delete_tags(bind)
    _delete_by_column(bind, "layer", "short_name", LAYER_SHORT_NAMES)
    _delete_by_column(
        bind,
        "aoi_type",
        "short_name",
        AOI_TYPE_SHORT_NAMES,
        reference_checks=[("aoi", "type")],
    )
    _delete_by_column(
        bind,
        "source_type",
        "short_name",
        SOURCE_TYPE_SHORT_NAMES,
        reference_checks=[("source", "type")],
    )
    _delete_by_column(
        bind,
        "frequency",
        "short_name",
        FREQUENCY_SHORT_NAMES,
        reference_checks=[("subscription", "frequency")],
    )
    _delete_permissions(bind)
    _delete_users(bind)

    for short_names in CLS_DELETE_GROUPS:
        _delete_cls_group(bind, short_names)
