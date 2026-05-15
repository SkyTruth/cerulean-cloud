"""Add initial records

Revision ID: c941681a050d
Revises: 39277f6278f4
Create Date: 2022-07-06 12:49:46.037868

"""

from datetime import datetime

from sqlalchemy import column, delete, exists, or_, orm, table

import cerulean_cloud.database_schema as database_schema
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


def _table_with_column(table_name: str, column_name: str):
    return table(table_name, column(column_name))


def _delete_by_column(
    bind,
    table_name: str,
    column_name: str,
    values: list[str],
    reference_checks: list[tuple[str, str]] | None = None,
) -> None:
    table_ = table(table_name, column("id"), column(column_name))
    statement = delete(table_).where(table_.c[column_name].in_(values))

    for reference_table_name, reference_column_name in reference_checks or []:
        reference_table = _table_with_column(
            reference_table_name, reference_column_name
        )
        statement = statement.where(
            ~exists().where(reference_table.c[reference_column_name] == table_.c.id)
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
    permission = table("permission", column("id"), column("short_name"))
    tag = table("tag", column("read_perm"), column("write_perm"))
    bind.execute(
        delete(permission).where(
            permission.c.short_name.in_(PERMISSION_SHORT_NAMES),
            ~exists().where(
                or_(
                    tag.c.read_perm == permission.c.id,
                    tag.c.write_perm == permission.c.id,
                )
            ),
        )
    )


def _delete_cls_group(bind, short_names: list[str]) -> None:
    cls = table("cls", column("id"), column("short_name"), column("supercls"))
    cls_child = cls.alias("cls_child")
    slick = table("slick", column("cls"), column("hitl_cls"))
    hitl_slick = _table_with_column("hitl_slick", "cls")

    bind.execute(
        delete(cls).where(
            cls.c.short_name.in_(short_names),
            ~exists().where(cls_child.c.supercls == cls.c.id),
            ~exists().where(or_(slick.c.cls == cls.c.id, slick.c.hitl_cls == cls.c.id)),
            ~exists().where(hitl_slick.c.cls == cls.c.id),
        )
    )


def upgrade() -> None:
    """add initial rows"""
    bind = op.get_bind()
    session = orm.Session(bind=bind)

    # EditTheDatabase
    with session.begin():
        clses = [
            database_schema.Cls(
                short_name="BACKGROUND",
                long_name="Background",
                description="Detections that are unlikely to be slicks (e.g. wind shadow, weather, ice, visual artifacts, over land, internal waves, etc.)",
            ),
            database_schema.Cls(
                short_name="ANTHRO",
                long_name="Anthropogenic",
                description="Detections that appear to be from anthropogenic sources, such as infrastructure, vessels",
            ),
            database_schema.Cls(
                short_name="NATURAL",
                long_name="Natural",
                description="Detections that appear to be from natural sources, such as oil seeps",
            ),
            database_schema.Cls(
                short_name="INFRA",
                long_name="Infrastructure",
                supercls=2,
                description="Detections that appear to be from infrastructure, such as oil platforms or other man-made structures",
            ),
            database_schema.Cls(
                short_name="VESSEL",
                long_name="Vessel",
                supercls=2,
                description="Detections that appear to be from vessels, such as ships, boats, or other watercraft",
            ),
            database_schema.Cls(
                short_name="OLD_VESSEL",
                long_name="Vessel, old",
                supercls=5,
                description="Detections that appear to be from vessels, but are old, so the slick is difficult to identify and the responsible party is very unlikely to be determined",
            ),
            database_schema.Cls(
                short_name="REC_VESSEL",
                long_name="Vessel, recent",
                supercls=5,
                description="Detections that appear to be from vessels, recent but not visible in the imagery, it may be possible to determine the responsible party, but it is unlikely",
            ),
            database_schema.Cls(
                short_name="COIN_VESSEL",
                long_name="Vessel, coincident",
                supercls=7,
                description="Detections that appear to be from vessels, and are coincident with a vessel that is visible in the imagery, so the responsible party is highly likely to be determined",
            ),
            database_schema.Cls(
                short_name="AMBIGUOUS",
                long_name="Ambiguous",
                description="Detections that are ambiguous, it is not clear after human review if the detection is oil or some other slick (e.g. wind shadow, precipitation, etc.)",
            ),
        ]
        session.add_all(clses)

        models = [
            database_schema.Model(
                type="MASKRCNN",
                file_path="experiments/2023_10_05_02_22_46_4cls_rnxt101_pr512_px1024_680min_maskrcnn_wd01/scripting_cpu_model.pt",
                layers=["VV", "ALL_255", "VESSEL"],
                cls_map={
                    0: "BACKGROUND",
                    1: "INFRA",
                    2: "NATURAL",
                    3: "VESSEL",
                },  # inference_idx maps to class table
                name="ResNext 101 hires56",
                tile_width_m=40844,
                tile_width_px=512,
                epochs=122,
                thresholds={
                    "poly_nms_thresh": 0.2,
                    "pixel_nms_thresh": 0.4,
                    "bbox_score_thresh": 0.3,
                    "poly_score_thresh": 0.1,
                    "pixel_score_thresh": 0.5,
                    "groundtruth_dice_thresh": 0.0,
                },
                backbone_size=101,
                pixel_f1=0.461,
                instance_f1=0.47,
            ),
            database_schema.Model(
                type="FASTAIUNET",
                file_path="experiments/2024_09_04_21_34_24_4cls_resnet34_pr512_px1024_100epochs_unet/tracing_cpu_model.pt",
                layers=["VV"],
                cls_map={
                    0: "BACKGROUND",
                    1: "INFRA",
                    2: "NATURAL",
                    3: "VESSEL",
                },  # inference_idx maps to class table
                name="ResNet34 46.6%",
                tile_width_m=40844,  # Used to calculate zoom
                tile_width_px=512,  # Used to calculate scale
                epochs=500,
                thresholds={
                    "poly_nms_thresh": 0.2,  # Minimum IoU between instances that will keep the higher scoring multipolygon
                    "pixel_nms_thresh": 0.0,  # NOT USED IN UNETS
                    "bbox_score_thresh": 0.0001,  # Smallest bridge value that will connect polygons into a multipolygon
                    "poly_score_thresh": 0.5,  # Determines the size of the outline of any given polygon
                    "pixel_score_thresh": 0.9,  # Minimum pixel score that will be required to keep a multipolygon
                    "groundtruth_dice_thresh": 0.0,
                },
                backbone_size=34,
                pixel_f1=0.532,
                # instance_f1=0.0, # TODO CALCULATE
            ),
        ]
        session.add_all(models)

        layers = [
            database_schema.Layer(
                short_name="VV",
                long_name="S1 VV",
                citation="Copernicus Sentinel data, processed by ESA, accessed via AWS Open Data Registry.",
                source_url="https://registry.opendata.aws/sentinel-1/",
            ),
            database_schema.Layer(
                short_name="INFRA",  # TODO Rename to something like INFRA_LAY, to avoid conflict with INFRA_PIXEL_CLASS
                long_name="Infrastructure Distance",
                citation="Generated by SkyTruth, using GFW's Infrastructure Dataset (pre-release)",
                source_url="https://storage.googleapis.com/ceruleanml/aux_datasets/infra_locations_01_cogeo.tiff",
            ),
            database_schema.Layer(
                short_name="VESSEL",  # TODO Rename to something like VESSEL_LAY, to avoid conflict with VESSEL_PIXEL_CLASS
                long_name="Vessel Density",
                citation="Global Maritime Traffic Density Service (GTMDS) retrieved from GlobalMaritimeTraffic.org, a service of MapLarge 2021",
                source_url="https://gmtds.maplarge.com/public/ext/GMTDS/Main",
                notes="Typically uses the previous month's density map. If unavailable will default to previous year.",
            ),
            database_schema.Layer(
                short_name="ALL_255",  # TODO Rename to something liketo avoid conflict with PIXEL CLASS
                long_name="All Pixels Value=255",
                citation="",
                source_url="",
                notes="Can be used for ablation or to replace unwanted layers.",
            ),
            database_schema.Layer(
                short_name="ALL_ZEROS",  # TODO Rename to something liketo avoid conflict with PIXEL CLASS
                long_name="All Pixels Value=0",
                citation="",
                source_url="",
                notes="Can be used for ablation or to replace unwanted layers.",
            ),
        ]
        session.add_all(layers)

        aoi_types = [
            database_schema.AoiType(
                table_name="aoi_eez",
                long_name="Exclusive Economic Zone",
                short_name="EEZ",
                source_url="https://www.marineregions.org/eez.php",
                citation="Flanders Marine Institute (2019). Maritime Boundaries Geodatabase, version 11. Available online at https://www.marineregions.org/. https://doi.org/10.14284/382.",
                update_time=datetime.now(),
            ),
            database_schema.AoiType(
                table_name="aoi_iho",
                long_name="IHO Sea Areas",
                short_name="IHO",
                source_url="https://www.marineregions.org/sources.php#iho",
                citation="Flanders Marine Institute (2018). IHO Sea Areas, version 3. Available online at https://www.marineregions.org/. https://doi.org/10.14284/323.",
                update_time=datetime.now(),
            ),
            database_schema.AoiType(
                table_name="aoi_mpa",
                long_name="Marine Protected Area",
                short_name="MPA",
                source_url="https://www.protectedplanet.net/en/thematic-areas/marine-protected-areas",
                citation="UNEP-WCMC and IUCN (2023), Protected Planet: The World Database on Protected Areas (WDPA) and World Database on Other Effective Area-based Conservation Measures (WD-OECM) [Online], July 2023, Cambridge, UK: UNEP-WCMC and IUCN. Available at: www.protectedplanet.net.",
                update_time=datetime.now(),
            ),
            database_schema.AoiType(
                table_name="aoi_user",
                long_name="User-generated",
                short_name="USER",
                update_time=datetime.now(),
            ),
        ]
        session.add_all(aoi_types)

        source_types = [
            database_schema.SourceType(
                table_name="source_vessel",
                long_name="Vessel Source",
                short_name="VESSEL",
                ext_id_name="mmsi",
                citation="AIS from GFW",
            ),
            database_schema.SourceType(
                table_name="source_infra",
                long_name="Infrastructure Source",
                short_name="INFRA",
                ext_id_name="structure_id",
                citation="S1 Detections from GFW",
            ),
            database_schema.SourceType(
                table_name="source_dark",
                long_name="Dark Vessel Source",
                short_name="DARK",
                ext_id_name="dark_id",
                citation="S1 Detections from GFW",
            ),
            database_schema.SourceType(
                table_name="source_natural",
                long_name="Natural Seep Source",
                short_name="NATURAL",
                ext_id_name="seep_id",
                citation="SkyTruth",
            ),
        ]
        session.add_all(source_types)

        permissions = [
            database_schema.Permission(
                short_name="own",
                long_name="Owner Only",
            ),
            database_schema.Permission(
                short_name="org",
                long_name="Organization Only",
            ),
            database_schema.Permission(
                short_name="any",
                long_name="Any User",
            ),
        ]
        session.add_all(permissions)

        first_user = database_schema.Users(
            firstName="dummy",
            lastName="dummy",
            email="dummy@dummy.dummy",
            role="dummy",
            organization="dummy",
            organizationType=["dummy"],
            location="dummy",
        )
        session.add(first_user)
        session.flush()  # guarantees system_user.id is available

        tags = [
            database_schema.Tag(
                short_name="fxo",
                long_name="FxO",
                description="Vessels that have been identified as FPSOs or FSOs",
                citation="SkyTruth: fxo_masterlist_uncompressed_v1_20241029.csv",
                owner=1,
                read_perm=3,
                write_perm=2,
                public=True,
                source_profile=True,
            ),
            database_schema.Tag(
                short_name="obs",
                long_name="Obsolete",
                description="Sources that should be referenced by other records instead",
                owner=1,
                read_perm=2,
                write_perm=2,
                public=False,
                source_profile=False,
            ),
            database_schema.Tag(
                short_name="lng",
                long_name="LNG",
                description="Vessels that are suspected to be LNG carriers",
                citation="Global Fishing Watch",
                owner=1,
                read_perm=3,
                write_perm=2,
                public=True,
                source_profile=True,
            ),
            database_schema.Tag(
                short_name="exc",
                long_name="Excluded",
                description="Sources that should be excluded from ASA",
                citation="SkyTruth",
                owner=1,
                read_perm=2,
                write_perm=2,
                public=False,
                source_profile=False,
            ),
        ]
        session.add_all(tags)

        frequencies = [
            database_schema.Frequency(
                short_name="REALTIME",
                long_name="Near real-time alerts",
            ),
            database_schema.Frequency(
                short_name="DAILY",
                long_name="Daily digest",
            ),
            database_schema.Frequency(
                short_name="WEEKLY",
                long_name="Weekly digest",
            ),
            database_schema.Frequency(
                short_name="MONTHLY",
                long_name="Monthly digest",
            ),
        ]
        session.add_all(frequencies)


def downgrade() -> None:
    """drop initial rows"""
    bind = op.get_bind()

    # Use lightweight table expressions instead of the live ORM mappings. This
    # historical migration must run after later schema changes have been undone.
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
