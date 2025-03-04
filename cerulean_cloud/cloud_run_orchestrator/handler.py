"""Cloud run handler for orchestration of inference
1. Generated base and offset tiles from a scene id
2. Asyncronously place requests to fetch tiles (base and offset), and get inference result
3. Send result of inference to merge tiles cloud run once done


needs env vars:
- TITILER_URL
- INFERENCE_URL
"""

import json
import math
import os
import signal
import traceback
import urllib.parse as urlparse
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import centerline.geometry
import geopandas as gpd
import morecantile
import networkx as nx
import numpy as np
import supermercado
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from global_land_mask import globe
from shapely.geometry import LineString, shape

from cerulean_cloud.auth import api_key_auth
from cerulean_cloud.cloud_function_asa.queuer import add_to_asa_queue
from cerulean_cloud.cloud_run_orchestrator.clients import CloudRunInferenceClient
from cerulean_cloud.cloud_run_orchestrator.schema import (
    OrchestratorInput,
    OrchestratorResult,
)
from cerulean_cloud.database_client import DatabaseClient, get_engine
from cerulean_cloud.models import get_model
from cerulean_cloud.roda_sentinelhub_client import RodaSentinelHubClient
from cerulean_cloud.structured_logger import (
    configure_structured_logger,
    context_dict_var,
)
from cerulean_cloud.tiling import TMS, offset_bounds_from_base_tiles
from cerulean_cloud.titiler_client import TitilerClient

# Configure logger
logger = configure_structured_logger("cerulean_cloud")


app = FastAPI(title="Cloud Run orchestrator", dependencies=[Depends(api_key_auth)])
# Allow CORS for local debugging
app.add_middleware(CORSMiddleware, allow_origins=["*"])

landmask_gdf = None


def flush_logs():
    """
    Flush all logger handlers.
    """
    for handler in logger.handlers:
        handler.flush()


def handle_sigterm(signum, frame):
    """
    Handle the SIGTERM signal.
    """
    logger.error(
        {
            "message": "SIGTERM signal received",
            "file_name": frame.f_code.co_filename,
            "line_number": frame.f_lineno,
        }
    )
    flush_logs()


# Register SIGTERM handler
signal.signal(signal.SIGTERM, handle_sigterm)


def get_landmask_gdf():
    """
    Retrieves the GeoDataFrame representing the land mask.
    This function uses lazy initialization to load the land mask data from a .shp file
    only upon the first call. Subsequent calls return the stored GeoDataFrame.
    Returns:
        GeoDataFrame: The GeoDataFrame object representing the land mask, with CRS set to "EPSG:3857".
    """
    global landmask_gdf
    if landmask_gdf is None:
        mask_path = "/app/cerulean_cloud/cloud_run_orchestrator/gadmLandMask_simplified/gadmLandMask_simplified.shp"
        landmask_gdf = gpd.read_file(mask_path).set_crs("4326")
    return landmask_gdf


def make_cloud_log_url(
    cloud_run_name: str, start_time, project_id: str, duration=2
) -> str:
    """Forges a cloud log url given a service name, a date and a project id

    Args:
        cloud_run_name (str): the cloud run service name
        start_time (datetime.datetime): the start time of the cloud run
        project_id (str): a project id
        duration (int, optional): Default duration of the logs to show (in min). Defaults to 2.

    Returns:
        str: A cloud log url.
    """
    base_url = "https://console.cloud.google.com/logs/query;query="
    query = f'resource.type = "cloud_run_revision" resource.labels.service_name = "{cloud_run_name}"'
    formatted_time = start_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    end_time = (start_time + timedelta(minutes=duration)).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )
    t_range = f";timeRange={formatted_time}%2F{end_time}"
    t = f";cursorTimestamp={formatted_time}"
    project_id = f"?project={project_id}"
    url = base_url + urlparse.quote(query) + t_range + t + project_id
    return url


def offset_group_shape_from_base_tiles(
    tiles: List[morecantile.Tile], scale=2
) -> Tuple[int, int]:
    """from a list of tiles, get the expected shape of the image (of offset tiles, +1)"""
    tiles_np = np.array([(tile.x, tile.y, tile.z) for tile in tiles])
    tilexmin, tilexmax, tileymin, tileymax = supermercado.super_utils.get_range(
        tiles_np
    )
    hw = scale * 256
    width = (tilexmax - tilexmin + 2) * hw
    height = (tileymax - tileymin + 2) * hw

    return height, width


def group_bounds_from_list_of_bounds(bounds: List[List[float]]) -> List[float]:
    """from a list of bounds, get the merged bounds (min max)"""
    bounds_np = np.array([(b[0], b[1], b[2], b[3]) for b in bounds])
    minx, miny, maxx, maxy = (
        np.min(bounds_np[:, 0]),
        np.min(bounds_np[:, 1]),
        np.max(bounds_np[:, 2]),
        np.max(bounds_np[:, 3]),
    )
    return list((minx, miny, maxx, maxy))


def get_tiler():
    """get tiler"""
    return TMS


def get_titiler_client():
    """get titiler client"""
    return TitilerClient(url=os.getenv("TITILER_URL"))


def get_roda_sentinelhub_client():
    """get roda sentinelhub client"""
    return RodaSentinelHubClient()


def get_database_engine():
    """get database engine"""
    return get_engine(db_url=os.getenv("DB_URL"))


@app.get("/", description="Health Check", tags=["Health Check"])
def ping() -> Dict:
    """Health check."""
    return {"ping": "pong!"}


@app.post(
    "/orchestrate",
    description="Run orchestration",
    tags=["Run orchestration"],
    response_model=OrchestratorResult,
)
async def orchestrate(
    payload: OrchestratorInput,
    tiler=Depends(get_tiler),
    titiler_client=Depends(get_titiler_client),
    roda_sentinelhub_client=Depends(get_roda_sentinelhub_client),
    db_engine=Depends(get_database_engine),
) -> Dict:
    """orchestrate"""
    return await _orchestrate(
        payload, tiler, titiler_client, roda_sentinelhub_client, db_engine
    )


def is_tile_over_water(tile_bounds: List[float]) -> bool:
    """are the tile bounds over water"""
    minx, miny, maxx, maxy = tile_bounds
    return any(globe.is_ocean([miny, maxy], [minx, maxx]))


async def _orchestrate(
    payload, tiler, titiler_client, roda_sentinelhub_client, db_engine
):
    context_dict_var.set({"scene_id": payload.scene_id})  # Set the scene_id for logging

    # Orchestrate inference
    start_time = datetime.now()
    logger.info(
        {
            "message": "Initiating Orchestrator",
            "start_time": start_time.isoformat() + "Z",
        }
    )

    async with DatabaseClient(db_engine) as db_client:
        async with db_client.session.begin():
            db_model = await db_client.get_db_model(os.getenv("MODEL"))
            model_dict = {
                column.name: (
                    getattr(db_model, column.name).isoformat()
                    if isinstance(getattr(db_model, column.name), datetime)
                    else getattr(db_model, column.name)
                )
                for column in db_model.__table__.columns
            }
    zoom = payload.zoom or model_dict["zoom_level"]
    scale = payload.scale or model_dict["scale"]

    if model_dict["zoom_level"] != zoom:
        logger.warning(
            {
                "message": "Model zoom level warning",
                "description": f"Model was trained on zoom level {model_dict['zoom_level']} but is being run on {zoom}",
            }
        )
    if model_dict["tile_width_px"] != scale * 256:
        logger.warning(
            {
                "message": "Model resolution warning",
                "description": f"Model was trained on image tile of resolution {model_dict['tile_width_px']} but is being run on {scale * 256}",
            }
        )

    # WARNING: until this is resolved https://github.com/cogeotiff/rio-tiler-pds/issues/77
    # When scene traverses the anti-meridian, scene_bounds are nonsensical
    # Example: S1A_IW_GRDH_1SDV_20230726T183302_20230726T183327_049598_05F6CA_31E7 >>> [-180.0, 61.06949078480844, 180.0, 62.88226850489882]
    logger.info("Getting scene bounds")
    scene_bounds = await titiler_client.get_bounds(payload.scene_id)

    logger.info("Getting scene statistics")
    scene_stats = await titiler_client.get_statistics(payload.scene_id, band="vv")

    logger.info("Getting SentinalHUB product info")
    scene_info = await roda_sentinelhub_client.get_product_info(payload.scene_id)

    logger.info(
        {
            "message": "Generating tilesets",
            "zoom": zoom,
            "scale": scale,
            "scene_bounds": json.dumps(scene_bounds),
            "scene_stats": json.dumps(scene_stats),
            "scene_info": json.dumps(scene_info),
        }
    )

    base_tiles = list(tiler.tiles(*scene_bounds, [zoom], truncate=False))
    n_basetiles = len(base_tiles)

    offset_amounts = [0.0, 0.33, 0.66]
    tileset_list = [
        offset_bounds_from_base_tiles(base_tiles, offset_amount=o_a)
        for o_a in offset_amounts
    ]

    n_offsettiles = len(tileset_list[0])
    tileset_hw_pixels = offset_group_shape_from_base_tiles(base_tiles, scale=scale)
    tileset_envelope_bounds = group_bounds_from_list_of_bounds(tileset_list[0])

    logger.info(
        {
            "message": "Removing invalid tiles (land filter)",
            "n_tiles": n_offsettiles,
        }
    )

    # Filter out land tiles
    try:
        tileset_list = [
            [b for b in tileset if is_tile_over_water(b)] for tileset in tileset_list
        ]
        logger.info(
            {
                "message": "Removed tiles over land",
                "n_tilesets": len(tileset_list[0]),
            }
        )
    except ValueError as e:
        # XXX BUG is_tile_over_water throws ValueError if the scene crosses or is close to the antimeridian. Example: S1A_IW_GRDH_1SDV_20230726T183302_20230726T183327_049598_05F6CA_31E7

        logger.error(
            {
                "message": "FAILURE - scene touches antimeridian, and is_tile_over_water() failed!",
                "traceback": traceback.format_exc(),
            }
        )
        return OrchestratorResult(status=str(e))

    if not any(set for set in tileset_list):
        # There are actually no tiles to be processed! This is because the scene relevancy ocean mask is coarser than globe.is_ocean().
        # WARNING this will return success, but there will be not trace in the DB of your request (i.e. in S1 or Orchestrator tables)
        # XXX TODO
        logger.info(
            "IS_OCEAN() MISMATCH: NO TILES TO BE PROCESSED (SUCCESS BUT NO DB ENTRY)"
        )
        return OrchestratorResult(status="Success (no oceanic tiles)")

    if payload.dry_run:
        # Only tests code above this point, without actually adding any new data to the database, or running inference.
        logger.info("DRY RUN (SUCCESS)")
        return OrchestratorResult(status="Success (dry run)")

    # write to DB
    async with DatabaseClient(db_engine) as db_client:
        async with db_client.session.begin():
            trigger = await db_client.get_trigger(trigger=payload.trigger)
            layers = [
                await db_client.get_layer(layer) for layer in model_dict["layers"]
            ]
            sentinel1_grd = await db_client.get_or_insert_sentinel1_grd(
                payload.scene_id,
                scene_info,
                titiler_client.get_base_tile_url(
                    payload.scene_id,
                    rescale=(0, 255),
                ),
            )
            stale_slick_count = await db_client.deactivate_stale_slicks_from_scene_id(
                payload.scene_id
            )
            logger.info(
                {
                    "message": "Deactivating slicks from stale runs.",
                    "n_stale_slicks": stale_slick_count,
                }
            )
            orchestrator_run = await db_client.add_orchestrator(
                start_time,
                start_time,
                n_basetiles,
                n_offsettiles,
                os.getenv("GIT_HASH"),
                os.getenv("GIT_TAG"),
                make_cloud_log_url(
                    os.getenv("CLOUD_RUN_NAME"),
                    start_time,
                    os.getenv("PROJECT_ID"),
                ),
                zoom,
                scale,
                scene_bounds,
                trigger,
                db_model,
                sentinel1_grd,
            )

        success = True
        try:
            logger.info("Instantiating inference client")
            cloud_run_inference = CloudRunInferenceClient(
                url=os.getenv("INFERENCE_URL"),
                titiler_client=titiler_client,
                scene_id=payload.scene_id,
                tileset_envelope_bounds=tileset_envelope_bounds,
                image_hw_pixels=tileset_hw_pixels,
                layers=layers,
                scale=scale,
                model_dict=model_dict,
            )

            # Perform inferences
            logger.info("Inference starting")
            tileset_results_list = [
                await cloud_run_inference.run_parallel_inference(tileset)
                for tileset in tileset_list
            ]

            # Stitch inferences
            logger.info(
                {
                    "message": "Initializing model",
                    "model_type": model_dict["type"],
                }
            )
            model = get_model(model_dict)

            logger.info("Stitching result")
            tileset_fc_list = []
            for tileset_results, tileset_bounds in zip(
                tileset_results_list, tileset_list
            ):
                if tileset_results and tileset_bounds:
                    fc = model.postprocess_tileset(
                        tileset_results, [[b] for b in tileset_bounds]
                    )  # extra square brackets needed because each stack only has one tile in it for now XXX HACK
                    tileset_fc_list.append(fc)

            # Ensemble inferences
            logger.info("Ensembling results")
            final_ensemble = model.nms_feature_reduction(
                features=tileset_fc_list, min_overlaps_to_keep=1
            )
            features = final_ensemble.get("features", [])
            n_features = len(features)

            # number of features that are over land (excluded from final feature list)
            n_background_slicks = 0
            if features:
                LAND_MASK_BUFFER_M = 1000
                logger.info(
                    {
                        "message": "Removing all slicks near land",
                        "n_features": n_features,
                        "LAND_MASK_BUFFER_M": LAND_MASK_BUFFER_M,
                    }
                )
                for feat in features:
                    slick_gdf = gpd.GeoDataFrame(
                        geometry=[shape(feat["geometry"])], crs="4326"
                    )
                    crs_meters = slick_gdf.estimate_utm_crs(datum_name="WGS 84")
                    buffered_gdf = gpd.GeoDataFrame(
                        geometry=slick_gdf.to_crs(crs_meters)
                        .buffer(LAND_MASK_BUFFER_M)
                        .to_crs("4326")
                    )
                    intersecting_land = gpd.sjoin(
                        get_landmask_gdf(),
                        buffered_gdf,
                        how="inner",
                        predicate="intersects",
                    )
                    # when the feature intersects land, update inf_idx and increase n_background_slicks
                    if not intersecting_land.empty:
                        feat["properties"]["inf_idx"] = model.background_class_idx
                        n_background_slicks += 1

                    logger.info("Calculating centerlines")
                    centerlines_geojson, aspect_ratio_factor = calculate_centerlines(
                        slick_gdf, crs_meters
                    )
                    logger.info(
                        {
                            "message": "Centerlines calculated",
                            "aspect_ratio_factor": aspect_ratio_factor,
                            "centerlines_geojson": centerlines_geojson,
                        }
                    )
                    feat["properties"]["centerlines"] = centerlines_geojson
                    feat["properties"]["aspect_ratio_factor"] = aspect_ratio_factor

                logger.info(
                    {
                        "message": "Adding slicks to database",
                        "LAND_MASK_BUFFER_M": LAND_MASK_BUFFER_M,
                        "n_background_slicks": n_background_slicks,
                        "n_slicks": n_features - n_background_slicks,
                    }
                )
                del features
                # Removed all preprocessing of features from within the
                # database session to avoid holidng locks on the
                # table while performing un-related calculations.
                async with db_client.session.begin():
                    for feat in final_ensemble.get("features"):
                        _ = await db_client.add_slick(
                            orchestrator_run,
                            sentinel1_grd.start_time,
                            feat.get("geometry"),
                            feat.get("properties").get("inf_idx"),
                            feat.get("properties").get("machine_confidence"),
                            feat.get("properties").get("centerlines"),
                            feat.get("properties").get("aspect_ratio_factor"),
                        )
                        logger.info("Added slick")

                logger.info("Queueing up Automatic Source Association")
                add_to_asa_queue(sentinel1_grd.scene_id)

        except Exception as e:
            success = False
            exc = e
            logger.error(
                {
                    "message": "Failed to process inference on scene",
                    "exception": str(e),
                    "traceback": traceback.format_exc(),
                }
            )
        async with db_client.session.begin():
            end_time = datetime.now()
            orchestrator_run.success = success
            orchestrator_run.inference_end_time = end_time
            logger.info(
                {
                    "message": "Orchestration complete!",
                    "timestamp": end_time.isoformat() + "Z",
                    "success": success,
                    "duration_minutes": (end_time - start_time).total_seconds() / 60,
                }
            )
        if success is False:
            raise exc
    return OrchestratorResult(status="Success")


def compute_tree_diameter(graph):
    """
    Computes the longest path (diameter) of a weighted tree graph.

    Uses two passes of depth-first search (DFS):
        1. Start from an arbitrary node to find the farthest node (node_a).
        2. Start from node_a to find the farthest node from it (node_b).

    Returns:
        list: An ordered list of nodes (coordinates) representing the longest path.
    """

    def farthest_node(g, start):
        distances = {}
        stack = [(start, 0)]
        while stack:
            node, dist = stack.pop()
            if node not in distances:
                distances[node] = dist
                for neighbor, attr in g[node].items():
                    weight = attr.get("weight", 1)
                    stack.append((neighbor, dist + weight))
        # Pick the node with maximum distance from 'start'
        farthest = max(distances, key=distances.get)
        return farthest, distances[farthest]

    # Choose an arbitrary starting node
    start = next(iter(graph.nodes))
    node_a, _ = farthest_node(graph, start)
    node_b, _ = farthest_node(graph, node_a)

    # In a tree, the unique shortest path between node_a and node_b is the diameter.
    return nx.shortest_path(graph, source=node_a, target=node_b, weight="weight")


def find_longest_path(centerline_geom):
    """
    Extracts the longest continuous path from a centerline geometry.

    Parameters:
        centerline_geom (LineString or MultiLineString): The centerline geometry
            produced by centerline.geometry.Centerline(item), where item is a Polygon.

    Returns:
        LineString: A LineString representing the longest path (tree diameter).
    """

    def round_point(pt, precision=8):
        """Round a coordinate tuple to avoid floating point issues."""
        return tuple(round(coord, precision) for coord in pt)

    # Normalize input: if it's a single LineString, treat it as a list with one element.
    if centerline_geom.geom_type == "LineString":
        lines = [centerline_geom]
    elif centerline_geom.geom_type == "MultiLineString":
        lines = list(centerline_geom.geoms)
    else:
        raise ValueError(f"Unsupported geometry type: {centerline_geom.geom_type}")

    # Build a graph where nodes are coordinates and each edge connects consecutive points.
    graph = nx.Graph()
    for line in lines:
        coords = list(line.coords)
        for i in range(len(coords) - 1):
            u, v = round_point(coords[i]), round_point(coords[i + 1])
            # Use Euclidean distance as the edge weight.
            weight = ((v[0] - u[0]) ** 2 + (v[1] - u[1]) ** 2) ** 0.5
            graph.add_edge(u, v, weight=weight)

    longest_paths = []
    for component in list(nx.connected_components(graph)):
        # In case the bug in Centerline produces disconnected centerlines, we need to compute the diameter of each connected component
        # Compute the tree diameter (longest path) of the graph.
        longest_path_coords = compute_tree_diameter(graph.subgraph(component))
        longest_paths.append(LineString(longest_path_coords))
    return longest_paths


def calculate_centerlines(
    slick_gdf: gpd.GeoDataFrame,
    crs_meters: str,
    close_buffer: int = 2000,
    smoothing_factor: float = 1e10,
):
    """
    From a set of polygons representing oil slick detections, estimate centerlines that go through the detections
    This process transforms a set of slick detections into LineStrings for each detection

    Inputs:
        slick_gdf: GeoDataFrame of slick detections
        crs_meters: crs of slick center in meters
        close_buffer: buffer size for cleaning up slick detections
        smoothing_factor: smoothing factor for smoothing centerline
    Returns:
        GeoDataFrame of slick centerlines
    """
    # clean up the slick detections by dilation followed by erosion
    # this process can merge some polygons but not others, depending on proximity
    slick_closed = (
        slick_gdf.to_crs(crs_meters).buffer(close_buffer).buffer(-close_buffer)
    )

    # split slicks into individual polygons
    slick_closed = slick_closed.explode(ignore_index=True, index_parts=False)

    # find a centerline through detections
    slick_cls = list()
    for _, item in slick_closed.items():
        # create centerline -> MultiLineString
        polygon_perimeter = item.length  # Perimeter of the polygon
        interp_dist = min(
            100, polygon_perimeter / 1000
        )  # Use a minimum of 1000 points for voronoi calculation
        cl = centerline.geometry.Centerline(item, interpolation_distance=interp_dist)
        longest_paths = find_longest_path(cl.geometry)
        slick_cls.extend(longest_paths)

    slick_centerline_gdf = gpd.GeoDataFrame(geometry=slick_cls, crs=crs_meters).to_crs(
        "4326"
    )
    slick_centerline_gdf["area"] = slick_closed.geometry.area
    slick_centerline_gdf["length"] = [c.length for c in slick_cls]

    aspect_ratio_factor = compute_aspect_ratio_factor(slick_centerline_gdf, ar_ref=16)
    return json.loads(slick_centerline_gdf.to_json()), aspect_ratio_factor


def compute_aspect_ratio_factor(
    slick_centerlines: gpd.GeoDataFrame, ar_ref=16
) -> float:
    """
    Computes the aspect ratio factor for a given geometry.

    Parameters:
        slick_centerlines (gpd.GeoDataFrame): A GeoDataFrame containing line geometries with a 'length' and 'area' column.
        ar_ref (float, optional): Reference aspect ratio factor. Default is 16.

    Returns:
        float: The computed aspect ratio factor, between 0 and 1, where 0 is a square and 1 is an infinite line
    """

    L = slick_centerlines["length"].values
    A = slick_centerlines["area"].values

    # Centerline Length Weighted Bulk Effective Aspect Ratio
    clwbear = np.average(L**2 / A, weights=L)

    # Note slwbear is between 1 and infinity, so the following transformation moves it between 0 and 1
    arf = 1 - math.exp((1 - clwbear) / ar_ref)  # Aspect Ratio Factor
    return arf
