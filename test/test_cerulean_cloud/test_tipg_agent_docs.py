"""Agent guidance is discoverable without altering feature downloads."""

import ast
import hashlib
import io
import json
import subprocess
import tarfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from starlette.responses import StreamingResponse

from cerulean_cloud.cloud_run_tipg.agent_docs import (
    ARCHIVE_URL,
    INDEX_URL,
    SKILL_ROOT,
    SKILL_URL,
    AgentDiscoveryMiddleware,
)


@pytest.mark.asyncio
async def test_published_skill_and_install_bundle():
    from cerulean_cloud.cloud_run_tipg.handler import app

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://test"
    ) as client:
        guide = await client.get("/llms.txt")
        assert guide.status_code == 200
        assert len(guide.content) < len((SKILL_ROOT / "SKILL.md").read_bytes())
        assert 'rel="describedby"' in guide.headers["link"]
        for path in SKILL_ROOT.rglob("*.md"):
            response = await client.get(
                f"/skills/cerulean-investigation/{path.relative_to(SKILL_ROOT).as_posix()}"
            )
            assert response.content == path.read_bytes()
            assert response.headers["content-type"].startswith("text/markdown")
        index = (await client.get(INDEX_URL)).json()
        skill = index["skills"][0]
        archive = await client.get(skill["url"])
        assert skill["url"] == ARCHIVE_URL
        assert (
            skill["digest"] == "sha256:" + hashlib.sha256(archive.content).hexdigest()
        )
        with tarfile.open(fileobj=io.BytesIO(archive.content), mode="r:gz") as bundle:
            assert set(bundle.getnames()) == {
                path.relative_to(SKILL_ROOT).as_posix()
                for path in SKILL_ROOT.rglob("*.md")
            }
            for member in bundle.getmembers():
                relative = Path(member.name)
                assert (
                    bundle.extractfile(member).read()
                    == (SKILL_ROOT / relative).read_bytes()
                )
        skill_response = await client.get(SKILL_URL)
        head = await client.head(SKILL_URL)
        assert head.status_code == 200
        assert not head.content
        assert head.headers["etag"] == skill_response.headers["etag"]
        cached = await client.get(
            SKILL_URL, headers={"If-None-Match": skill_response.headers["etag"]}
        )
        assert cached.status_code == 304
        assert not cached.content
        assert (
            await client.get(
                "/skills/cerulean-investigation-maintenance/check_contract.py"
            )
        ).status_code == 404
        assert (
            await client.get("/skills/cerulean-investigation/%2e%2e/%2e%2e/robots.txt")
        ).status_code == 404
        schema = await client.get("/api")
        assert schema.status_code == 200
        assert schema.json()["externalDocs"]["url"].endswith("/llms.txt")
        assert (await client.get("/robots.txt")).status_code == 200


@pytest.mark.asyncio
async def test_metadata_links_and_feature_stream_passthrough():
    app = FastAPI()
    app.add_middleware(AgentDiscoveryMiddleware)
    sent_bodies = []

    @app.get("/collections")
    def collections():
        return {
            "links": [{"rel": "self", "href": "/collections"}],
            "collections": [{"id": "public.slick_plus", "links": []}],
        }

    first = b'{"type":"FeatureCollection","features":['
    last = b'],"links":[]}'

    async def features():
        yield first
        assert sent_bodies == [first], (
            "Feature downloads must not wait for the entire body"
        )
        yield last

    @app.get("/collections/public.slick_plus/items")
    def items():
        return StreamingResponse(features(), media_type="application/geo+json")

    async def capture(scope, receive, send):
        async def record(message):
            if message["type"] == "http.response.body" and message.get("body"):
                sent_bodies.append(message["body"])
            await send(message)

        await app(scope, receive, record)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=capture), base_url="https://test"
    ) as client:
        metadata = await client.get("/collections")
        data = metadata.json()
        assert data["links"][0]["rel"] == "self"
        assert data["links"][-1]["href"] == "https://test/llms.txt"
        assert data["collections"][0]["links"][-1]["rel"] == "describedby"
        assert int(metadata.headers["content-length"]) == len(metadata.content)
        sent_bodies.clear()
        response = await client.get("/collections/public.slick_plus/items")
        assert response.content == first + last
        assert 'rel="describedby"' in response.headers["link"]
        assert (await client.get("/missing")).headers.get("link")


@pytest.fixture
def notebook_helpers(tmp_path):
    root = Path(__file__).resolve().parents[2]
    notebook = json.loads((root / "notebooks" / "Cerulean API Guide.ipynb").read_text())
    namespace = {
        "json": json,
        "Path": Path,
        "defaultdict": defaultdict,
        "datetime": datetime,
        "timezone": timezone,
        "OUTPUT_DIR": tmp_path,
        "API": "https://api.cerulean.skytruth.org/collections",
        "UTM": {"utm_medium": "agentic", "utm_source": "test"},
        "request_log": [],
        "subprocess": subprocess,
        "shlex": __import__("shlex"),
        "time": __import__("time"),
    }
    definitions = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if source.startswith("%pip"):
            continue
        tree = ast.parse(source)
        definitions.extend(
            node for node in tree.body if isinstance(node, ast.FunctionDef)
        )
    exec(
        compile(ast.Module(body=definitions, type_ignores=[]), "API notebook", "exec"),
        namespace,
    )
    return namespace


@pytest.mark.parametrize(
    "review,types,status,label,group",
    [
        (None, [("DARK", 0.6), ("DARK", 0.2)], "run", "Dark", "dark_only"),
        (None, [("DARK", 0.6), ("VESSEL", 0.1)], "run", "Uncertain", "dark_mixed"),
        (None, [("DARK", 0.6), ("INFRA", 0)], "run", "Dark", "dark_only"),
        (None, [("INFRA", 0.4)], "pending", "Infrastructure (Pending)", None),
        (None, [], "pending", "Pending", None),
        (None, [], "unavailable", "Unknown", None),
        (4, [("DARK", 0.8)], "run", "Infrastructure", "conflicting_human_origin"),
        (8, [("DARK", 0.8)], "run", "Vessel", "reviewed_vessel_dark_only"),
    ],
)
def test_notebook_origins_follow_evidence_hierarchy(
    notebook_helpers, review, types, status, label, group
):
    slicks = [{"properties": {"id": 1, "hitl_cls": review, "hitl_cls_name": None}}]
    sources = [
        {
            "properties": {
                "slick_id": 1,
                "source_type": kind,
                "source_collated_score": score,
            }
        }
        for kind, score in types
    ]
    rows = notebook_helpers["possible_origins"](slicks, sources, {1: status})
    assert rows[0]["possible_origin"] == label
    assert rows[0]["dark_group"] == group


def test_notebook_fetches_every_page_and_rejects_errors(notebook_helpers, monkeypatch):
    calls = []

    def curl(command, **kwargs):
        calls.append(command)
        offset = next(value for value in command if value.startswith("offset="))
        value = 1 if offset == "offset=0" else 2
        Path(command[command.index("-o") + 1]).write_text(
            json.dumps(
                {
                    "numberMatched": 2,
                    "features": [{"properties": {"id": value}}],
                }
            )
        )
        return SimpleNamespace(stdout="200")

    monkeypatch.setattr(subprocess, "run", curl)
    data = notebook_helpers["fetch_features"](
        "public.slick_plus", {"bbox": "11,36,20,43"}, "pages"
    )
    assert [feature["properties"]["id"] for feature in data["features"]] == [1, 2]
    assert len(calls) == 2
    assert all("utm_medium=agentic" in call for call in calls)
    monkeypatch.setattr(
        subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="500")
    )
    with pytest.raises(RuntimeError, match="HTTP 500"):
        notebook_helpers["fetch_features"]("public.slick_plus", {}, "failed")
