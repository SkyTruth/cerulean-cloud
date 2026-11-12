"""Publish the checked-in investigation skill and advertise it across the API."""

import gzip
import hashlib
import io
import json
import re
import tarfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException, Request, Response
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from tipg.middleware import CatalogUpdateMiddleware

SKILL_NAME = "cerulean-investigation"
SKILL_ROOT = Path(__file__).resolve().parents[2] / "docs" / "agent-skills" / SKILL_NAME
SKILL_URL = f"/skills/{SKILL_NAME}/SKILL.md"
ARCHIVE_URL = f"/skills/{SKILL_NAME}.tar.gz"
INDEX_URL = "/.well-known/agent-skills/index.json"
GUIDES = {
    "api.md": "API queries, pagination and exports",
    "origin.md": "Possible origins and counts by source type",
    "methodology.md": "Evidence, coverage and comparisons",
    "computer-use.md": "Using the Cerulean map",
}


@dataclass(frozen=True)
class Document:
    """An immutable publication artifact and its content digest."""

    content: bytes
    media_type: str

    @property
    def digest(self) -> str:
        return "sha256:" + hashlib.sha256(self.content).hexdigest()


@lru_cache(maxsize=1)
def publication() -> dict[str, Document]:
    """Derive every published artifact from the skill shipped in this image."""
    files = {
        path.relative_to(SKILL_ROOT).as_posix(): path.read_bytes()
        for path in sorted(SKILL_ROOT.rglob("*.md"))
    }
    skill = files["SKILL.md"].decode()
    metadata = yaml.safe_load(skill.split("---", 2)[1])
    evidence = skill.split("## Evidence rules\n\n", 1)[1].split("\n## ", 1)[0]
    guide_links = "\n".join(
        f"- [{title}](/skills/{SKILL_NAME}/references/{filename})"
        for filename, title in GUIDES.items()
    )
    llms = (
        "# Cerulean OGC API\n\n"
        f"> {metadata['description']}\n\n"
        "Read the investigation skill before constructing requests. It defines "
        "agentic attribution, quality screening, completeness and evidence limits. "
        "Fetch only the task-specific references you need.\n\n"
        f"## Start here\n\n- [Investigation skill]({SKILL_URL})\n\n"
        f"## Evidence rules\n\n{evidence}\n\n"
        f"## Task guides\n\n{guide_links}\n\n"
        f"## Optional\n\n- [Installable skill bundle]({ARCHIVE_URL})\n"
        f"- [Skill discovery index]({INDEX_URL})\n- [OpenAPI schema](/api)\n"
    )
    archive = io.BytesIO()
    with gzip.GzipFile(fileobj=archive, mode="wb", mtime=0, filename="") as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as bundle:
            for filename, content in files.items():
                entry = tarfile.TarInfo(filename)
                entry.size = len(content)
                entry.mode = 0o644
                bundle.addfile(entry, io.BytesIO(content))
    archive_doc = Document(archive.getvalue(), "application/gzip")
    index = {
        "$schema": "https://schemas.agentskills.io/discovery/0.2.0/schema.json",
        "skills": [
            {
                "name": metadata["name"],
                "type": "archive",
                "description": metadata["description"],
                "url": ARCHIVE_URL,
                "digest": archive_doc.digest,
            }
        ],
    }
    return {
        **{
            f"/skills/{SKILL_NAME}/{filename}": Document(content, "text/markdown")
            for filename, content in files.items()
        },
        "/llms.txt": Document(llms.encode(), "text/markdown"),
        ARCHIVE_URL: archive_doc,
        INDEX_URL: Document(json.dumps(index, indent=2).encode(), "application/json"),
    }


def document_response(request: Request, path: str) -> Response:
    """Serve only published files, with conditional retrieval and a short cache."""
    document = publication().get(path)
    if document is None:
        raise HTTPException(status_code=404, detail="Documentation not found")
    etag = f'"{document.digest}"'
    # Compression can change wire bytes while the document content stays equivalent.
    headers = {"ETag": f"W/{etag}", "Cache-Control": "public, max-age=300"}
    requested_tags = request.headers.get("if-none-match", "").split(",")
    if any(tag.strip().removeprefix("W/") in (etag, "*") for tag in requested_tags):
        return Response(status_code=304, headers=headers)
    return Response(document.content, media_type=document.media_type, headers=headers)


router = APIRouter()


@router.api_route("/llms.txt", methods=["GET", "HEAD"], include_in_schema=False)
def agent_guide(request: Request):
    return document_response(request, "/llms.txt")


@router.api_route(INDEX_URL, methods=["GET", "HEAD"], include_in_schema=False)
def skill_index(request: Request):
    return document_response(request, INDEX_URL)


@router.api_route(
    "/skills/{document:path}", methods=["GET", "HEAD"], include_in_schema=False
)
def skill_document(request: Request, document: str):
    return document_response(request, f"/skills/{document}")


class DocumentationCatalogMiddleware(CatalogUpdateMiddleware):
    """Documentation requests do not need or refresh the database catalog."""

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        path = scope.get("path", "")
        if path in (
            "/llms.txt",
            INDEX_URL,
            "/api",
            "/api.html",
            "/robots.txt",
        ) or path.startswith("/skills/"):
            await self.app(scope, receive, send)
            return
        await super().__call__(scope, receive, send)


class AgentDiscoveryMiddleware:
    """Link every response; enrich small OGC metadata without buffering features."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request = Request(scope)
        guide_url = str(request.base_url) + "llms.txt"
        link = {
            "href": guide_url,
            "rel": "describedby",
            "type": "text/markdown",
            "title": "Cerulean investigation guide for agents",
        }
        metadata_path = scope["path"] in ("/", "/api") or re.fullmatch(
            r"/collections(?:/[^/]+)?", scope["path"]
        )
        start = None
        chunks = []

        async def send_with_discovery(message: Message):
            nonlocal start
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers.append(
                    "Link", f'<{guide_url}>; rel="describedby"; type="text/markdown"'
                )
                if (
                    metadata_path
                    and message["status"] == 200
                    and headers.get("content-type", "").startswith("application/json")
                ):
                    start = message
                    return
            elif message["type"] == "http.response.body" and start is not None:
                chunks.append(message.get("body", b""))
                if message.get("more_body", False):
                    return
                data = json.loads(b"".join(chunks))
                if scope["path"] == "/api":
                    data["externalDocs"] = {
                        "description": link["title"],
                        "url": guide_url,
                    }
                else:
                    data.setdefault("links", []).append(link)
                    for collection in data.get("collections", []):
                        collection.setdefault("links", []).append(link)
                body = json.dumps(data, separators=(",", ":")).encode()
                MutableHeaders(scope=start)["content-length"] = str(len(body))
                await send(start)
                await send({"type": "http.response.body", "body": body})
                start = None
                return
            await send(message)

        await self.app(scope, receive, send_with_discovery)
