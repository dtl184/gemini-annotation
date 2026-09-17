"""HTTP client for ogoudey/Annotator's Flask API.

Everything this pipeline knows about a dataset - its views, per-view file
layout, episode placement, fps, existing annotations - comes from this
client. We never read the dataset directory ourselves; Annotator's own
scanner (dataset.py: build_timeline / read_episodes) is the single source of
truth, so the pipeline stays a pure client of the GUI's API and works
against any machine the Annotator server can see, not just the one it runs
on.

Endpoints used (see Annotator's app.py):
  GET  /api/session   - dataset info, global timeline, episodes, and the
                         current project (creating one if none exists yet)
  GET  /media          - stream a video file from inside the dataset root
  POST /api/project    - save (or update) the project's styles/layers/clips
  POST /api/export/lerobot - bake language_persistent into the dataset
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import requests

DEFAULT_TIMEOUT = 30
MEDIA_TIMEOUT = 300  # downloading a multi-hundred-MB source file can be slow


class AnnotatorAPIError(RuntimeError):
    pass


class AnnotatorClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def _get(self, path: str, params: dict[str, Any] | None = None, *, timeout: int = DEFAULT_TIMEOUT) -> Any:
        resp = requests.get(f"{self.base_url}{path}", params=params, timeout=timeout)
        return self._unwrap(resp)

    def _post(self, path: str, json_body: dict[str, Any], *, timeout: int = DEFAULT_TIMEOUT) -> Any:
        resp = requests.post(f"{self.base_url}{path}", json=json_body, timeout=timeout)
        return self._unwrap(resp)

    @staticmethod
    def _unwrap(resp: requests.Response) -> Any:
        if not resp.ok:
            try:
                detail = resp.json().get("error")
            except Exception:
                detail = resp.text[:500]
            raise AnnotatorAPIError(f"{resp.request.method} {resp.url} -> {resp.status_code}: {detail}")
        return resp.json()

    # ----------------------------------------------------------------
    # dataset / session
    # ----------------------------------------------------------------

    def get_session(self, root: str, *, scope: str = "dataset", chunk: int = 0, file_index: int = 0) -> dict[str, Any]:
        return self._get(
            "/api/session",
            {"root": root, "scope": scope, "chunk": chunk, "file": file_index},
        )

    # ----------------------------------------------------------------
    # media
    # ----------------------------------------------------------------

    def download_media(self, root: str, rel_path: str, dest: Path) -> Path:
        """Stream a video file from the dataset root to a local path, cached by caller."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(
            f"{self.base_url}/media", params={"root": root, "path": rel_path},
            stream=True, timeout=MEDIA_TIMEOUT,
        ) as resp:
            if not resp.ok:
                raise AnnotatorAPIError(f"GET /media?root={root}&path={rel_path} -> {resp.status_code}")
            tmp = dest.with_suffix(dest.suffix + ".part")
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
            tmp.replace(dest)
        return dest

    # ----------------------------------------------------------------
    # project
    # ----------------------------------------------------------------

    def save_project(self, project: dict[str, Any]) -> dict[str, Any]:
        return self._post("/api/project", project)

    # ----------------------------------------------------------------
    # export
    # ----------------------------------------------------------------

    def export_lerobot(
        self,
        root: str,
        *,
        style_map: dict[str, str],
        scope: str = "dataset",
        chunk: int = 0,
        file_index: int = 0,
        role: str = "assistant",
        dry_run: bool = True,
        backup: bool = True,
    ) -> dict[str, Any]:
        return self._post(
            "/api/export/lerobot",
            {
                "root": root,
                "scope": scope,
                "chunk": chunk,
                "file": file_index,
                "style_map": style_map,
                "role": role,
                "dry_run": dry_run,
                "backup": backup,
            },
            timeout=300,
        )
