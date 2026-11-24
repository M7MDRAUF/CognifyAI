"""Local file storage for uploaded documents.

Files live under ``{root}/{project_id}/{uuid}_{sanitized-name}`` — outside
the source tree, with resolved-path containment as defense in depth.
"""

import shutil
import uuid
from pathlib import Path


class FileStorage:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).resolve()

    @property
    def root(self) -> Path:
        return self._root

    def _project_dir(self, project_id: uuid.UUID) -> Path:
        return self._contained(self._root / str(project_id))

    def _contained(self, path: Path) -> Path:
        resolved = path.resolve()
        if not resolved.is_relative_to(self._root):
            msg = f"Storage path escapes the upload root: {resolved}"
            raise RuntimeError(msg)
        return resolved

    def build_stored_filename(self, file_id: uuid.UUID, sanitized_filename: str) -> str:
        return f"{file_id}_{sanitized_filename}"

    def save(self, project_id: uuid.UUID, stored_filename: str, content: bytes) -> Path:
        project_dir = self._project_dir(project_id)
        project_dir.mkdir(parents=True, exist_ok=True)
        target = self._contained(project_dir / stored_filename)
        target.write_bytes(content)
        return target

    def read(self, project_id: uuid.UUID, stored_filename: str) -> bytes:
        return self._contained(self._project_dir(project_id) / stored_filename).read_bytes()

    def delete_file(self, project_id: uuid.UUID, stored_filename: str) -> None:
        target = self._contained(self._project_dir(project_id) / stored_filename)
        target.unlink(missing_ok=True)

    def delete_project_dir(self, project_id: uuid.UUID) -> None:
        project_dir = self._project_dir(project_id)
        if project_dir.is_dir():
            shutil.rmtree(project_dir)
