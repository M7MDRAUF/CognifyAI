"""Project use cases (FR-PROJ-001/002/003)."""

import logging
import uuid

from app.core.errors import DomainProfileDisabledError, ProjectNotFoundError
from app.domain.enums import DomainProfile
from app.domain.models import Project
from app.integrations.qdrant.client import QdrantService
from app.schemas.projects import ProjectCreate, ProjectStats
from app.services.interfaces import ProjectRepositoryLike, SupportsCommit
from app.utils.files import FileStorage
from app.utils.ids import new_uuid
from app.utils.time import utc_now

logger = logging.getLogger(__name__)


class ProjectService:
    def __init__(
        self,
        session: SupportsCommit,
        projects: ProjectRepositoryLike,
        storage: FileStorage,
        qdrant: QdrantService,
        pharma_profile_enabled: bool = True,
    ) -> None:
        """``pharma_profile_enabled`` is the ``PHARMA_PROFILE_ENABLED`` rollout gate; its
        default is the setting's, so a construction that predates it behaves as deployed."""
        self._session = session
        self._projects = projects
        self._storage = storage
        self._qdrant = qdrant
        self._pharma_profile_enabled = pharma_profile_enabled

    async def create(self, data: ProjectCreate) -> Project:
        if data.domain_profile is DomainProfile.PHARMA_MANUFACTURING and not (
            self._pharma_profile_enabled
        ):
            msg = "Pharmaceutical manufacturing projects are disabled on this deployment."
            raise DomainProfileDisabledError(msg)
        now = utc_now()
        project = Project(
            id=new_uuid(),
            name=data.name,
            description=data.description,
            domain=data.domain,
            objective=data.objective,
            domain_profile=data.domain_profile,
            created_at=now,
            updated_at=now,
        )
        await self._projects.add(project)
        await self._session.commit()
        return project

    async def list_projects(self) -> list[Project]:
        return await self._projects.list_all()

    async def get_project(self, project_id: uuid.UUID) -> Project:
        project = await self._projects.get(project_id)
        if project is None:
            raise ProjectNotFoundError("Project not found.")
        return project

    async def get_detail(self, project_id: uuid.UUID) -> tuple[Project, ProjectStats]:
        project = await self.get_project(project_id)
        stats = await self._projects.stats(project_id)
        return project, stats

    async def delete_project(self, project_id: uuid.UUID) -> None:
        """Remove the project's Qdrant points, then its rows (documents and chunks cascade),
        then its stored files: the order document delete keeps (RMD-DEF-037, RE2E-DEF-027). A
        Qdrant failure raises before anything is deleted, so the 503 is true and a retry finds
        the project. The read's transaction ends before the vector store is called, so no pooled
        connection waits idle in transaction on it (RMD-DEF-121). Files that cannot be removed
        are logged under the project's id."""
        project = await self.get_project(project_id)
        await self._session.commit()
        await self._qdrant.delete_by_project(project_id)
        await self._projects.delete(project)
        await self._session.commit()
        try:
            self._storage.delete_project_dir(project_id)
        except OSError:
            logger.warning(
                "Deleted project %s, but its files could not be removed", project_id, exc_info=True
            )
