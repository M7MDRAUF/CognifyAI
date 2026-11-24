"""Document routes (SDD section 7.3). Handlers stay logic-free."""

import uuid

from fastapi import APIRouter, BackgroundTasks, UploadFile

from app.api.deps import DocumentServiceDep, IngestionRunnerDep, SettingsDep
from app.schemas.documents import WITHOUT_CONTROLLED_METADATA, DocumentDetail, DocumentRead

router = APIRouter(tags=["documents"])


@router.post(
    "/projects/{project_id}/documents",
    response_model=DocumentRead,
    response_model_exclude=set(WITHOUT_CONTROLLED_METADATA),
    status_code=201,
)
async def upload_document(
    project_id: uuid.UUID,
    file: UploadFile,
    background: BackgroundTasks,
    service: DocumentServiceDep,
    runner: IngestionRunnerDep,
    settings: SettingsDep,
) -> DocumentRead:
    content = await file.read(settings.max_upload_size_bytes + 1)
    document = await service.upload(
        project_id,
        file.filename or "upload",
        file.content_type or "application/octet-stream",
        content,
    )
    background.add_task(runner, document.id)
    return DocumentRead.model_validate(document)


@router.get("/projects/{project_id}/documents", response_model=list[DocumentRead])
async def list_documents(project_id: uuid.UUID, service: DocumentServiceDep) -> list[DocumentRead]:
    documents = await service.list_for_project(project_id)
    controlled = await service.controlled_metadata([document.id for document in documents])
    return [
        DocumentRead.model_validate(document).model_copy(
            update={"controlled_metadata": controlled.get(document.id)}
        )
        for document in documents
    ]


@router.get("/documents/{document_id}", response_model=DocumentDetail)
async def get_document(document_id: uuid.UUID, service: DocumentServiceDep) -> DocumentDetail:
    document, chunk_count = await service.get_detail(document_id)
    controlled = await service.controlled_metadata([document.id])
    return DocumentDetail(
        **DocumentRead.model_validate(document).model_dump(
            exclude=set(WITHOUT_CONTROLLED_METADATA)
        ),
        controlled_metadata=controlled.get(document.id),
        chunk_count=chunk_count,
    )


@router.delete("/documents/{document_id}", status_code=204)
async def delete_document(document_id: uuid.UUID, service: DocumentServiceDep) -> None:
    await service.delete(document_id)


@router.post(
    "/documents/{document_id}/reindex",
    response_model=DocumentRead,
    response_model_exclude=set(WITHOUT_CONTROLLED_METADATA),
    status_code=202,
)
async def reindex_document(
    document_id: uuid.UUID,
    background: BackgroundTasks,
    service: DocumentServiceDep,
    runner: IngestionRunnerDep,
) -> DocumentRead:
    document = await service.prepare_reindex(document_id)
    background.add_task(runner, document.id)
    return DocumentRead.model_validate(document)
