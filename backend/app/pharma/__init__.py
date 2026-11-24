"""The pharmaceutical manufacturing profile.

Everything a ``pharma_manufacturing`` project adds to the baseline lives here: controlled-document
metadata, section chunking, version-aware retrieval, evidence sufficiency, conflict detection,
the regulatory-boundary policy, human escalation, the nine-section response and the workflow
subclass. The General code paths use none of it: the ORM models and the query schema import its
enumerations, and the ingestion service and job import its ingestion collaborator, which they
build and call for pharmaceutical projects only; no General workflow, retrieval or agent module
imports this package.
"""
