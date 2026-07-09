"""Optional Postgres persistence for the structured resume output.

Nothing here is imported by the pipeline itself — `run_pipeline` stays
database-agnostic and only calls an injected `ingest_fn`. The CLI wires this
package in on demand when `--db-uri` is passed.
"""

from __future__ import annotations
