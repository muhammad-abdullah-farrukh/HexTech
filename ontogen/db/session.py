"""Session factory for Ontogen's database work.

Ontogen points at the *same* Postgres as ocr_resume_parser, so it reuses the
parser's ``make_session_factory`` rather than duplicating engine construction —
same ``pool_pre_ping`` / ``pool_size`` policy, same fork-safety note (build the
factory fresh inside each worker process if parallelism is ever added; a
SQLAlchemy engine inherited across a fork shares one TCP socket and corrupts
concurrent queries).
"""
from __future__ import annotations

from resume_parser.db.session import make_session_factory

__all__ = ["make_session_factory"]
