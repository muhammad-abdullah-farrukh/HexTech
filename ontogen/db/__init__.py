"""Ontogen's Postgres layer — shares one database (and one SQLAlchemy Base)
with ocr_resume_parser.

Ontogen runs inside the parser repo's venv; this package reaches the parser by
putting both source roots on sys.path (idempotent), matching Ontogen's existing
``sys.path.insert(0, parent)`` idiom:

  - the Ontogen source root, so submodules can ``from config import …``;
  - the ocr_resume_parser repo root, so ``import resume_parser…`` resolves and
    the FK to ``resumes`` is declared against the same models/Base.
"""
import sys
from pathlib import Path

_ONTOGEN_ROOT = Path(__file__).resolve().parents[1]          # …/ontogen
_PARSER_ROOT = _ONTOGEN_ROOT.parent / "ocr-resume-paser"     # sibling repo

for _p in (str(_ONTOGEN_ROOT), str(_PARSER_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
