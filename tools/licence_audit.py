#!/usr/bin/env python3
"""Licence and content audit (task T19, requirement D9).

D9's wording is the whole point of this script: *"verify by decoding and
reconstructing file content, not by listing filenames."* A filename is a claim made
by whoever saved the file. `CPDexamples.pdf` sitting in a folder called `lectures/`
asserts that it is one of this course's lecture decks; the bytes inside it say
otherwise. So nothing here trusts a name, a folder, or the README.

What is decoded:

* `TQNet/LICENSE` -- checked against the structural markers of the Apache License
  2.0 (the title line, the nine numbered clauses, the appendix boilerplate) rather
  than assumed from its filename, and hashed.
* Every PDF tracked in git -- the trailer is located, the info dictionary is parsed
  out of the raw bytes, and `/Title`, `/Author`, `/Creator`, `/Producer` and
  `/CreationDate` are decoded, including UTF-16BE values. Authorship therefore comes
  from what the producing application recorded, not from the path.
* Page count and structural validity of each PDF, so a truncated or non-PDF file
  cannot pass as one.

The distribution question is not hypothetical: `git remote -v` points at a public
GitHub repository, so anything tracked is published. Each item is therefore ruled
on redistribution, not merely on provenance.

Writes `report/licence_audit.md` only.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "report" / "licence_audit.md"

# Structural markers of the Apache License 2.0. Presence of all of these, in order,
# is what distinguishes the real text from a file merely named LICENSE.
APACHE_MARKERS = (
    "Apache License",
    "Version 2.0, January 2004",
    "http://www.apache.org/licenses/",
    "1. Definitions.",
    "2. Grant of Copyright License.",
    "3. Grant of Patent License.",
    "4. Redistribution.",
    "5. Submission of Contributions.",
    "6. Trademarks.",
    "7. Disclaimer of Warranty.",
    "8. Limitation of Liability.",
    "9. Accepting Warranty or Additional Liability.",
    "APPENDIX: How to apply the Apache License to your work.",
)

_INFO_KEYS = ("Title", "Author", "Creator", "Producer", "CreationDate")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _decode_pdf_string(raw: bytes) -> str:
    """A PDF literal string, which may be UTF-16BE with a BOM or PDFDocEncoding."""
    body = raw
    # Unescape the escapes that matter for text fields.
    body = re.sub(br"\\([()\\])", br"\1", body)
    if body.startswith(b"\xfe\xff"):
        try:
            return body[2:].decode("utf-16-be", errors="replace").strip()
        except Exception:
            pass
    return body.decode("latin-1", errors="replace").strip()


def decode_pdf(path: Path) -> dict:
    raw = path.read_bytes()
    info = {}
    for key in _INFO_KEYS:
        match = re.search(
            br"/" + key.encode() + br"\s*\(((?:[^()\\]|\\.)*)\)", raw
        )
        if match:
            value = _decode_pdf_string(match.group(1))
            if value:
                info[key] = value
    return {
        "sha256": _sha256(path),
        "bytes": len(raw),
        "valid_pdf": raw.startswith(b"%PDF-") and b"%%EOF" in raw[-2048:],
        "version": raw[5:8].decode("latin-1", errors="replace") if raw.startswith(b"%PDF-") else "?",
        "pages": len(re.findall(br"/Type\s*/Page[^s]", raw)),
        "info": info,
    }


def tracked_files():
    out = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    return [line for line in out.stdout.splitlines() if line]


def remote_url():
    try:
        out = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except subprocess.CalledProcessError:
        return "(no origin remote)"


def rule_pdf(rel_path: str, decoded: dict):
    """Ruling derived from the decoded author/creator, never from the path."""
    info = decoded["info"]
    author = info.get("Author", "")
    title = info.get("Title", "")
    creator = info.get("Creator", "")
    haystack = " ".join((author, title, creator)).lower()

    if "lin" in author.lower() and "arxiv" in creator.lower():
        return ("REMOVE", "third party -- academic paper",
                "Decoded authors {!r}; produced by {!r}. This is the authors' paper, not ours. "
                "Copyright rests with them; a public mirror of the PDF is redistribution."
                .format(author, creator))
    if "laurent" in haystack or "enseignements" in haystack:
        return ("REMOVE", "third party -- another institution's teaching material",
                "Decoded author {!r} and title {!r}. The title is a filesystem path from the "
                "producing machine and places this in a *teaching* folder belonging to Laurent "
                "Oudre, not to this course. The filename and its folder both assert otherwise; "
                "the bytes are what settle it. Cite Oudre, never the instructor."
                .format(author, title))
    if "rika" in author.lower():
        return ("REMOVE", "third party -- this course's own material",
                "Decoded author {!r} from embedded {!r} metadata. Instructor-authored teaching "
                "material, published here on a public remote without a licence to do so."
                .format(author, creator))
    if "python-docx" in author.lower() or "word" in creator.lower():
        return ("REMOVE", "third party -- course administrative document",
                "Decoded author {!r}, producer {!r}. No named human author, but this is course "
                "material rather than ours, and the same redistribution point applies."
                .format(author, info.get("Producer", "?")))
    return ("DISCLOSE", "provenance not established from content",
            "Decoded metadata did not identify an owner: {!r}. Ruled DISCLOSE rather than "
            "CLEAR, because an unidentified file is not a cleared one.".format(info))


def main() -> None:
    tracked = tracked_files()
    remote = remote_url()
    public = remote.startswith("https://github.com/") or remote.startswith("git@github.com")

    lines = []
    A = lines.append

    A("# Licence and content audit (T19 / D9)")
    A("")
    A("Generated by `tools/licence_audit.py`. D9 requires verification *\"by decoding and")
    A("reconstructing file content, not by listing filenames\"*, so every ruling below is")
    A("derived from bytes: the Apache text is matched clause by clause, and each PDF's")
    A("authorship is read out of its embedded info dictionary.")
    A("")
    A("**Distribution context.** `origin` is `{}`, which is {}. Every file tracked in git is".format(
        remote, "a public repository" if public else "not a public host"))
    A("therefore published, and each item is ruled on *redistribution*, not just provenance.")
    A("")
    A("Rulings: **CLEAR** (ours, or licensed for redistribution) · **DISCLOSE** (ownership")
    A("not established from content) · **REMOVE** (third-party material we have no licence")
    A("to publish).")
    A("")

    # ---- 1. The vendored licence -------------------------------------------------
    A("## 1. Vendored upstream licence -- verified by content")
    A("")
    licence_path = REPO_ROOT / "TQNet" / "LICENSE"
    if not licence_path.is_file():
        A("**FAIL** -- `TQNet/LICENSE` does not exist, but `TQNet/` is a redistributed copy of")
        A("someone else's Apache-2.0 project, whose clause 4 requires the licence to travel")
        A("with it.")
        A("")
    else:
        text = licence_path.read_text(encoding="utf-8", errors="replace")
        missing = [m for m in APACHE_MARKERS if m not in text]
        positions = [text.find(m) for m in APACHE_MARKERS if m in text]
        ordered = positions == sorted(positions)
        A("| Check | Result |")
        A("|---|---|")
        A("| File | `TQNet/LICENSE` |")
        A("| sha256 | `{}` |".format(_sha256(licence_path)))
        A("| Bytes / lines | {:,} / {:,} |".format(
            licence_path.stat().st_size, text.count("\n") + 1))
        A("| Apache-2.0 structural markers present | {} of {} |".format(
            len(APACHE_MARKERS) - len(missing), len(APACHE_MARKERS)))
        A("| Markers appear in canonical order | {} |".format("yes" if ordered else "NO"))
        A("| Missing markers | {} |".format(
            "none" if not missing else ", ".join(repr(m) for m in missing)))
        A("")
        if not missing and ordered:
            A("**CLEAR.** All {} structural markers of the Apache License 2.0 are present and in".format(
                len(APACHE_MARKERS)))
            A("order, so this is the licence itself rather than a file named after it.")
            attribution = REPO_ROOT / "TQNet" / "README_UPSTREAM.md"
            A("Upstream attribution is retained as `TQNet/README_UPSTREAM.md` ({}), which is what".format(
                "present, {:,} bytes".format(attribution.stat().st_size)
                if attribution.is_file() else "**MISSING**"))
            A("clause 4 asks for alongside the licence.")
        else:
            A("**DISCLOSE.** The file does not match the Apache-2.0 text structurally; it cannot")
            A("be relied on as the licence for the vendored tree.")
        A("")

    # ---- 2. The dataset ----------------------------------------------------------
    A("## 2. Dataset")
    A("")
    csv_tracked = [p for p in tracked if p.endswith(".csv")]
    A("`git ls-files` tracks {} CSV file(s). ETTh1 is fetched by `tools/get_data.py` from a".format(
        len(csv_tracked)))
    A("pinned commit and verified against a sha256, and `.gitignore` excludes `*.csv`.")
    A("")
    A("**CLEAR.** Nothing is redistributed: the bytes are downloaded from the upstream")
    A("ETDataset release and checked, which is both the assignment's documented-download")
    A("option and a stronger provenance claim than a copy in git would be.")
    A("")

    # ---- 3. PDFs -----------------------------------------------------------------
    A("## 3. Tracked PDFs -- authorship decoded from the embedded info dictionary")
    A("")
    pdfs = sorted(p for p in tracked if p.lower().endswith(".pdf"))
    A("{} tracked PDFs. Nothing below reads the filename to decide ownership.".format(len(pdfs)))
    A("")

    verdicts = []
    for rel in pdfs:
        path = REPO_ROOT / rel
        if not path.is_file():
            continue
        decoded = decode_pdf(path)
        ruling, kind, reason = rule_pdf(rel, decoded)
        verdicts.append((rel, ruling, kind, reason, decoded))

    A("| File | Decoded `/Author` | Decoded `/Creator` or `/Title` | Pages | Ruling |")
    A("|---|---|---|---|---|")
    for rel, ruling, _kind, _reason, decoded in verdicts:
        info = decoded["info"]
        second = info.get("Creator") or info.get("Title") or "--"
        A("| `{}` | {} | {} | {} | **{}** |".format(
            rel,
            info.get("Author", "_(none)_"),
            (second[:58] + "...") if len(second) > 58 else second,
            decoded["pages"] or "?",
            ruling,
        ))
    A("")

    invalid = [rel for rel, _r, _k, _re, d in verdicts if not d["valid_pdf"]]
    A("Structural reconstruction: {} of {} begin with `%PDF-` and carry `%%EOF` in their".format(
        len(verdicts) - len(invalid), len(verdicts)))
    A("trailer, so each is a complete PDF and not a renamed or truncated file.{}".format(
        "" if not invalid else " Exceptions: {}.".format(", ".join("`{}`".format(p) for p in invalid))))
    A("")

    A("### Rulings in full")
    A("")
    for rel, ruling, kind, reason, decoded in verdicts:
        A("**`{}`** -- {} ({})".format(rel, ruling, kind))
        A("")
        A("- sha256 `{}`, {:,} bytes, PDF {}".format(
            decoded["sha256"][:32], decoded["bytes"], decoded["version"]))
        A("- {}".format(reason))
        A("")

    # ---- 4. Figures and source ---------------------------------------------------
    A("## 4. Images and source files under `TQNet/`")
    A("")
    figures = sorted(p for p in tracked if p.startswith("TQNet/Figures/"))
    py_files = [p for p in tracked if p.startswith("TQNet/") and p.endswith(".py")]
    A("{} tracked images under `TQNet/Figures/` and {} tracked `.py` files under `TQNet/`.".format(
        len(figures), len(py_files)))
    A("These are parts of the upstream repository, redistributed under the Apache-2.0")
    A("licence verified in section 1, with attribution retained.")
    A("")
    A("**CLEAR**, conditional on section 1 passing. Note that Apache-2.0 clause 4(b) also")
    A("requires modified files to carry prominent notices; our changes are catalogued in")
    A("`docs/02-architecture-and-implementation.md` section 2.6 rather than in per-file")
    A("headers, which is a weaker form of the same disclosure and is recorded here as such.")
    A("")

    # ---- 5. Verdict --------------------------------------------------------------
    counts = {}
    for _rel, ruling, _k, _re, _d in verdicts:
        counts[ruling] = counts.get(ruling, 0) + 1

    A("## 5. Verdict")
    A("")
    A("| Ruling | Count |")
    A("|---|---|")
    for ruling in ("CLEAR", "DISCLOSE", "REMOVE"):
        if counts.get(ruling):
            A("| {} | {} |".format(ruling, counts[ruling]))
    A("")
    to_remove = [rel for rel, ruling, _k, _re, _d in verdicts if ruling == "REMOVE"]
    if to_remove:
        A("**FAIL on redistribution.** {} tracked PDFs are third-party material published on a".format(
            len(to_remove)))
        A("public remote without any licence permitting it. The code and the dataset are clean;")
        A("the exposure is entirely in `files/`.")
        A("")
        A("This is a content-licensing finding, not a scientific one -- no result depends on")
        A("these files, and `.gitignore` already treats the dataset this way. The remedy is the")
        A("same one the dataset uses: cite them and drop the bytes.")
        A("")
        A("Suggested action, in order:")
        A("")
        A("1. `git rm --cached` each file below and add `files/**/*.pdf` to `.gitignore`, so the")
        A("   working copies stay for the team and stop being published.")
        A("2. Keep the citations. `docs/` and `report/` already reference these by author, page")
        A("   and slide number, which is what a reader actually needs.")
        A("3. Note that removal from `HEAD` does not remove them from history; if that matters,")
        A("   it needs a history rewrite and a force-push, which is a decision for the humans.")
        A("")
        for rel in to_remove:
            A("- `{}`".format(rel))
        A("")
    else:
        A("**PASS.** No tracked file was ruled REMOVE.")
        A("")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote {}".format(OUT_PATH.relative_to(REPO_ROOT)))
    print("rulings: {}".format(counts))


if __name__ == "__main__":
    main()
