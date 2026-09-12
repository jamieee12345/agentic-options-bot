"""Assemble the static site GitHub Pages serves (see .github/workflows/pages.yml).

Layout of the output directory:
    index.html              the live dashboard (dashboard/live_account_dashboard.py --once)
    journals/index.html     list of every daily journal, newest first
    journals/<file>.html    each committed reports/journal-<date>.html, unchanged

Runs with no credentials -- everything comes from committed repo files.
"""
from __future__ import annotations

import html
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
JOURNAL_RE = re.compile(r"^journal-(\d{4}-\d{2}-\d{2})\.html$")


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "site").resolve()
    if out.exists():
        shutil.rmtree(out)
    (out / "journals").mkdir(parents=True)

    subprocess.run(
        [sys.executable, "-m", "dashboard.live_account_dashboard", "--once", "--output", str(out / "index.html")],
        cwd=ROOT, check=True,
    )

    dates = []
    if REPORTS.exists():
        for f in sorted(REPORTS.iterdir(), reverse=True):
            m = JOURNAL_RE.match(f.name)
            if m:
                shutil.copy2(f, out / "journals" / f.name)
                dates.append(m.group(1))

    rows = "".join(
        f"<li><a href='journal-{d}.html'>{html.escape(d)}</a></li>" for d in dates
    ) or "<li class='empty'>No journals committed yet -- the first one lands after the 5:40pm ET routine.</li>"
    (out / "journals" / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Daily journals</title>"
        "<style>body{font:15px/1.5 system-ui,sans-serif;max-width:40rem;margin:2rem auto;padding:0 1rem;color:#1b1f24;background:#fff}"
        "a{color:#0b5fff}ul{padding-left:1.2rem}.empty{list-style:none;color:#666}nav a{margin-right:1rem}</style>"
        "<nav><a href='../'>&larr; Dashboard</a></nav><h1>Daily journals</h1><ul>" + rows + "</ul>",
        encoding="utf-8",
    )
    # Tell Pages not to run Jekyll over the output.
    (out / ".nojekyll").write_text("", encoding="utf-8")
    print(f"site built at {out}: dashboard + {len(dates)} journal(s)")


if __name__ == "__main__":
    main()
