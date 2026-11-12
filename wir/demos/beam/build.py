"""Build the WIR Beam page: python build.py beam_demo.json out.html

beam_demo.json comes from
`cargo run --release -- beam --beams 8 --vocab 16 --cap 48 --qmax 31 --export beam_demo.json`.
The Rust test vectors are left out of the page (harness.js uses them)."""
import json
import pathlib
import sys

here = pathlib.Path(__file__).parent
data = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
data.pop("vectors", None)
blob = json.dumps(data, separators=(",", ":")).replace("</", "<" + chr(92) + "/")
core = (here / "beam_core.js").read_text(encoding="utf-8")
assert "</script" not in core
page = (here / "beam_template.html").read_text(encoding="utf-8")
page = page.replace("__DATA__", blob)
page = page.replace("__CORE__", core)
page = page.replace("__UI__", (here / "beam_ui.js").read_text(encoding="utf-8"))
pathlib.Path(sys.argv[2]).write_text(page, encoding="utf-8")
print(f"wrote {sys.argv[2]} ({len(page):,} bytes)")
