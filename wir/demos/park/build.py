"""Build the Yumon Park page: python build.py park_demo.json out.html

park_demo.json comes from `cargo run --release -- park --export park_demo.json`."""
import pathlib
import sys

here = pathlib.Path(__file__).parent
data = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8").replace("</", "<" + chr(92) + "/")
page = (here / "park_template.html").read_text(encoding="utf-8")
page = page.replace("__DATA__", data)
page = page.replace("__CORE__", (here / "park_core.js").read_text(encoding="utf-8"))
page = page.replace("__UI__", (here / "park_ui.js").read_text(encoding="utf-8"))
pathlib.Path(sys.argv[2]).write_text(page, encoding="utf-8")
print(f"wrote {sys.argv[2]} ({len(page):,} bytes)")
