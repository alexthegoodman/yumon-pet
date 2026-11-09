"""Build the WIR Chess page: python build.py chess_demo.json out.html

chess_demo.json comes from `cargo run --release -- chess --export chess_demo.json`."""
import pathlib
import sys

here = pathlib.Path(__file__).parent
data = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8").replace("</", "<" + chr(92) + "/")
core = (here / "chess_core.js").read_text(encoding="utf-8")
assert "</script" not in core
page = (here / "chess_template.html").read_text(encoding="utf-8")
page = page.replace("__DATA__", data)
page = page.replace("__CORE__", core)
page = page.replace("__UI__", (here / "chess_ui.js").read_text(encoding="utf-8"))
pathlib.Path(sys.argv[2]).write_text(page, encoding="utf-8")
print(f"wrote {sys.argv[2]} ({len(page):,} bytes)")
