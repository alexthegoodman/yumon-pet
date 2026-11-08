"""Build the Yumon Ant page: python build.py ant_demo.json out.html

ant_demo.json comes from `cargo run --release -- ant --export ant_demo.json`."""
import pathlib
import sys

here = pathlib.Path(__file__).parent
data = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8").replace("</", "<" + chr(92) + "/")
page = (here / "ant_template.html").read_text(encoding="utf-8")
page = page.replace("__DATA__", data)
page = page.replace("__CORE__", (here / "ant_core.js").read_text(encoding="utf-8"))
page = page.replace("__UI__", (here / "ant_ui.js").read_text(encoding="utf-8"))
pathlib.Path(sys.argv[2]).write_text(page, encoding="utf-8")
print(f"wrote {sys.argv[2]} ({len(page):,} bytes)")
