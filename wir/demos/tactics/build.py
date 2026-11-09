"""Build the WIR Skirmish page: python build.py tactics_demo.json out.html

tactics_demo.json comes from `cargo run --release -- tactics --export tactics_demo.json`."""
import pathlib
import sys

here = pathlib.Path(__file__).parent
data = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8").replace("</", "<" + chr(92) + "/")
page = (here / "skirmish_template.html").read_text(encoding="utf-8").replace("__DATA__", data)
pathlib.Path(sys.argv[2]).write_text(page, encoding="utf-8")
print(f"wrote {sys.argv[2]} ({len(page):,} bytes)")
