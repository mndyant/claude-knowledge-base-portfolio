"""公開デモの配置ファイルを明示的な許可リストから組み立てる。"""

import shutil
from pathlib import Path


def main() -> None:
    """コード・独自要約・画面だけを専用ディレクトリにコピーする。"""
    root = Path(__file__).resolve().parents[1]
    target = root / ".public-deploy"
    for source, destination in [
        ("public_demo/app.py", "public_demo/app.py"),
        ("public_demo/requirements.txt", "requirements.txt"),
        ("scripts/source_notes.py", "scripts/source_notes.py"),
        ("web/index.html", "web/index.html"),
        ("web/styles.css", "web/styles.css"),
        ("web/app.js", "web/app.js"),
    ]:
        output = target / destination
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / source, output)
    (target / "app.py").write_text(
        "from public_demo.app import app\n", encoding="utf-8"
    )
    print(target)


if __name__ == "__main__":
    main()
