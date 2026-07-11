import subprocess
import urllib.request
from pathlib import Path

_BASE = "https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1792"

HR500K_FILES = [
    f"{_BASE}/hr500k-train.conllu.gz?sequence=3&isAllowed=y",
    f"{_BASE}/hr500k-dev.conllu.gz?sequence=4&isAllowed=y",
    f"{_BASE}/hr500k-test.conllu.gz?sequence=5&isAllowed=y",
]


def ispravime_repo_url() -> str:
    return "https://github.com/Ispravi-Me/Dataset-of-Misspelings-and-Corrections.git"


def ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def download_hr500k(raw_dir):
    d = ensure_dir(Path(raw_dir) / "hr500k")
    for url in HR500K_FILES:
        name = url.split("/")[-1].split("?")[0]
        dest = d / name
        if not dest.exists():
            urllib.request.urlretrieve(url, dest)
    return d


def clone_ispravime(raw_dir):
    d = Path(raw_dir) / "ispravime"
    if not d.exists():
        subprocess.run(["git", "clone", "--depth", "1", ispravime_repo_url(), str(d)], check=True)
    return d
