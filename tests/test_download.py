from crogrammar.data.download import HR500K_FILES, ispravime_repo_url, ensure_dir
from pathlib import Path

def test_hr500k_files_are_gz_urls():
    assert any("hr500k-train" in u for u in HR500K_FILES)
    assert all(u.startswith("https://") for u in HR500K_FILES)

def test_ispravime_repo_url():
    assert ispravime_repo_url().endswith(".git")
    assert "Ispravi-Me" in ispravime_repo_url()

def test_ensure_dir_creates(tmp_path):
    target = tmp_path / "raw" / "sub"
    ensure_dir(target)
    assert Path(target).is_dir()
