from crogrammar.data.download import HR500K_FILES, ispravime_repo_url, ensure_dir, HUNSPELL_HR_DIC_URL, RAPUT_URL
from pathlib import Path

def test_hr500k_files_are_gz_urls():
    assert any("hr500k-train" in u for u in HR500K_FILES)
    assert all(u.startswith("https://") for u in HR500K_FILES)

def test_hunspell_dic_url():
    assert HUNSPELL_HR_DIC_URL.startswith("https://")
    assert HUNSPELL_HR_DIC_URL.endswith("hr_HR.dic")

def test_raput_url():
    assert RAPUT_URL.startswith("https://")
    assert "1435" in RAPUT_URL

def test_ispravime_repo_url():
    assert ispravime_repo_url().endswith(".git")
    assert "Ispravi-Me" in ispravime_repo_url()

def test_ensure_dir_creates(tmp_path):
    target = tmp_path / "raw" / "sub"
    ensure_dir(target)
    assert Path(target).is_dir()
