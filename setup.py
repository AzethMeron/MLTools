from pathlib import Path
from setuptools import setup, find_packages
import os

def read_requirements(path: Path) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [
        ln.strip()
        for ln in lines
        if ln.strip() and not ln.strip().startswith("#")
    ]

HERE = Path(__file__).parent
install_requires = read_requirements(Path(os.path.join(HERE, "requirements.txt")))

setup(
    name="MLTools",
    version="1.2.0",
    packages=find_packages(),
    install_requires=install_requires,
)
