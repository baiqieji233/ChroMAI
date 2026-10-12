"""Load the packaged numeric MiaCode reference tables, never a native runtime."""
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path

REFERENCE_COMMIT = "6573efb611056bccd046dd17bc2f4cceda78bf88"


@lru_cache(maxsize=1)
def reference_data():
    root = Path(__file__).parent / "reference"
    manifest = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    if manifest["commit"] != REFERENCE_COMMIT:
        raise RuntimeError("Python 无理检测的参考版本不匹配，请恢复项目文件")
    packed = (root / "slide_data.json.gz").read_bytes()
    if hashlib.sha256(packed).hexdigest() != manifest["sha256"]:
        raise RuntimeError("Python 无理检测的路径参考数据校验失败，请恢复项目文件")
    return json.loads(gzip.decompress(packed))
