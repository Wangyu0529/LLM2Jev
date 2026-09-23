"""Fetch the pinned Apache-2.0 Panda assets used by the MuJoCo demo."""

from __future__ import annotations

import concurrent.futures
import hashlib
import io
import json
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path


REVISION = "822c2d8f877dd166c5b7d3c9f7e3c3b6589473b7"
REPOSITORY = "google-deepmind/mujoco_menagerie"
ROOT = Path(__file__).resolve().with_name("panda")


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "llm2jev-mujoco-demo"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(1 + attempt)
    raise RuntimeError("unreachable")


def git_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def main() -> None:
    try:
        tree = json.loads(fetch(f"https://api.github.com/repos/{REPOSITORY}/git/trees/{REVISION}?recursive=1"))
    except urllib.error.HTTPError:
        archive = tarfile.open(fileobj=io.BytesIO(fetch(
            f"https://github.com/{REPOSITORY}/archive/{REVISION}.tar.gz")), mode="r:gz")
        prefix = f"mujoco_menagerie-{REVISION}/franka_emika_panda/"
        selected = [member for member in archive.getmembers()
                    if member.name.startswith(prefix) and member.isfile()]
        hashes = {}
        for member in selected:
            relative = member.name[len(prefix):]
            target = ROOT / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            data = archive.extractfile(member).read()
            target.write_bytes(data)
            hashes[relative] = hashlib.sha256(data).hexdigest()
        ROOT.mkdir(parents=True, exist_ok=True)
        (ROOT / "manifest.json").write_text(json.dumps({
            "repository": REPOSITORY, "revision": REVISION, "license": "Apache-2.0", "files": hashes,
        }, indent=2) + "\n", encoding="utf-8")
        print(f"Verified {len(hashes)} Panda files at {ROOT}")
        return
    prefix = "franka_emika_panda/"
    entries = [item for item in tree["tree"] if item["type"] == "blob"
               and item["path"].startswith(prefix)
               and (item["path"][len(prefix):] in {"panda.xml", "LICENSE", "README.md"}
                    or item["path"].startswith(prefix + "assets/"))]

    def download(item):
        relative = item["path"][len(prefix):]
        target = ROOT / relative
        data = target.read_bytes() if target.is_file() else b""
        if git_sha(data) != item["sha"]:
            data = fetch(f"https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/{item['path']}")
            if git_sha(data) != item["sha"]:
                raise RuntimeError(f"asset checksum mismatch: {relative}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        return relative, hashlib.sha256(data).hexdigest()

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        hashes = dict(pool.map(download, entries))
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "manifest.json").write_text(json.dumps({
        "repository": REPOSITORY, "revision": REVISION, "license": "Apache-2.0", "files": hashes,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(hashes)} Panda files at {ROOT}")


if __name__ == "__main__":
    main()
