"""Install echoflow_v2_artifacts.zip (from the Colab notebook) into results/ safely.

Steps: open zip → validate every member path (allow-list, no absolute paths / '..' / zip-slip, size cap) →
verify SHA-256 of every file against manifest.json → extract to a temporary directory → load the router
bundle and re-score its check vectors → only then move files into results/. A zip that fails any step
leaves the current models untouched.

Security note: the model bundle is a joblib (pickle) file; loading a pickle can execute code, so only
install artifacts you produced yourself with the EchoFlow-Q notebook.

CLI:  .venv\\Scripts\\python.exe -m src.pipeline.artifacts path\\to\\echoflow_v2_artifacts.zip
"""
from __future__ import annotations

import hashlib
import io
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from src.data.hf_utils import PROJECT_ROOT
from src.diarization.speaker_assigner import ArtifactError, build_largescale_routers

RESULTS = PROJECT_ROOT / "results"
BUNDLE_REL = "models/multilingual/echoflow_v2_routers.joblib"
ALLOWED_PREFIXES = ("models/multilingual/", "metrics/", "figures/")
MAX_TOTAL_BYTES = 1_000_000_000            # 1 GB uncompressed cap


def _safe_members(z: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members, total = [], 0
    for info in z.infolist():
        if info.is_dir():
            continue
        p = PurePosixPath(info.filename)
        if p.is_absolute() or ".." in p.parts or "\\" in info.filename or ":" in info.filename:
            raise ArtifactError(f"unsafe path in zip: {info.filename!r}")
        if info.filename != "manifest.json" and not info.filename.startswith(ALLOWED_PREFIXES):
            raise ArtifactError(f"unexpected file in zip: {info.filename!r} (allowed: manifest.json, {', '.join(ALLOWED_PREFIXES)})")
        total += info.file_size
        if total > MAX_TOTAL_BYTES:
            raise ArtifactError("zip expands beyond the 1 GB limit")
        members.append(info)
    return members


def install_artifacts(src: str | Path | bytes, results_dir: Path = RESULTS) -> dict:
    """Validate + install. Returns a summary dict; raises ArtifactError and changes nothing on failure."""
    data = src if isinstance(src, (bytes, bytearray)) else Path(src).read_bytes()
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as e:
        raise ArtifactError(f"not a zip file: {e}") from e
    with z:
        members = _safe_members(z)
        names = {m.filename for m in members}
        if "manifest.json" not in names or BUNDLE_REL not in names:
            raise ArtifactError(f"zip must contain manifest.json and {BUNDLE_REL}")
        manifest = json.loads(z.read("manifest.json"))
        if manifest.get("format") != "echoflow-artifacts-v2":
            raise ArtifactError("manifest format is not echoflow-artifacts-v2")
        for name, sha in manifest.get("files", {}).items():
            if name not in names:
                raise ArtifactError(f"manifest lists missing file {name}")
            if hashlib.sha256(z.read(name)).hexdigest() != sha:
                raise ArtifactError(f"checksum mismatch for {name}")
        unlisted = names - set(manifest.get("files", {})) - {"manifest.json"}
        if unlisted:
            raise ArtifactError(f"files not covered by the manifest: {sorted(unlisted)}")

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for m in members:
                dest = (tmp / m.filename).resolve()
                if not dest.is_relative_to(tmp.resolve()):
                    raise ArtifactError(f"zip-slip attempt: {m.filename!r}")
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(z.read(m))
            routers = build_largescale_routers(tmp / BUNDLE_REL, verify=True)       # raises on mismatch
            installed = []
            for m in members:
                target = results_dir / m.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(tmp / m.filename, target)
                installed.append(target.relative_to(results_dir).as_posix())
    return {"installed": installed, "n_enrolled": len(routers["rbf_192"].labels), "smoke": manifest.get("smoke", False),
            "summary": manifest.get("summary", {}), "created_utc": manifest.get("created_utc")}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m src.pipeline.artifacts <echoflow_v2_artifacts.zip>")
    out = install_artifacts(sys.argv[1])
    print(json.dumps(out, indent=2, default=str))
