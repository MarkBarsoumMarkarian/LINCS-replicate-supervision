#!/usr/bin/env python3
"""Resumable, checksum-verified downloader for GSE70138 Phase II Level 2."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = ROOT / "work/replicate_metric_learning/external_gse70138"
RAW_DIR = RUN_DIR / "raw"
STATUS = RUN_DIR / "download_status.json"
CURL_LOG = RUN_DIR / "curl.log"

FILENAME = "GSE70138_Broad_LINCS_Level2_GEX_n345976x978_2017-03-06.gctx.gz"
URL = f"https://ftp.ncbi.nlm.nih.gov/geo/series/GSE70nnn/GSE70138/suppl/{FILENAME}"
TARGET = RAW_DIR / FILENAME
PARTIAL = RAW_DIR / f"{FILENAME}.part"
EXPECTED_BYTES = 668_884_165
EXPECTED_SHA512 = (
    "8fbb94560545a779141a2fc857998934f4d9bc421024152f28fdd0d3b136d4e"
    "287b120f5876a5b6f4d7f687924f736b15ef92455d9c34e324ec838e3b8add1d6"
)


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(state: str, **fields: object) -> None:
    payload = {"state": state, "updated_at": timestamp(), **fields}
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATUS)


def human_bytes(value: float) -> str:
    units = ["B", "KiB", "MiB", "GiB"]
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GiB"


def verify(path: Path) -> str:
    digest = hashlib.sha512()
    checked = 0
    last_report = time.monotonic()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
            checked += len(block)
            now = time.monotonic()
            if now - last_report >= 5:
                percent = 100 * checked / EXPECTED_BYTES
                print(
                    f"VERIFY {human_bytes(checked)} / {human_bytes(EXPECTED_BYTES)} "
                    f"({percent:.1f}%)",
                    flush=True,
                )
                write_status(
                    "verifying",
                    bytes_checked=checked,
                    total_bytes=EXPECTED_BYTES,
                    percent=round(percent, 2),
                    target=str(TARGET),
                )
                last_report = now
    return digest.hexdigest()


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(ROOT).free
    if free_bytes < 2 * EXPECTED_BYTES:
        raise RuntimeError("less than twice the compressed file size is free")

    if TARGET.exists():
        if TARGET.stat().st_size != EXPECTED_BYTES:
            raise RuntimeError(f"existing target has wrong size: {TARGET.stat().st_size}")
        print("Existing completed-size file found; verifying checksum.", flush=True)
    else:
        initial = PARTIAL.stat().st_size if PARTIAL.exists() else 0
        print(f"START {timestamp()}", flush=True)
        print(f"URL {URL}", flush=True)
        print(f"TARGET {TARGET}", flush=True)
        print(
            f"RESUME_AT {human_bytes(initial)} / {human_bytes(EXPECTED_BYTES)}",
            flush=True,
        )
        write_status(
            "downloading",
            downloaded_bytes=initial,
            total_bytes=EXPECTED_BYTES,
            percent=round(100 * initial / EXPECTED_BYTES, 2),
            target=str(TARGET),
            partial=str(PARTIAL),
            url=URL,
        )
        command = [
            "curl", "-L", "--fail", "--retry", "20", "--retry-all-errors",
            "--continue-at", "-", "--output", str(PARTIAL), URL,
        ]
        started = time.monotonic()
        with CURL_LOG.open("ab") as curl_log:
            process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=curl_log,
            )
            while process.poll() is None:
                time.sleep(5)
                downloaded = PARTIAL.stat().st_size if PARTIAL.exists() else 0
                elapsed = max(time.monotonic() - started, 1e-9)
                transferred = max(downloaded - initial, 0)
                rate = transferred / elapsed
                remaining = max(EXPECTED_BYTES - downloaded, 0)
                eta = remaining / rate if rate > 0 else None
                percent = 100 * downloaded / EXPECTED_BYTES
                eta_text = f"{eta / 60:.1f} min" if eta is not None else "unknown"
                print(
                    f"DOWNLOAD {human_bytes(downloaded)} / {human_bytes(EXPECTED_BYTES)} "
                    f"({percent:.2f}%) rate={human_bytes(rate)}/s eta={eta_text}",
                    flush=True,
                )
                write_status(
                    "downloading",
                    downloaded_bytes=downloaded,
                    total_bytes=EXPECTED_BYTES,
                    percent=round(percent, 2),
                    bytes_per_second=round(rate, 2),
                    eta_seconds=round(eta, 1) if eta is not None else None,
                    target=str(TARGET),
                    partial=str(PARTIAL),
                    url=URL,
                )
        if process.returncode != 0:
            write_status("download_failed", returncode=process.returncode)
            raise RuntimeError(f"curl exited with code {process.returncode}")
        if PARTIAL.stat().st_size != EXPECTED_BYTES:
            raise RuntimeError(
                f"downloaded {PARTIAL.stat().st_size} bytes; expected {EXPECTED_BYTES}"
            )
        PARTIAL.replace(TARGET)

    write_status(
        "verifying",
        bytes_checked=0,
        total_bytes=EXPECTED_BYTES,
        percent=0,
        target=str(TARGET),
    )
    print("DOWNLOAD COMPLETE; STARTING SHA-512 VERIFICATION", flush=True)
    observed = verify(TARGET)
    if observed != EXPECTED_SHA512:
        write_status(
            "checksum_failed",
            expected_sha512=EXPECTED_SHA512,
            observed_sha512=observed,
            target=str(TARGET),
        )
        raise RuntimeError("SHA-512 checksum mismatch")
    write_status(
        "verified",
        downloaded_bytes=EXPECTED_BYTES,
        total_bytes=EXPECTED_BYTES,
        percent=100.0,
        sha512=observed,
        target=str(TARGET),
    )
    print(f"VERIFIED SHA512 {observed}", flush=True)
    print("STATE verified", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_status("failed", error=repr(error))
        print(f"FAILED {error!r}", flush=True)
        raise
