#!/usr/bin/env python3
"""Build a hashable, harmless NoCloud ISO for Loop171 VM acceptance."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Mapping

import pycdlib

SCHEMA = "axon_loop171_harmless_nocloud_fixture_v2"
PYCDLIB_VERSION = "1.16.0"
VOLUME_IDENTIFIER = "CIDATA"
INSTANCE_IDENTIFIER = "axon-loop171-harmless-fixture-v2"
FILES = {
    "meta-data": f"instance-id: {INSTANCE_IDENTIFIER}\nlocal-hostname: axon-loop171-fixture\n".encode("utf-8"),
    "network-config": b"version: 2\nethernets: {}\n",
    "user-data": b"""#cloud-config
write_files:
  - path: /usr/local/sbin/axon-loop171-fixture.sh
    permissions: '0700'
    content: |
      #!/bin/sh
      set -eu
      readonly_media=false
      if findmnt -no OPTIONS /dev/sr0 2>/dev/null | tr ',' '\\n' | grep -qx ro; then readonly_media=true; fi
      write_attempt_blocked=true
      if dd if=/dev/zero of=/dev/sr0 bs=1 count=1 conv=notrunc 2>/dev/null; then write_attempt_blocked=false; fi
      no_default_route=true
      if ip route show default | grep -q .; then no_default_route=false; fi
      only_loopback=true
      if ip -o link show | awk -F': ' '{print $2}' | grep -vx lo | grep -q .; then only_loopback=false; fi
      printf '{"schema":"axon_loop171_fixture_v1","readonly_media":%s,"write_attempt_blocked":%s,"no_default_route":%s,"only_loopback":%s}\\n' "$readonly_media" "$write_attempt_blocked" "$no_default_route" "$only_loopback" > /dev/ttyS0
      poweroff
runcmd:
  - /usr/local/sbin/axon-loop171-fixture.sh
""",
}
ISO_PATHS = {
    "meta-data": "/META_DAT.;1",
    "network-config": "/NETWORK_.;1",
    "user-data": "/USER_DAT.;1",
}


class FixtureBuildError(RuntimeError):
    """Raised when the harmless fixture cannot be created or verified."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_new_file(path: Path) -> None:
    if path.suffix != ".iso":
        raise FixtureBuildError("fixture output must use the .iso suffix")
    if path.exists() or path.is_symlink():
        raise FixtureBuildError("fixture ISO overwrite is forbidden")
    if not path.parent.is_dir():
        raise FixtureBuildError("fixture output parent must already exist")


def _verify_iso(path: Path) -> None:
    with path.open("rb") as handle:
        handle.seek(32768)
        descriptor = handle.read(6)
        if descriptor[1:] != b"CD001":
            raise FixtureBuildError("fixture does not contain an ISO9660 primary volume descriptor")
        handle.seek(32808)
        if handle.read(32).decode("ascii", "strict").strip() != VOLUME_IDENTIFIER:
            raise FixtureBuildError("fixture ISO volume identifier is not CIDATA")

    image = pycdlib.PyCdlib()
    try:
        image.open(str(path))
        for name, expected in FILES.items():
            payload = io.BytesIO()
            image.get_file_from_iso_fp(payload, joliet_path=f"/{name}")
            if payload.getvalue() != expected:
                raise FixtureBuildError(f"fixture content drifted: {name}")
    finally:
        image.close()


def build_fixture(output: Path) -> Mapping[str, object]:
    """Create a new ISO atomically and return only its harmless manifest."""
    if importlib.metadata.version("pycdlib") != PYCDLIB_VERSION:
        raise FixtureBuildError("pycdlib version drifted")
    output = output.resolve(strict=False)
    _require_new_file(output)

    # 先在同卷临时文件中构建并回读，再用硬链接一次性发布，避免覆盖已有验收介质。
    descriptor, temporary_name = tempfile.mkstemp(prefix=".loop171-fixture-", suffix=".iso", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        image = pycdlib.PyCdlib()
        try:
            image.new(interchange_level=3, joliet=3, vol_ident=VOLUME_IDENTIFIER)
            for name, content in FILES.items():
                image.add_fp(io.BytesIO(content), len(content), iso_path=ISO_PATHS[name], joliet_path=f"/{name}")
            image.write(str(temporary))
        finally:
            image.close()
        _verify_iso(temporary)
        os.link(temporary, output)
    except FileExistsError as error:
        raise FixtureBuildError("fixture ISO overwrite is forbidden") from error
    finally:
        temporary.unlink(missing_ok=True)

    return {
        "schema": SCHEMA,
        "pycdlib": {"version": PYCDLIB_VERSION, "license": "LGPL-2.1-only"},
        "volume_identifier": VOLUME_IDENTIFIER,
        "instance_id": INSTANCE_IDENTIFIER,
        "fixture_iso_sha256": _sha256(output),
        "fixture_iso_bytes": output.stat().st_size,
        "files": {name: {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)} for name, content in FILES.items()},
        "hard_boundaries": {
            "contains_sample": False,
            "contains_network_url": False,
            "contains_credentials_or_ssh_keys": False,
            "creates_or_starts_vm": False,
            "sample_access_allowed": False,
            "parser_execution_allowed": False,
            "training_allowed": False,
            "heldout_allowed": False,
            "f1_claim_allowed": False,
        },
        "decision": "harmless_fixture_iso_created_no_vm_action_authorized",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.receipt.exists() or arguments.receipt.is_symlink():
        raise FixtureBuildError("fixture receipt overwrite is forbidden")
    if not arguments.receipt.parent.is_dir():
        raise FixtureBuildError("fixture receipt parent must already exist")
    manifest = build_fixture(arguments.output)
    encoded = (json.dumps(manifest, ensure_ascii=True, sort_keys=True) + "\n").encode("ascii")
    descriptor = os.open(arguments.receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


if __name__ == "__main__":
    main()
