from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from kvd_features import extractor as extractor_module  # noqa: E402
from kvd_features.content_pe_v1 import _section_data_prefix  # noqa: E402
from kvd_features.extractor import ExtractionConfig, PEFeatureExtractor  # noqa: E402


class _FakePefileModule:
    DIRECTORY_ENTRY = {
        "IMAGE_DIRECTORY_ENTRY_IMPORT": 1,
        "IMAGE_DIRECTORY_ENTRY_EXPORT": 2,
        "IMAGE_DIRECTORY_ENTRY_DEBUG": 3,
        "IMAGE_DIRECTORY_ENTRY_BASERELOC": 4,
        "IMAGE_DIRECTORY_ENTRY_TLS": 5,
        "IMAGE_DIRECTORY_ENTRY_EXCEPTION": 6,
        "IMAGE_DIRECTORY_ENTRY_SECURITY": 7,
        "IMAGE_DIRECTORY_ENTRY_RESOURCE": 8,
    }

    def __init__(self, fake_pe):
        self.fake_pe = fake_pe

    def PE(self, _file_path, fast_load=True):
        assert fast_load is True
        return self.fake_pe


def test_fallback_runs_after_pe_close_when_extraction_fails(tmp_path: Path, monkeypatch):
    sample_path = tmp_path / "sample.exe"
    sample_path.write_bytes(b"MZ" + b"\0" * 128)
    fake_pe = SimpleNamespace(closed=False)
    fake_pe.parse_data_directories = lambda directories: None
    fake_pe.close = lambda: setattr(fake_pe, "closed", True)

    monkeypatch.setattr(extractor_module, "PEFILE_AVAILABLE", True)
    monkeypatch.setattr(extractor_module, "pefile", _FakePefileModule(fake_pe))
    extractor = PEFeatureExtractor(
        ExtractionConfig(
            pe_schema_version="fixed_v2",
            pe_feature_dim=150,
            allow_pe_fallback=True,
        )
    )
    monkeypatch.setattr(
        extractor,
        "_extract_fixed_v2_features",
        lambda _pe, _file_path: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    def fake_fallback(_file_path):
        assert fake_pe.closed is True
        return np.ones(extractor.config.pe_feature_dim, dtype=np.float32)

    monkeypatch.setattr(extractor, "_extract_fallback", fake_fallback)

    features = extractor.extract(str(sample_path))

    assert fake_pe.closed is True
    assert features.shape == (150,)


def test_section_entropy_reads_only_sample_size():
    captured_lengths = []

    class FakeSection:
        Name = b".text\x00\x00\x00"
        SizeOfRawData = 10_000
        Misc_VirtualSize = 10_000
        Characteristics = 0x60000000

        def get_data(self, *args, **kwargs):
            captured_lengths.append(kwargs.get("length"))
            return b"\x00\x01" * 128

    fake_pe = SimpleNamespace(
        FILE_HEADER=SimpleNamespace(NumberOfSections=1),
        sections=[FakeSection()],
    )
    extractor = PEFeatureExtractor(ExtractionConfig(section_entropy_min_size=256))

    stats = extractor._collect_section_and_import_stats(fake_pe)

    assert captured_lengths == [256]
    assert len(stats["section_entropies"]) == 1


def test_content_pe_v1_section_prefix_uses_bounded_get_data():
    captured = {}

    class FakeSection:
        SizeOfRawData = 10_000_000

        def get_data(self, *args, **kwargs):
            captured["length"] = kwargs.get("length")
            if "length" not in kwargs:
                raise AssertionError("full section get_data() must not be called")
            return b"\x00\x01" * 128

    payload = _section_data_prefix(FakeSection(), Path("unused.exe"))

    assert len(payload) == 256
    assert captured["length"] == 4096


def test_section_entropy_sample_does_not_fallback_to_full_section_copy():
    class OldPefileSection:
        SizeOfRawData = 512

        def get_data(self, *args, **kwargs):
            if "length" in kwargs:
                raise TypeError("old pefile")
            raise AssertionError("unbounded get_data() fallback should not run")

    extractor = PEFeatureExtractor(ExtractionConfig(section_entropy_min_size=256))

    assert extractor._read_section_entropy_sample(OldPefileSection(), 512) == b""


def test_parse_data_directories_only_requests_consumed_directories(tmp_path: Path, monkeypatch):
    sample_path = tmp_path / "sample.exe"
    sample_path.write_bytes(b"MZ" + b"\0" * 128)
    captured = {}

    class FakePE:
        def parse_data_directories(self, directories):
            captured["directories"] = list(directories)

        def close(self):
            captured["closed"] = True

    fake_pe = FakePE()
    monkeypatch.setattr(extractor_module, "PEFILE_AVAILABLE", True)
    monkeypatch.setattr(extractor_module, "pefile", _FakePefileModule(fake_pe))
    extractor = PEFeatureExtractor(
        ExtractionConfig(
            pe_schema_version="fixed_v2",
            pe_feature_dim=150,
            allow_pe_fallback=False,
        )
    )
    monkeypatch.setattr(
        extractor,
        "_extract_fixed_v2_features",
        lambda _pe, _file_path: np.zeros(extractor.config.pe_feature_dim, dtype=np.float32),
    )

    features = extractor.extract(str(sample_path))

    assert features.shape == (150,)
    assert captured["closed"] is True
    assert captured["directories"] == [1, 3, 4, 5, 6, 7]
    assert 2 not in captured["directories"]
    assert 8 not in captured["directories"]
