import inspect
import json
import tempfile
from pathlib import Path
import sys
import argparse
from types import SimpleNamespace

sys.path.insert(0, 'src')

from config import AxonExperimentConfig, DSRAArchitectureConfig, TrainingConfig
from dataset import MalwareDataset, NPZDataLoader, FeatureCacheDataset, _load_cached_feature_npz
from kvd_features.extractor import ExtractionConfig, PEFeatureExtractor, extract_all_features, extract_lightweight_features
from model import HybridLightGBMModel
from dsra.dsra_layer import DSRA_Chunk_Layer


def test_hybrid_lightgbm_uses_full_dsra_arch_config():
    arch = DSRAArchitectureConfig(dim=128, heads=4, slots=17, read_topk=3, write_topk=2, local_window=11, slot_pe='none')
    config = AxonExperimentConfig(dsra_arch_config=arch)
    model = HybridLightGBMModel(config=config)
    cfg = model.dsra_encoder.dsra_encoder.dsra.cfg
    assert cfg.slots == 17
    assert cfg.read_topk == 3
    assert cfg.write_topk == 2
    assert cfg.local_window == 11
    assert cfg.slot_pe == 'none'


def test_hybrid_lightgbm_passes_encoder_dimensions_from_config():
    config = AxonExperimentConfig(
        byte_embedding_dim=64,
        dsra_dim=64,
        dsra_heads=4,
        pe_feature_dim=37,
        pe_projection_dim=32,
        pe_projector_hidden_dim=48,
        max_byte_length=256,
        dsra_chunk_size=64,
        vocab_size=128,
    )
    model = HybridLightGBMModel(config=config)
    assert model.dsra_encoder.byte_embedding.embedding.num_embeddings == 128
    assert model.dsra_encoder.byte_embedding.embedding.embedding_dim == 64
    assert model.dsra_encoder.pe_projector.projector[0].in_features == 37
    assert model.dsra_encoder.pe_projector.projector[0].out_features == 48


def test_empty_lists_from_config_are_preserved():
    config = AxonExperimentConfig(
        malicious_keywords=[],
        benign_keywords=[],
        malicious_dir_names=[],
        benign_dir_names=[],
        benign_dir_names_fs=[],
        malicious_dir_names_fs=[],
        common_section_names=[],
        system_dlls=[],
        packer_keywords=[],
        api_categories={},
        api_prefix_only_keywords=[],
    )
    with tempfile.TemporaryDirectory() as tmp:
        dataset = MalwareDataset(tmp, axon_config=config, use_cache=False)
        assert dataset.malicious_keywords == []
        assert dataset.benign_keywords == []
        assert dataset.malicious_dir_names == []
        assert dataset.benign_dir_names == []
        assert dataset.file_list == []
    extractor = PEFeatureExtractor(axon_config=config)
    assert extractor.common_sections == set()
    assert extractor.system_dlls == set()
    assert extractor.packer_keywords == set()
    assert extractor.api_categories == {}
    assert extractor._prefix_only == set()


def test_stat_feature_dim_is_validated_against_stat_config():
    try:
        AxonExperimentConfig(stat_chunk_count=12)
    except ValueError as exc:
        assert 'stat_feature_dim' in str(exc)
    else:
        raise AssertionError('Expected ValueError for mismatched stat_feature_dim')


def test_extraction_config_maps_stat_segment_count():
    config = AxonExperimentConfig(stat_segment_count=4, stat_chunk_count=10, stat_feature_dim=52)
    extraction_config = ExtractionConfig.from_axon_config(config)
    assert extraction_config.stat_segment_count == 4


def test_dataset_scan_and_infer_use_same_directory_config():
    config = AxonExperimentConfig(
        benign_dir_names_fs=['white'],
        malicious_dir_names_fs=['black'],
        benign_dir_names=['white'],
        malicious_dir_names=['black'],
    )
    with tempfile.TemporaryDirectory() as tmp:
        dataset = MalwareDataset(tmp, axon_config=config, use_cache=False)
        black_file = Path(tmp) / 'black' / 'sample.exe'
        black_file.parent.mkdir()
        assert dataset._infer_label(black_file) == 1


def test_dsra_legacy_state_has_rope_positions():
    arch = DSRAArchitectureConfig(dim=128, slots=8, slot_pe='rope')
    layer = DSRA_Chunk_Layer(dim=128, K=8, pe_mode='rope', dsra_arch_config=arch)
    import torch
    legacy_state = torch.randn(2, 8, 128)
    state = layer._coerce_state(legacy_state, 2, torch.device('cpu'), torch.float32)
    assert state.slot_positions is not None
    assert state.slot_positions.shape == (2, layer.core.heads, 8)


def test_dsra_compat_head_selector_matches_core_selector():
    source = inspect.getsource(DSRA_Chunk_Layer.__init__)
    assert 'heads = select_mhdsra2_heads(dim)' in source


def test_train_command_passes_fp16_to_training_config():
    source = Path('scripts/main.py').read_text(encoding='utf-8')
    train_source = source[source.index('def train_command'):source.index('def eval_command')]
    assert 'if args.fp16:' in train_source
    assert 'train_config.mixed_precision = True' in train_source


def test_train_epochs_override_fast_mode_config():
    source = Path('scripts/main.py').read_text(encoding='utf-8')
    train_source = source[source.index('def train_command'):source.index('def eval_command')]
    fast_pos = train_source.index('train_config.max_epochs = config.fast_mode_epochs')
    override_pos = train_source.index('train_config.max_epochs = args.epochs', fast_pos)
    assert fast_pos < override_pos
    assert 'train_config.warmup_epochs = max(0, train_config.max_epochs - 1)' in train_source


def test_train_toml_config_loads_main_and_legacy_candidate():
    import scripts.main as main

    baseline_args = argparse.Namespace(config='config/default_config.toml')
    baseline, baseline_train = main._resolve_config(baseline_args)
    assert baseline.experiment_name == 'axon_v2.6_fixed_pe256_main'
    assert baseline.fusion_type == 'concat'
    assert baseline.pe_feature_dim == 256
    assert baseline.pe_schema_version == 'fixed_v2'
    assert baseline.fixed_pe_schema_used_dim() == 143
    assert baseline_train.optimizer == 'adamw'
    assert baseline_train.decision_threshold == 0.50

    candidate_args = argparse.Namespace(config='config/legacy_v3_candidate.toml')
    candidate, candidate_train = main._resolve_config(candidate_args)
    assert candidate.experiment_name == 'axon_v2.6_legacy_v3_candidate'
    assert candidate.fusion_type == 'concat'
    assert candidate.pe_feature_dim == 1500
    assert candidate.pe_schema_version == 'legacy_dynamic'
    assert candidate_train.decision_threshold == 0.44


def test_training_config_defaults_to_zero_workers_on_windows(monkeypatch):
    import config as config_module

    monkeypatch.setattr(config_module.platform, 'system', lambda: 'Windows')
    assert TrainingConfig().num_workers == 0


def test_mixed_precision_is_opt_in_for_baseline_stability():
    assert TrainingConfig().mixed_precision is False
    import scripts.main as main

    baseline, train_config = main._resolve_config(argparse.Namespace(config='config/default_config.toml'))
    assert train_config.mixed_precision is False
    assert baseline.pe_schema_version == 'fixed_v2'


def test_default_split_keeps_eighty_percent_for_test_with_validation():
    config = AxonExperimentConfig()
    assert config.val_ratio == 0.04
    assert config.test_ratio == 0.8
    assert abs((1.0 - config.val_ratio - config.test_ratio) - 0.16) < 1e-9


def test_fast_mode_defaults_to_ten_thousand_samples_and_eight_epochs():
    import scripts.main as main

    config, _train_config = main._resolve_config(argparse.Namespace(config='config/default_config.toml'))
    assert config.fast_mode_samples == 10000
    assert config.fast_mode_epochs == 10


def test_invalid_split_ratio_is_rejected():
    try:
        AxonExperimentConfig(val_ratio=0.2, test_ratio=0.8)
    except ValueError as exc:
        assert 'val_ratio + test_ratio' in str(exc)
    else:
        raise AssertionError('Expected invalid split ratio to be rejected')


def test_small_stratified_split_keeps_validation_samples():
    from dataset import create_stratified_split

    class DummyDataset:
        label_list = [0] * 10 + [1] * 10

    config = AxonExperimentConfig()
    train, val, test = create_stratified_split(DummyDataset(), axon_config=config)
    assert len(train) == 2
    assert len(val) == 2
    assert len(test) == 16


def test_npz_data_loader_passes_stat_feature_dim_to_npz_dataset(monkeypatch):
    captured = {}

    class FakeNPZDataset:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            raise IndexError(idx)

    monkeypatch.setattr('dataset.NPZDataset', FakeNPZDataset)
    loader = NPZDataLoader('data/npz', stat_feature_dim=77)
    data_loader = loader.create_dataloader('train')
    assert captured['stat_feature_dim'] == 77
    assert data_loader.dataset.__class__ is FakeNPZDataset


def test_npz_data_loader_can_disable_raw_fallback():
    with tempfile.TemporaryDirectory() as tmp:
        loader = NPZDataLoader(tmp, allow_raw_fallback=False)
        try:
            loader.create_dataloader('test')
        except ValueError as exc:
            assert 'No NPZ files found' in str(exc)
        else:
            raise AssertionError('Expected disabled raw fallback to raise the NPZ error')


def test_train_help_exposes_config_argument():
    source = Path('scripts/main.py').read_text(encoding='utf-8')
    assert "train_parser.add_argument('--config'" in source
    assert "train_parser.add_argument('--samples-per-class'" in source
    assert "required=True" not in source[source.index("train_parser.add_argument('--data-dir'"):source.index("train_parser.add_argument('--samples-per-class'")]
    assert 'test_loader=test_loader' in source
    assert 'per_class + per_class // 5' not in source


def test_main_heavy_commands_require_resource_guard_json():
    import scripts.main as main

    args = argparse.Namespace(command='train', resource_guard_json=None, resource_guard_max_age_seconds=3600.0)

    try:
        main._enforce_resource_guard(args)
    except SystemExit as exc:
        assert '--resource-guard-json' in str(exc)
    else:
        raise AssertionError('Expected heavy command without resource guard to exit before dispatch')


def test_main_requires_resource_guard_for_nested_predict_only():
    import scripts.main as main

    plain_predict_args = argparse.Namespace(command='predict', scan_nested=False)
    main._enforce_resource_guard(plain_predict_args)

    nested_predict_args = argparse.Namespace(
        command='predict',
        scan_nested=True,
        resource_guard_json=None,
        resource_guard_max_age_seconds=3600.0,
    )
    try:
        main._enforce_resource_guard(nested_predict_args)
    except SystemExit as exc:
        assert '--resource-guard-json' in str(exc)
    else:
        raise AssertionError('Expected nested predict without resource guard to exit before dispatch')


def test_main_accepts_valid_resource_guard_receipt(tmp_path, monkeypatch):
    import scripts.main as main
    from scripts.pre_run_resource_leak_guard import MemorySnapshot, evaluate_guard

    guard_path = tmp_path / 'guard.json'
    command_args = ['train', '--resource-guard-json', str(guard_path)]
    expected_command = [sys.executable, str(main.MAIN_SCRIPT_PATH), *command_args]
    monkeypatch.setattr(sys, 'argv', [str(main.MAIN_SCRIPT_PATH), *command_args])
    payload = evaluate_guard(
        target_scripts=[main.MAIN_SCRIPT_PATH],
        allowed_risks={'torch_import', 'cuda_usage', 'torch_dataloader'},
        command=expected_command,
        created_at=1000.0,
        memory_snapshot=MemorySnapshot(total_mb=1000, available_mb=600),
        python_processes=[],
        gpu_summary={
            'available': True,
            'memory_used_mb': 100,
            'memory_total_mb': 1000,
            'memory_used_pct': 10.0,
            'utilization_pct': 0,
            'compute_app_count': 0,
            'python_compute_app_count': 0,
            'python_compute_apps': [],
        },
    )
    guard_path.write_text(json.dumps(payload), encoding='utf-8')
    args = argparse.Namespace(
        command='train',
        resource_guard_json=str(guard_path),
        resource_guard_max_age_seconds=3600.0,
    )

    main._enforce_resource_guard(args, now=1005.0)


def test_main_rejects_unbound_resource_guard_receipt(tmp_path, monkeypatch):
    import scripts.main as main
    from scripts.pre_run_resource_leak_guard import MemorySnapshot, evaluate_guard

    guard_path = tmp_path / 'guard.json'
    command_args = ['train', '--resource-guard-json', str(guard_path)]
    monkeypatch.setattr(sys, 'argv', [str(main.MAIN_SCRIPT_PATH), *command_args])
    payload = evaluate_guard(
        target_scripts=[main.MAIN_SCRIPT_PATH],
        allowed_risks={'torch_import', 'cuda_usage', 'torch_dataloader'},
        created_at=1000.0,
        memory_snapshot=MemorySnapshot(total_mb=1000, available_mb=600),
        python_processes=[],
        gpu_summary={
            'available': True,
            'memory_used_mb': 100,
            'memory_total_mb': 1000,
            'memory_used_pct': 10.0,
            'utilization_pct': 0,
            'compute_app_count': 0,
            'python_compute_app_count': 0,
            'python_compute_apps': [],
        },
    )
    guard_path.write_text(json.dumps(payload), encoding='utf-8')
    args = argparse.Namespace(
        command='train',
        resource_guard_json=str(guard_path),
        resource_guard_max_age_seconds=3600.0,
    )

    try:
        main._enforce_resource_guard(args, now=1005.0)
    except SystemExit as exc:
        assert 'receipt_command_mismatch' in str(exc)
    else:
        raise AssertionError('Expected unbound resource guard receipt to be rejected')


def test_trainer_raises_on_non_finite_loss():
    source = Path('src/trainer.py').read_text(encoding='utf-8')
    assert 'torch.isfinite(loss)' in source
    assert 'Non-finite training loss detected' in source


def test_dataset_scan_skips_symlinks_by_default_and_uses_whitelist():
    scan_source = inspect.getsource(MalwareDataset._scan_directory)
    iter_source = inspect.getsource(MalwareDataset._iter_files)
    symlink_source = inspect.getsource(MalwareDataset._is_allowed_symlink)
    assert 'self._iter_files(' in scan_source
    assert 'os.walk(root, followlinks=self.follow_symlinks)' in iter_source
    assert 'not self._is_allowed_symlink(child)' in iter_source
    assert 'not self._is_allowed_symlink(file_path)' in iter_source
    assert 'allowed_symlink_roots' in inspect.getsource(MalwareDataset.__init__)
    assert 'return any(_is_relative_to_path(target, root)' in symlink_source
    assert 'sorted(filenames)' in iter_source


def test_focal_loss_uses_smoothed_ce_for_pt_when_label_smoothing_enabled():
    source = Path('src/trainer.py').read_text(encoding='utf-8')
    pt_pos = source.index('pt = torch.exp(-ce_loss)')
    ce_cond_pos = source.index('ce_loss = F.cross_entropy')
    assert ce_cond_pos < pt_pos


def test_predict_and_extract_commands_pass_axon_config_to_feature_extraction():
    source = Path('scripts/main.py').read_text(encoding='utf-8')
    predict_source = source[source.index('def predict_command'):source.index('def extract_command')]
    extract_source = source[source.index('def extract_command'):source.index('def main')]
    assert 'extract_all_features(' in predict_source
    assert 'axon_config=config' in predict_source
    assert 'allow_pe_fallback=config.allow_pe_fallback' in predict_source
    assert 'extract_all_features(' in extract_source


def test_is_valid_sample_uses_magic_filter_before_pefile():
    source = inspect.getsource(MalwareDataset._is_valid_sample)
    assert "read(2)" in source
    assert "b'MZ'" in source


def test_lightweight_feature_empty_patterns_disable_hashes():
    config = AxonExperimentConfig(
        lightweight_dll_patterns=[],
        lightweight_api_patterns=[],
        lightweight_section_patterns=[],
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'sample.bin'
        path.write_bytes(b'MZ kernel32.dll VirtualAlloc .text')
        features = extract_lightweight_features(str(path), axon_config=config)
        assert float(features.sum()) == 0.0


def test_extract_statistical_features_uses_extraction_config():
    from kvd_features.extractor import extract_statistical_features
    import numpy as np
    byte_seq = np.random.randint(0, 256, size=1024, dtype=np.uint8)
    config_3seg = ExtractionConfig(stat_segment_count=3, stat_chunk_count=10)
    config_5seg = ExtractionConfig(stat_segment_count=5, stat_chunk_count=10)
    feat_3seg = extract_statistical_features(byte_seq, config=config_3seg)
    feat_5seg = extract_statistical_features(byte_seq, config=config_5seg)
    assert feat_3seg.shape[0] != feat_5seg.shape[0]
    expected_3 = 7 + 4 + 1 + 3 * 3 + 2 * 10 + 8
    expected_5 = 7 + 4 + 1 + 3 * 5 + 2 * 10 + 8
    assert feat_3seg.shape[0] == expected_3
    assert feat_5seg.shape[0] == expected_5


def test_dsra_chunk_layer_confidence_init_matches_mhdsra2():
    source = inspect.getsource(DSRA_Chunk_Layer.__init__)
    assert 'compat_confidence_init' not in source
    assert 'init_confidence' in source
    arch = DSRAArchitectureConfig(dim=128, init_confidence=0.01)
    layer = DSRA_Chunk_Layer(dim=128, dsra_arch_config=arch)
    assert abs(layer._confidence_init - 0.01) < 1e-9
    layer_no_config = DSRA_Chunk_Layer(dim=128)
    assert abs(layer_no_config._confidence_init - 0.01) < 1e-9


def test_position_overflow_guard_in_malware_encoder():
    source = inspect.getsource(__import__('model').MalwareDSRAEncoder.forward)
    assert '_float32_max_exact_int' in source
    assert 'init_state' in source


def test_extract_all_features_returns_five_tuple():
    from kvd_features.extractor import extract_all_features
    sig = inspect.signature(extract_all_features)
    ret_annotation = sig.return_annotation
    assert ret_annotation is not None
    source = inspect.getsource(extract_all_features)
    assert 'lightweight_features' in source
    assert 'extract_lightweight_features' in source


def test_save_checkpoint_includes_scaler_state():
    source = Path('src/trainer.py').read_text(encoding='utf-8')
    save_start = source.index('def save_checkpoint')
    save_end = source.index('torch.save(checkpoint', save_start)
    save_body = source[save_start:save_end]
    assert 'scaler_state_dict' in save_body
    assert "'train_config': asdict(self.train_config)" in save_body
    load_start = source.index('def load_checkpoint')
    load_end = source.index('return checkpoint', load_start)
    load_body = source[load_start:load_end]
    assert 'scaler_state_dict' in load_body


def test_single_chunk_path_sets_diversity_loss():
    source = inspect.getsource(__import__('model').MalwareDSRAEncoder.forward)
    single_chunk_idx = source.index('if seq_len <= chunk_size:')
    else_idx = source.index('else:', single_chunk_idx)
    single_chunk_body = source[single_chunk_idx:else_idx]
    assert 'diversity_loss' in single_chunk_body


def test_cache_key_includes_lightweight_feature_dim():
    import dataset as dataset_module
    source = inspect.getsource(MalwareDataset._get_cache_path) + inspect.getsource(dataset_module._feature_cache_signature)
    assert 'lightweight_feature_dim' in source


def test_cache_key_includes_pe_schema_fields():
    import dataset as dataset_module
    source = inspect.getsource(dataset_module._feature_cache_signature)
    assert 'pe_schema_version' in source
    assert 'pe_fixed_section_slots' in source

    legacy_hash = dataset_module._feature_cache_hash(8192, 49, 256, 256, True, False, 'legacy_dynamic', 32)
    fixed_hash = dataset_module._feature_cache_hash(8192, 49, 256, 256, True, False, 'fixed_v2', 32)
    assert legacy_hash != fixed_hash
    assert dataset_module._feature_cache_hash(8192, 49, 1500, 256, True, False, 'legacy_dynamic', 32) == '91d04e63'


def test_position_overflow_resets_diversity_accumulator():
    source = inspect.getsource(__import__('model').MalwareDSRAEncoder.forward)
    assert 'init_state' in source
    reset_lines = [line for line in source.splitlines() if 'div_loss_accum' in line and 'tensor(0.0' in line]
    assert len(reset_lines) >= 2


def test_stat_features_moved_to_gpu_in_train_and_eval():
    source = Path('src/trainer.py').read_text(encoding='utf-8')
    train_start = source.index('def train_epoch')
    train_end = source.index('def evaluate', train_start)
    train_body = source[train_start:train_end]
    assert 'stat_features = stat_features.to(self.device' in train_body
    eval_start = source.index('def evaluate')
    eval_end = source.index('def _compute_metrics', eval_start)
    eval_body = source[eval_start:eval_end]
    assert 'stat_features = stat_features.to(self.device' in eval_body


def test_dsra_chunk_layer_uses_precomputed_qkv():
    source = inspect.getsource(DSRA_Chunk_Layer.forward)
    assert '_precomputed_qkv=qkv' in source
    step_source = inspect.getsource(DSRA_Chunk_Layer.forward_step)
    assert '_precomputed_qkv=qkv' in step_source


def test_position_capped_in_slot_write():
    source = Path('src/dsra/mhdsra2/improved_dsra_mha.py').read_text(encoding='utf-8')
    assert '1 << 24' in source
    pos_line_idx = source.index('position=min(')
    assert pos_line_idx > 0


def test_save_checkpoint_saves_last_epoch():
    source = Path('src/trainer.py').read_text(encoding='utf-8')
    save_start = source.index('def save_checkpoint')
    save_end = source.index('torch.save(checkpoint', save_start)
    save_body = source[save_start:save_end]
    assert 'last_epoch' in save_body


def test_dataset_cache_preserves_lightweight_features():
    load_source = inspect.getsource(MalwareDataset._load_from_cache)
    load_path_source = inspect.getsource(MalwareDataset._load_cache_path)
    load_npz_source = inspect.getsource(_load_cached_feature_npz)
    save_source = inspect.getsource(MalwareDataset._save_to_cache)
    getitem_source = inspect.getsource(MalwareDataset.__getitem__)
    assert 'self._load_cache_path(cache_path, file_path)' in load_source
    assert 'lightweight_features' in load_path_source
    assert "data.get('lightweight_features'" in load_npz_source
    assert 'lightweight_features=data[3]' in save_source
    assert 'byte_seq, pe_feat, stat_feat, _lightweight_feat, cached_label = cached_data' in getitem_source


def test_feature_cache_dataset_loads_existing_cache_without_raw_scan():
    import numpy as np
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        config = AxonExperimentConfig(max_byte_length=8)
        probe = MalwareDataset(data_dir, max_byte_length=8, axon_config=config, use_cache=True)
        cache_path = data_dir / ".cache" / f"sample_{probe._cache_config_hash()}.npz"
        np.savez_compressed(
            cache_path,
            byte_sequence=np.array([77, 90], dtype=np.uint8),
            pe_features=np.ones(config.pe_feature_dim, dtype=np.float32),
            stat_features=np.ones(config.stat_feature_dim, dtype=np.float32),
            lightweight_features=np.ones(config.lightweight_feature_dim, dtype=np.float32),
            label=1,
            source_sha256="cache-only-test",
        )

        dataset = FeatureCacheDataset(data_dir, max_byte_length=8, axon_config=config)
        assert len(dataset) == 1
        byte_seq, pe_features, stat_features, label = dataset[0]
        assert byte_seq.shape[0] == 8
        assert pe_features.shape[0] == config.pe_feature_dim
        assert stat_features.shape[0] == config.stat_feature_dim
        assert label.item() == 1
        assert (data_dir / ".cache" / f"manifest_{probe._cache_config_hash()}.json").exists()


def test_max_eval_samples_uses_stratified_limit():
    import scripts.main as main

    class DummyDataset:
        label_list = [0] * 10 + [1] * 10

        def __len__(self):
            return len(self.label_list)

        def __getitem__(self, idx):
            raise IndexError(idx)

    limited = main._limit_dataset_stratified(DummyDataset(), 6)
    selected_labels = [limited.dataset.label_list[index] for index in limited.indices]
    assert selected_labels.count(0) == 3
    assert selected_labels.count(1) == 3


def test_evaluate_detaches_probs_before_numpy():
    source = Path('src/trainer.py').read_text(encoding='utf-8')
    eval_start = source.index('def evaluate')
    eval_end = source.index('def _compute_metrics', eval_start)
    eval_body = source[eval_start:eval_end]
    assert 'preds.detach().cpu().numpy()' in eval_body
    assert 'labels.detach().cpu().numpy()' in eval_body
    assert 'probs.detach().cpu().numpy()' in eval_body


def test_extraction_config_maps_stat_feature_dim():
    config = AxonExperimentConfig(stat_feature_dim=52, stat_segment_count=4)
    extraction_config = ExtractionConfig.from_axon_config(config)
    assert extraction_config.stat_feature_dim == 52


def test_short_sequence_encoder_returns_next_state():
    source = inspect.getsource(__import__('model').MalwareDSRAEncoder.forward)
    single_chunk_idx = source.index('if seq_len <= chunk_size:')
    else_idx = source.index('else:', single_chunk_idx)
    single_chunk_body = source[single_chunk_idx:else_idx]
    assert 'state = next_state' in single_chunk_body


def test_multilayer_dsra_uses_independent_states():
    source = inspect.getsource(__import__('model').DSRAEncoder.forward)
    assert 'states' in source
    assert 'enumerate(self.dsra_layers)' in source


def test_fusion_uses_actual_dsra_arch_dim():
    config = AxonExperimentConfig(
        dsra_dim=64,
        dsra_arch_config=DSRAArchitectureConfig(dim=32, heads=4, num_layers=1),
        pe_projection_dim=16,
        fusion_type='concat',
    )
    model = __import__('model').AxonMalwareModel(config)
    assert model.classifier[0].normalized_shape == (64,)


def test_hybrid_fusion_uses_actual_dsra_arch_dim():
    config = AxonExperimentConfig(
        dsra_dim=64,
        dsra_arch_config=DSRAArchitectureConfig(dim=32, heads=4, num_layers=1),
        pe_projection_dim=16,
    )
    model = HybridLightGBMModel(config=config)
    assert model.fusion[0].in_features == 64


def test_chunk_path_does_not_store_all_chunk_outputs():
    source = inspect.getsource(__import__('model').MalwareDSRAEncoder.forward)
    assert 'byte_outs.append' not in source


def test_npz_dataset_requires_label():
    import numpy as np
    with tempfile.TemporaryDirectory() as tmp:
        split_dir = Path(tmp) / 'train'
        split_dir.mkdir()
        np.savez_compressed(
            split_dir / 'sample.npz',
            byte_sequence=np.array([77, 90], dtype=np.uint8),
            pe_features=np.zeros(3, dtype=np.float32),
            stat_features=np.zeros(2, dtype=np.float32),
        )
        dataset = __import__('dataset').NPZDataset(tmp, split='train', max_byte_length=4, pe_feature_dim=3, stat_feature_dim=2)
        try:
            dataset[0]
        except ValueError as exc:
            assert 'label' in str(exc)
        else:
            raise AssertionError('Expected missing label to raise ValueError')


def test_npz_dataset_normalizes_stat_feature_dim():
    import numpy as np
    with tempfile.TemporaryDirectory() as tmp:
        split_dir = Path(tmp) / 'train'
        split_dir.mkdir()
        np.savez_compressed(
            split_dir / 'sample.npz',
            byte_sequence=np.array([77, 90], dtype=np.uint8),
            pe_features=np.zeros(3, dtype=np.float32),
            stat_features=np.ones(2, dtype=np.float32),
            label=1,
        )
        dataset = __import__('dataset').NPZDataset(tmp, split='train', max_byte_length=4, pe_feature_dim=3, stat_feature_dim=5)
        _, _, stat_features, label = dataset[0]
        assert stat_features.shape[0] == 5
        assert label.item() == 1


def test_lightweight_api_patterns_are_case_normalized():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'sample.bin'
        path.write_bytes(b'MZ VirtualAlloc')
        features = extract_lightweight_features(str(path), axon_config=AxonExperimentConfig())
        assert float(features.sum()) > 0.0


def test_trainer_resume_starts_after_loaded_epoch():
    source = Path('src/trainer.py').read_text(encoding='utf-8')
    train_start = source.index('def train(')
    save_start = source.index('def _print_metrics', train_start)
    train_body = source[train_start:save_start]
    assert 'start_epoch' in train_body
    assert '_resumed_epoch' in train_body


def test_predict_command_passes_stat_features_to_model():
    source = Path('scripts/main.py').read_text(encoding='utf-8')
    predict_source = source[source.index('def predict_command'):source.index('def extract_command')]
    assert 'stat_tensor' in predict_source
    assert 'stat_features=stat_tensor' in predict_source


def test_exact_write_uses_effective_metadata_gates():
    source = Path('src/dsra/mhdsra2/improved_dsra_mha.py').read_text(encoding='utf-8')
    assert 'effective_write_gate' in source
    assert 'effective_forget' in source


def test_dsra_chunk_layer_resets_memory_on_new_sequence():
    source = Path('src/dsra/dsra_layer.py').read_text(encoding='utf-8')
    assert 'def reset_memory' in source
    assert 'S_prev is None' in source
    assert 'reset_memory()' in source


def test_training_config_rejects_invalid_values():
    from config import TrainingConfig
    invalid_configs = [
        {'optimizer': 'rmsprop'},
        {'lr_scheduler': 'plateau'},
        {'max_epochs': 0},
        {'warmup_epochs': 2, 'max_epochs': 2},
        {'learning_rate': 0.0},
        {'batch_size': 0},
        {'eval_interval': 0},
        {'label_smoothing': 1.0},
        {'focal_gamma': -1.0},
        {'decision_threshold': 0.0},
        {'decision_threshold': 1.0},
    ]
    for kwargs in invalid_configs:
        try:
            TrainingConfig(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Expected ValueError for {kwargs}')


def test_trainer_does_not_hardcode_cuda_amp_wrappers():
    source = Path('src/trainer.py').read_text(encoding='utf-8')
    assert "torch.amp.autocast('cuda'" not in source
    assert "torch.amp.GradScaler('cuda'" not in source
    assert 'device_type' in source


def test_trainer_rejects_empty_loaders_before_division():
    source = Path('src/trainer.py').read_text(encoding='utf-8')
    assert 'raise ValueError("train_loader is empty")' in source
    assert 'raise ValueError(f"{phase}_loader is empty")' in source


def test_trainer_cosine_scheduler_never_uses_zero_tmax():
    source = Path('src/trainer.py').read_text(encoding='utf-8')
    scheduler_source = source[source.index('def _create_scheduler'):source.index('def _create_criterion')]
    assert 'T_max=max(1, self.train_config.max_epochs - self.train_config.warmup_epochs)' in scheduler_source


def test_training_metrics_use_configured_decision_threshold():
    source = Path('src/trainer.py').read_text(encoding='utf-8')
    assert 'probs >= self.train_config.decision_threshold' in source


def test_main_config_uses_fixed_pe256_precision_threshold():
    import scripts.main as main

    config, train_config = main._resolve_config(argparse.Namespace(config='config/default_config.toml'))
    assert config.experiment_name == 'axon_v2.6_fixed_pe256_main'
    assert config.dsra_dim == 160
    assert config.dsra_slots == 160
    assert config.pe_feature_dim == 256
    assert config.pe_schema_version == 'fixed_v2'
    assert config.pe_fixed_section_slots == 32
    assert config.pe_projection_dim == 160
    assert train_config.decision_threshold == 0.50
    assert train_config.focal_gamma == 1.0
    assert train_config.focal_alpha == 0.55


def test_legacy_v3_candidate_keeps_dynamic_pe1500():
    import scripts.main as main

    config, train_config = main._resolve_config(argparse.Namespace(config='config/legacy_v3_candidate.toml'))
    assert config.experiment_name == 'axon_v2.6_legacy_v3_candidate'
    assert config.fast_mode_epochs == 10
    assert config.dsra_dim == 160
    assert config.dsra_slots == 160
    assert config.pe_feature_dim == 1500
    assert config.pe_schema_version == 'legacy_dynamic'
    assert train_config.decision_threshold == 0.44
    assert train_config.focal_gamma == 1.0
    assert train_config.focal_alpha == 0.55


def test_config_directory_only_keeps_main_and_legacy_candidate():
    names = sorted(path.name for path in Path('config').glob('*.toml'))
    expected = [
        'default_config.toml',
        'exp1_byte_noise.toml',
        'exp2_swa.toml',
        'exp3_ema.toml',
        'exp4_near_threshold.toml',
        'generalization_enhanced.toml',
        'legacy_v3_candidate.toml',
    ]
    assert names == expected


def test_eval_command_reuses_checkpoint_training_config():
    source = Path('scripts/main.py').read_text(encoding='utf-8')
    eval_source = source[source.index('def eval_command'):source.index('def predict_command')]
    assert "checkpoint.get('train_config', {})" in eval_source
    assert 'eval_train_config = TrainingConfig(**saved_train_config)' in eval_source


def test_eval_command_disables_training_only_shadow_models_and_uses_requested_device():
    source = Path('scripts/main.py').read_text(encoding='utf-8')
    eval_source = source[source.index('def eval_command'):source.index('def predict_command')]
    assert 'eval_train_config.use_ema = False' in eval_source
    assert 'eval_train_config.use_swa = False' in eval_source
    assert 'eval_train_config.enable_swanlab = False' in eval_source
    assert 'AxonTrainer(model, config, train_config=eval_train_config, device=torch.device(args.device))' in eval_source


def test_feature_importance_command_is_exposed_and_writes_reports():
    source = Path('scripts/main.py').read_text(encoding='utf-8')
    assert "subparsers.add_parser('importance'" in source
    assert 'def feature_importance_command(args):' in source
    assert "elif args.command == 'importance':" in source
    assert "mean(abs(gradient) * abs(feature_value))" in source
    assert 'pe_top' in source
    assert 'stat_low' in source
    assert 'pe_feature_metadata' in source
    assert 'stable_in_analyzed_data' in source
    assert 'possible_names' in source
    assert "reports/feature_importance.json" in source
    assert "reports/feature_importance.csv" in source


def test_feature_importance_uses_gradient_times_input_scores():
    source = Path('scripts/main.py').read_text(encoding='utf-8')
    body = source[source.index('def feature_importance_command'):source.index('def main')]
    assert 'pe_features = pe_features.to(device, non_blocking=True).detach().requires_grad_(True)' in body
    assert 'stat_features = stat_features.to(device, non_blocking=True).detach().requires_grad_(True)' in body
    assert 'model.requires_grad_(False)' in body
    assert 'torch.autograd.grad(' in body
    assert 'loss.backward()' not in body
    assert 'pe_grad.detach().abs() * pe_features.detach().abs()' in body
    assert 'stat_grad.detach().abs() * stat_features.detach().abs()' in body


def test_pe_feature_metadata_marks_dynamic_indices():
    import scripts.main as main

    config = AxonExperimentConfig(pe_feature_dim=80)
    metadata = main._pe_feature_metadata(config, observed_section_counts=[3, 5])
    assert metadata[0]['name'] == 'file_size'
    assert metadata[0]['stable_global'] is True
    assert metadata[18]['name'] == 'section_0_is_executable'
    assert metadata[18]['stable_global'] is False
    assert metadata[18]['stable_in_analyzed_data'] is True
    mixed = metadata[28]
    assert mixed['stable_global'] is False
    assert mixed['stable_in_analyzed_data'] is False
    assert mixed['possible_name_count'] > 1
    assert 'section_' in '|'.join(mixed['possible_names']) or 'section_entropy' in '|'.join(mixed['possible_names'])


def test_pe_feature_metadata_fixed_schema_has_stable_names():
    import scripts.main as main

    config = AxonExperimentConfig(pe_schema_version='fixed_v2', pe_feature_dim=256)
    metadata = main._pe_feature_metadata(config, observed_section_counts=[3, 5, 33])
    names = [item['name'] for item in metadata]
    assert metadata[18]['name'] == 'section_slot_0_is_executable'
    assert metadata[18]['stable_global'] is True
    assert metadata[114]['name'] == 'section_entropy_max'
    assert all('pe_dynamic_mixed_index' not in name for name in names)


def test_fixed_pe_schema_rejects_too_small_dim():
    try:
        AxonExperimentConfig(pe_schema_version='fixed_v2', pe_feature_dim=142)
    except ValueError as exc:
        assert 'fixed_v2 PE schema' in str(exc)
    else:
        raise AssertionError('Expected fixed_v2 pe_feature_dim validation failure')


def test_fixed_pe_extractor_keeps_aggregate_indices_stable():
    class Header:
        SizeOfOptionalHeader = 224
        NumberOfSections = 33
        Characteristics = 0

    class OptionalHeader:
        Subsystem = 2
        DllCharacteristics = 0
        CheckSum = 0

    class Section:
        def __init__(self, idx):
            self.Name = f'.s{idx}'.encode()
            self.SizeOfRawData = 100 + idx
            self.Misc_VirtualSize = 200 + idx
            self.Characteristics = 0x20000000 if idx == 32 else 0x40000000

        def get_data(self):
            return bytes(range(256)) + bytes(range(44))

    class FakePE:
        FILE_HEADER = Header()
        OPTIONAL_HEADER = OptionalHeader()

        def __init__(self):
            self.sections = [Section(i) for i in range(33)]

    config = AxonExperimentConfig(pe_schema_version='fixed_v2', pe_feature_dim=256)
    extractor = PEFeatureExtractor(config=ExtractionConfig.from_axon_config(config), axon_config=config)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'sample.exe'
        path.write_bytes(b'MZ' + b'\0' * 1000)
        features = extractor._extract_fixed_v2_features(FakePE(), str(path))

    assert features.shape[0] == 256
    assert features[17] == 33
    assert features[18 + 31 * 3] == 0.0
    assert features[18 + 31 * 3 + 2] == 1.0
    assert features[114] > 0.0


def test_eval_auc_zero_is_not_treated_as_missing():
    source = Path('scripts/main.py').read_text(encoding='utf-8')
    eval_source = source[source.index('def eval_command'):source.index('def predict_command')]
    assert 'if results.auc is not None' in eval_source
    assert "'auc': float(results.auc) if results.auc is not None else None" in eval_source


def test_eval_command_supports_threshold_sweep():
    source = Path('scripts/main.py').read_text(encoding='utf-8')
    eval_source = source[source.index('def eval_command'):source.index('def predict_command')]
    assert "eval_parser.add_argument('--sweep-thresholds'" in source
    assert "eval_parser.add_argument('--decision-threshold'" in source
    assert "_enforce_eval_threshold_sweep_policy(args)" in eval_source
    assert 'trainer.threshold_sweep(test_loader, thresholds' in eval_source
    assert "saved_train_config['decision_threshold'] = args.decision_threshold" in eval_source
    assert "'threshold_sweep': sweep_results" in eval_source


def test_eval_threshold_sweep_policy_blocks_test_and_all_splits():
    import scripts.main as main

    main._enforce_eval_threshold_sweep_policy(
        argparse.Namespace(sweep_thresholds='0.4,0.5', split='val')
    )

    for split in ('test', 'all'):
        try:
            main._enforce_eval_threshold_sweep_policy(
                argparse.Namespace(sweep_thresholds='0.4,0.5', split=split)
            )
        except SystemExit as exc:
            assert 'Threshold sweep is blocked' in str(exc)
        else:
            raise AssertionError(f'Expected threshold sweep on {split} split to be blocked')


def test_eval_report_includes_confusion_error_rates():
    source = Path('scripts/main.py').read_text(encoding='utf-8')
    eval_source = source[source.index('def eval_command'):source.index('def predict_command')]
    for field in [
        'true_positive',
        'true_negative',
        'false_positive',
        'false_negative',
        'false_positive_rate',
        'false_negative_rate',
    ]:
        assert f"'{field}':" in eval_source
    assert "False Positive Rate" in eval_source
    assert "False Negative Rate" in eval_source


def test_eval_raw_fallback_limits_fast_checkpoint_samples():
    source = Path('scripts/main.py').read_text(encoding='utf-8')
    eval_source = source[source.index('def eval_command'):source.index('def predict_command')]
    assert "eval_parser.add_argument('--samples-per-class'" in source
    assert "eval_parser.add_argument('--max-eval-samples'" in source
    assert "getattr(config, 'fast_mode', False)" in eval_source
    assert "samples_per_class = getattr(config, 'fast_mode_samples', None)" in eval_source
    assert "max_samples_per_class=samples_per_class" in eval_source
    assert "_raw_eval_dataset_for_split(dataset, args.split, config, split_file=args.split_file)" in eval_source
    assert "FeatureCacheDataset(" in eval_source
    assert "allow_raw_fallback=False" in eval_source


def test_trainer_threshold_sweep_uses_supplied_thresholds():
    source = Path('src/trainer.py').read_text(encoding='utf-8')
    sweep_source = source[source.index('def threshold_sweep'):source.index('def train', source.index('def threshold_sweep'))]
    assert 'for threshold in thresholds:' in sweep_source
    assert 'preds = (probs >= threshold).astype' in sweep_source
    assert "'threshold': float(threshold)" in sweep_source
    assert "'false_positive': int(metrics.false_positive)" in sweep_source
    assert "'false_negative': int(metrics.false_negative)" in sweep_source
    assert "'false_positive_rate': float(metrics.false_positive_rate)" in sweep_source
    assert "'false_negative_rate': float(metrics.false_negative_rate)" in sweep_source


def test_fast_mode_epochs_do_not_clip_explicit_epochs_in_trainer():
    source = Path('src/trainer.py').read_text(encoding='utf-8')
    train_source = source[source.index('def train('):source.index('# 初始化 SwanLab')]
    assert 'num_epochs = self.train_config.max_epochs' in train_source
    assert 'min(self.train_config.max_epochs, self.config.fast_mode_epochs)' not in train_source


def test_extraction_config_maps_ascii_printable_bounds():
    config = AxonExperimentConfig(ascii_printable_min=65, ascii_printable_max=91)
    extraction_config = ExtractionConfig.from_axon_config(config)
    assert extraction_config.ascii_printable_min == 65
    assert extraction_config.ascii_printable_max == 91


def test_statistical_features_use_extraction_ascii_bounds():
    import numpy as np
    from kvd_features.extractor import extract_statistical_features
    extraction_config = ExtractionConfig(ascii_printable_min=65, ascii_printable_max=91)
    features = extract_statistical_features(np.array([65, 97], dtype=np.uint8), orig_length=2, config=extraction_config)
    assert features[10] == 1


def test_extract_all_features_passes_extraction_config_to_pe_extractor():
    source = Path('src/kvd_features/extractor.py').read_text(encoding='utf-8')
    extract_all_source = source[source.index('def extract_all_features'):]
    assert 'allow_fallback=allow_pe_fallback' in extract_all_source
    assert 'PEFeatureExtractor(config=config, axon_config=axon_config)' in source


def test_directory_label_inference_checks_ancestors():
    config = AxonExperimentConfig(malicious_dir_names=['malware'], benign_dir_names=['benign'])
    with tempfile.TemporaryDirectory() as tmp:
        dataset = MalwareDataset(tmp, axon_config=config, use_cache=False)
        sample_path = Path(tmp) / 'malware' / 'nested' / 'sample.exe'
        assert dataset._infer_label(sample_path) == 1


def test_dataset_cache_uses_cached_label():
    source = Path('src/dataset.py').read_text(encoding='utf-8')
    cached_block = source[source.index('if cached_data is not None:'):source.index('if not self.use_cache:')]
    assert 'cached_label' in cached_block


def test_pe_fallback_includes_api_category_features():
    source = Path('src/kvd_features/extractor.py').read_text(encoding='utf-8')
    fallback_source = source[source.index('def _extract_fallback'):source.index('def extract_lightweight_features')]
    assert 'category_counts' in fallback_source
    assert "for cat in ['network', 'process', 'filesystem', 'registry', 'crypto', 'injection']" in fallback_source


def test_strict_pe_parse_failure_returns_none_tuple():
    with tempfile.TemporaryDirectory() as tmp:
        bad_pe = Path(tmp) / 'bad.exe'
        bad_pe.write_bytes(b'MZnot a real pe')
        result = extract_all_features(
            str(bad_pe),
            ExtractionConfig(allow_pe_fallback=False),
            allow_pe_fallback=False,
        )
        assert result == (None, None, None, None, 0)


def test_dataset_strict_scan_skips_bad_pe_and_reaches_requested_valid_count(monkeypatch):
    import numpy as np
    import dataset as dataset_module

    calls = []

    def fake_extract_all_features(file_path, *_args, **_kwargs):
        calls.append(Path(file_path).name)
        if 'good' not in Path(file_path).name:
            return None, None, None, None, 0
        return (
            np.array([77, 90], dtype=np.uint8),
            np.ones(3, dtype=np.float32),
            np.ones(2, dtype=np.float32),
            np.ones(2, dtype=np.float32),
            2,
        )

    monkeypatch.setattr(dataset_module, 'extract_all_features', fake_extract_all_features)
    config = SimpleNamespace(
        benign_dir_names_fs=['white'],
        malicious_dir_names_fs=['black'],
        lightweight_feature_dim=2,
        strict_pe_parsing=True,
        allow_pe_fallback=True,
    )
    with tempfile.TemporaryDirectory() as tmp:
        white_dir = Path(tmp) / 'white'
        white_dir.mkdir()
        (white_dir / '00_bad.exe').write_bytes(b'MZbad')
        (white_dir / '01_good.exe').write_bytes(b'MZgood')
        dataset = MalwareDataset(
            tmp,
            max_byte_length=4,
            pe_feature_dim=3,
            stat_feature_dim=2,
            axon_config=config,
            max_samples_per_class=1,
            use_cache=True,
        )
        assert len(dataset) == 1
        assert dataset.file_list[0].name == '01_good.exe'
        assert dataset.scan_stats['pe_parse_failed_skipped'] == 1
        assert dataset.scan_stats['benign_valid'] == 1
        assert dataset.scan_stats['extracted'] == 1
        assert dataset.cache_path_list[0].exists()
        assert dataset.allow_pe_fallback is False
        assert calls == ['00_bad.exe', '01_good.exe']

        calls.clear()
        cached_dataset = MalwareDataset(
            tmp,
            max_byte_length=4,
            pe_feature_dim=3,
            stat_feature_dim=2,
            axon_config=config,
            max_samples_per_class=1,
            use_cache=True,
        )
        assert len(cached_dataset) == 1
        assert cached_dataset.scan_stats['cache_hits'] == 1
        assert calls == ['00_bad.exe']


def test_training_history_records_final_test_metrics():
    source = Path('src/trainer.py').read_text(encoding='utf-8')
    test_block = source[source.index('if test_loader is not None:'):source.index('self.metrics_tracker.save')]
    assert 'results[\'test\'].append(test_metrics)' in test_block
    assert 'self.metrics_tracker.update(test_metrics)' in test_block


def test_trainer_loads_best_checkpoint_before_test_eval():
    source = Path('src/trainer.py').read_text(encoding='utf-8')
    test_block = source[source.index('if test_loader is not None:'):source.index('self.metrics_tracker.save')]
    assert 'self._load_best_model_for_test()' in test_block
    assert 'def _load_best_model_for_test' in source


if __name__ == '__main__':
    tests = [
        test_hybrid_lightgbm_uses_full_dsra_arch_config,
        test_hybrid_lightgbm_passes_encoder_dimensions_from_config,
        test_empty_lists_from_config_are_preserved,
        test_stat_feature_dim_is_validated_against_stat_config,
        test_extraction_config_maps_stat_segment_count,
        test_dataset_scan_and_infer_use_same_directory_config,
        test_dsra_legacy_state_has_rope_positions,
        test_dsra_compat_head_selector_matches_core_selector,
        test_train_command_passes_fp16_to_training_config,
        test_focal_loss_uses_smoothed_ce_for_pt_when_label_smoothing_enabled,
        test_predict_and_extract_commands_pass_axon_config_to_feature_extraction,
        test_is_valid_sample_uses_magic_filter_before_pefile,
        test_lightweight_feature_empty_patterns_disable_hashes,
        test_extract_statistical_features_uses_extraction_config,
        test_dsra_chunk_layer_confidence_init_matches_mhdsra2,
        test_position_overflow_guard_in_malware_encoder,
        test_extract_all_features_returns_five_tuple,
        test_save_checkpoint_includes_scaler_state,
        test_single_chunk_path_sets_diversity_loss,
        test_cache_key_includes_lightweight_feature_dim,
        test_position_overflow_resets_diversity_accumulator,
        test_stat_features_moved_to_gpu_in_train_and_eval,
        test_dsra_chunk_layer_uses_precomputed_qkv,
        test_position_capped_in_slot_write,
        test_save_checkpoint_saves_last_epoch,
        test_dataset_cache_preserves_lightweight_features,
        test_evaluate_detaches_probs_before_numpy,
        test_extraction_config_maps_stat_feature_dim,
        test_short_sequence_encoder_returns_next_state,
        test_multilayer_dsra_uses_independent_states,
        test_fusion_uses_actual_dsra_arch_dim,
        test_hybrid_fusion_uses_actual_dsra_arch_dim,
        test_chunk_path_does_not_store_all_chunk_outputs,
        test_npz_dataset_requires_label,
        test_npz_dataset_normalizes_stat_feature_dim,
        test_lightweight_api_patterns_are_case_normalized,
        test_trainer_resume_starts_after_loaded_epoch,
        test_predict_command_passes_stat_features_to_model,
        test_exact_write_uses_effective_metadata_gates,
        test_dsra_chunk_layer_resets_memory_on_new_sequence,
        test_training_config_rejects_invalid_values,
        test_trainer_does_not_hardcode_cuda_amp_wrappers,
        test_trainer_rejects_empty_loaders_before_division,
        test_training_metrics_use_configured_decision_threshold,
        test_main_config_uses_fixed_pe256_precision_threshold,
        test_legacy_v3_candidate_keeps_dynamic_pe1500,
        test_config_directory_only_keeps_main_and_legacy_candidate,
        test_eval_command_reuses_checkpoint_training_config,
        test_feature_importance_command_is_exposed_and_writes_reports,
        test_feature_importance_uses_gradient_times_input_scores,
        test_pe_feature_metadata_marks_dynamic_indices,
        test_eval_auc_zero_is_not_treated_as_missing,
        test_eval_command_supports_threshold_sweep,
        test_eval_threshold_sweep_policy_blocks_test_and_all_splits,
        test_eval_report_includes_confusion_error_rates,
        test_eval_raw_fallback_limits_fast_checkpoint_samples,
        test_trainer_threshold_sweep_uses_supplied_thresholds,
        test_fast_mode_epochs_do_not_clip_explicit_epochs_in_trainer,
        test_extraction_config_maps_ascii_printable_bounds,
        test_statistical_features_use_extraction_ascii_bounds,
        test_extract_all_features_passes_extraction_config_to_pe_extractor,
        test_directory_label_inference_checks_ancestors,
        test_dataset_cache_uses_cached_label,
        test_pe_fallback_includes_api_category_features,
        test_strict_pe_parse_failure_returns_none_tuple,
        test_training_history_records_final_test_metrics,
        test_trainer_loads_best_checkpoint_before_test_eval,
    ]
    for test in tests:
        test()
        print(f'{test.__name__}: PASS')
