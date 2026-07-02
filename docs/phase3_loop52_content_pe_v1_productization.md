# Phase 3 Loop52: Content PE v1 Productization

日期：2026-07-02

## 目标

Loop52 不是新的模型冲榜候选，而是把 Loop28 已验证有效的 100 维 content PE metadata 从 Stage-2 临时脚本中抽出来，固化为稳定、可复用、可测试的 `content_pe_v1` schema。

这轮继续执行硬规则：文件名、路径、扩展名、目录名、`source_sha256`、`cache_path`、`sample_index`、`split` 和行顺序只能用于打开文件、缓存对齐、覆盖审计、去重、人工复核或生成一次性的人工标签清单，不能作为模型特征、二阶段融合特征、阈值捷径、自动改标证据或上线推理依据。

训练集目录或文件名最多说明“样本当时被放进哪个人工标签桶”，不是文件内容证据。实战部署中的文件名和训练集命名分布通常完全不同，攻击者也可以低成本改名，因此模型必须学习字节、PE 结构、统计量、证书/资源/导入表等内容侧信号。

## 改动

- 新增 `src/kvd_features/content_pe_v1.py`，提供稳定的 `CONTENT_PE_V1_FEATURE_NAMES` 和 `extract_content_pe_v1_features()`。
- 保留向后兼容别名 `CONTENT_PE_FEATURE_NAMES` 和 `_content_pe_features_from_path`，让现有 Loop28/Stage-2 脚本不需要改变实验语义。
- `scripts/train_stage2_cache_matrix.py` 改为引用稳定 schema，移除脚本内重复的 v1 特征定义与提取函数。
- `scripts/build_content_pe_feature_cache.py` 改为从稳定 schema 导入 v1 特征，并新增 `--smoke --limit`，只用于 smoke-test 时防止误跑全量；单独使用 `--limit` 会被程序拒绝，避免正式缓存被误截断。
- cache builder 对已有 `.npz` 不再只看文件存在，而是打开验证 `features` shape、dtype 和 finite 数值；坏缓存会被重新提取并计入 `refreshed_invalid`。
- 测试补充：
  - 同一内容在不同文件名下提取结果必须一致。
  - productized schema 必须等于 Stage-2 alias。
  - `CONTENT_PE_FEATURE_NAMES` 必须通过 identity feature guard，不能出现命名/路径/hash/split/行号等身份字段。

## Smoke 结果

命令：

```powershell
.\vnev\Scripts\python.exe scripts\build_content_pe_feature_cache.py `
  --predictions reports\random_20w_split\loop27_val_predictions.csv `
  --cache-dir reports\random_20w_split\loop52_content_pe_v1_smoke_cache_limit32_guarded `
  --workers 1 `
  --smoke `
  --limit 32 `
  --output-json reports\random_20w_split\loop52_content_pe_v1_smoke_cache_limit32_guarded.json
```

结果：

| 字段 | 值 |
| --- | ---: |
| input rows | 20000 |
| deduplicated rows before limit | 20000 |
| smoke | true |
| smoke limit | 32 |
| processed unique rows | 32 |
| feature dim | 100 |
| created | 32 |
| refreshed invalid | 0 |
| zero features | 0 |

这个 smoke 只验证 extractor/cache writer 链路，不构成 Val 结果，也不允许进入 Test-10k。

## 验证

```powershell
.\vnev\Scripts\python.exe -m py_compile src\kvd_features\content_pe_v1.py scripts\build_content_pe_feature_cache.py scripts\train_stage2_cache_matrix.py
.\vnev\Scripts\python.exe -m pytest tests\test_stage2_content_pe_features.py tests\test_content_pe_v1_productized.py tests\test_build_content_pe_feature_cache.py tests\test_audit_content_pe_productization.py tests\test_identity_feature_guard.py -q
```

结果：`11 passed`。

## 决策

Loop52 合格进入代码基线：它把 Loop28 的高价值内容侧 PE 信号从实验脚本提升为稳定 schema，且继续阻断命名/路径/扩展名/hash/split/行号这类身份线索。

下一步若要把它变成新候选，必须重新构建完整 Train/Val sidecar cache，并在 `20000 train / 20000 val` 上跑完整 Val 漏斗。正式 sidecar cache 报告必须满足 `"limit": null` 且 `"unique_rows" == "deduplicated_rows_before_limit"`。只有 Val 明显超过 Loop28 参考门槛，才允许进入 Test-10k。
