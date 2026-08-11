---
name: corpus-conflict-cleanup
description: 语料跨树冲突清理与缓存重建。同一 sha256 同时存在于良性树与恶意树时，先复核判定归属，再物理归位（移动/删除错误树副本）、重建 dir_index/name_index/meta 缓存并验证冲突清零。避免标签污染与训练数据交叉污染。
---

# 语料跨树冲突清理 + 缓存重建

同一 sha256 在良性树与恶意树**同时存在**时，训练会把同一样本既当良又当恶（标签污染）。本 skill 是从 2026-08-10 一次真实清理提炼的完整闭环：**复核判定 → 物理归位（零数据丢失）→ 缓存重建 → 验证**。

## 何时使用

- `clean_corpus_labels.py` 产出 `corpus_conflicts.csv`（跨树冲突 sha 全量清单）——冲突检测的**权威入口**。
- `build_label_governance_queue.py` 在 `review_queue.csv` 中把冲突样本标为 `candidate=cross_tree_conflict`（连同 FN/FP/unlocatable 候选一起出复核队列）。
- 用户指令如"直接把文件移动进入各自的目录并重建缓存"（跳过进一步验证，直接物理清理）。
- 任何"删一侧、留一侧"且带缓存的语料归位操作。

## 语料与缓存前置知识

**权威路径（禁止目录枚举）**：`F:\私人\良性文件\待加入白名单`（+`_upx`）、`F:\私人\恶意\MB\unziped\<日期>\`、`E:\Project\python\KoloVirusDetector_ML_V2-main\benign_samples\待加入白名单` 与 `malicious_samples\待拉黑\<日期>\`。

- **文件名即 sha256**（有的带 `.exe/.dll` 扩展）。同 sha 必同内容 → 良/恶两树副本内容相同，归位可放心覆盖/删冗余。
- **盘符映射**：G:/H: 已改为 F:，纯前缀替换（`resolve()`）。旧缓存/meta 里的 G:/H: 路径是失效指针。
- **F: 恶意盘按访问触发 AV 扫描**：个别文件 open 可能无限阻塞。纯 copy/rename 不受影响，但**别为"确认同内容"去读文件哈希**。

**缓存三件套**：
| 缓存 | 路径 | 用途 | 重建方式 |
|---|---|---|---|
| `dir_index.pkl` | `reports/full_739k_benign/content_versionstr/` | 冲突检测（目录→文件名 set） | **增量**：只重 listdir 受影响目录 |
| `meta.csv` | `reports/full_739k_benign/content_pe_v1/` | Stage-2 训练消费 label/raw_path | 更新冲突 sha 行的 label+raw_path |
| `name_index.pkl` | **`reports/full_739k/`**（不在 benign 子目录） | cache 提取定位（文件名→路径） | **全量** os.walk（须加 `--name-index` 开关） |

命令统一用 `vnev\Scripts\python.exe`（虚拟环境目录名 `vnev` 无点）；脚本用绝对 `PROJECT_ROOT`，无 cwd 依赖。

## 完整链路

### Phase A 复核判定 → `review_verdicts.csv`（**前置，必须先跑**）
`review_verdicts.csv` 是 `clean_corpus_labels.py` 的**硬依赖**（脚本无条件 open，缺失即 `FileNotFoundError`）。生成链路：

1. `build_label_governance_queue.py` → `review_queue.csv` + `summary.json`（需已生成 base_prob、Stage-2 预测、versionstr 特征）。
2. 从 `review_queue.csv` 挑高价值样本存 `priority_review.csv`（本次 28 个），`copy_priority_samples.py` 拷到 `D:\待复核\samples\`。
3. （覆盖全部未判定冲突）`copy_undecided_conflict_samples.py`：从 `corpus_conflicts.csv` 的 `conflict_undecided` 行拷 72×2 样本到 `D:\待复核\samples_batch2\`，产 `batch2_manifest.csv` + VT 链接。
4. 辅助判据（可组合）：
   - **人工**：`D:\待复核\复核完成的列表.txt`，每行 `<sha256>:<Begin|Mal>`（Begin=良性 label0，Mal=恶意 label1）。
   - **VirusTotal**：`vt_recheck.py`。key 用环境变量 `VT_API_KEY` 或 `config/vt_api_key.txt`（免费 key 限速 4/min → `--delay 15`）。verdict 语义：`malicious≥3→Mal(label1)`、`==0→Begin(label0)`、其余 `sus`（人工看引擎明细）。
   - **本地 Avast**：先跑 `copy_undecided_conflict_samples.py` 得 `batch2_manifest.csv`，再 `avast_scan.py`。⚠️ ashCmd 需**桌面会话**，非交互会崩 exit=-529697949；`--ashcmd` 可指定路径；exit 0→clean、1/3/4→infected。
5. `parse_review_results.py` 读复核 txt + `priority_review.csv` → `review_verdicts.csv` + `label_corrections.csv` + `review_results.json`。

**判定策略**：未判定的冲突 sha **保守维持原 label** 作 final（只消除跨树并存，不猜未判定的可能错标）；如需彻底防污染，`clean_corpus_labels.py --drop-undecided` 把未判定行剔除出重训。

### Phase B 冲突检测与决策（`clean_corpus_labels.py`）
读 dir_index.pkl + meta.csv + review_verdicts.csv → 产 `corpus_conflicts.csv`（每 sha 一个 final `new_label`）、`label_override.csv`（改标契约）、`cleaned_meta.csv`、`corpus_clean_report.json`。action 三态：`conflict_corrected`（有判定且改标）/ `conflict_kept`（有判定保持）/ `conflict_undecided`（无判定）。

### Phase C move plan（只读探测，`probe_conflict_move_plan.py`）
从 dir_index + meta + corpus_conflicts 生成 `move_plan_preview.csv`：
- 每 sha：`benign_locs` / `malware_locs`（两侧完整路径，可能多份）、`move_srcs`（drop 侧待归位文件）、`action_move`。
- `action_move`：`KEEP_BENIGN_DROP_MALWARE`（final=0）或 `KEEP_MALWARE_DROP_BENIGN`（final=1）。
- 输出含 `[rows missing one side]` 统计。

### Phase D dry-run 前置校验（破坏性操作前必须）
移动前逐 sha 验证（只读）：keep 侧存在 `any(os.path.exists(p) for p in keep_locs)`、drop 侧每个 move_src 存在、目标目录 `dirname(keep_locs[0])` 存在。**任一侧缺失 = 该 sha 变孤儿，先查明再动。**
注意：这是**人工只读校验**，脚本不 gate（src 缺失只 `[SKIP]` 跳过；keep 侧在移动后才输出验证）。

### Phase E 物理归位（`apply_conflict_moves.py`）
对每个 move_src：
```
if tgt exists:            # keep 树已有同 sha（必同内容）
    os.remove(src)        # 删冗余即可，等效"移动合并"
else:
    shutil.move(src, tgt) # 跨盘自动 copy+unlink；同盘即 rename
```
**关键坑**：
- ⚠️ `os.replace`/`os.rename` **不支持跨盘**（F:→E: 会 `WinError 17`）。必须 `shutil.move`。实测首轮 70/119 因此失败。
- **扩展名不一致边界**：tgt 按 `basename(src)` 拼接；两侧扩展名不同（sha 无扩展 vs `.exe`）时是"新增同内容副本"而非合并——不破坏零丢失与 BOTH-side 检查，但会留冗余（dir_index/name_index 记两份），可接受。
- 审计日志 `move_executed.csv` **每轮 'w' 截断写，只含最后一轮**（实测仅 70 行）；**权威全集 = `move_plan_preview.csv` 的 `move_srcs` 列**（dir_index 刷新以 plan 为主源、move_log 仅补充）。
- 脚本**幂等**：src 已不存在 → 记 `already-moved` 跳过。
- 移动后脚本打印验证（**以脚本实际打印为准**）：`drop-side cleared` = 无残留 ops/总 ops（本次 70/70）、`keep-side intact` = len(rows)（100/100）。

### Phase F 重建缓存（`rebuild_corpus_cache.py`）
⚠️ 默认只重建 dir_index + meta；**必须加 `--name-index`** 才重建 name_index.pkl。
1. **dir_index.pkl 增量**：受影响目录集合 = 所有 `move_srcs` 父目录 + keep 目录。⚠️ **不要只从 move_log 收集**（多轮执行 log 只含最后一轮）；从 move_plan 的 `move_srcs` 列直接推导覆盖全部。对每个受影响目录 `os.listdir` 更新。边界：`if d in index` 只更新旧索引已有的目录；**keep 目标目录若不在旧 index 会被跳过，应新增键** `set(os.listdir(d))`。
2. **meta.csv**：100 冲突 sha 行 `label→new_label`、`raw_path→keep 侧路径（resolve F:）`、`located=1`（keep_path 可得时）。其余约 81 万行（813,098）不动。
3. **name_index.pkl 全量**：os.walk 全部现存 roots（**必须含 `F:\私人\良性文件` 根目录以覆盖 UPX 白名单子目录**，不能只列 `待加入白名单`），清旧 G:/H: 失效指针。全树 ~1.37M 文件约 2-3 分钟。建议：先 `copy2` 备份旧 pkl，再 `.tmp`+`os.replace` 原子写；长任务放 detached 进程 + 日志监视，避免被杀留下截断缓存。

### Phase G 验证冲突清零（只读）
1. 重跑 `probe_conflict_move_plan.py`：`[rows missing one side]=100` 恰是"双侧残留=0、清理成功"的正确解读（**不要误读为全是问题**）。
2. 只读统计：加载重建后 dir_index.pkl，对 100 冲突 sha：`still BOTH-side`（良+恶双侧残留）应为 **0**、`side-vs-final mismatches`（所在树 ⊄ 最终判定树）应为 **0**。
3. `meta.csv` 抽查：Grep 关键 sha（如 avast 报毒 → 恶意树+label 1；人工确认白文件 → 良性树+label 0）。
4. `name_index` 抽查：冲突 sha 路径指向现存文件、不再含 `G:`/`H:` 前缀。

### Phase H 回滚/恢复（需要时）
- **covered 分支**：无需回滚——keep 树同内容仍在，删冗余安全。
- **moved 分支**：按 `move_executed.csv` 审计行 `tgt→src` 反向 `shutil.move`。
- 回滚后重跑 Phase F 缓存重建 + Phase G 验证。

## 清理后接入重训
- **Stage-2 重训**：读 `label_override.csv`（按 index 覆盖 meta label）。
- **基座重训**：须再同步 manifest/cache npz 的 label（clean_corpus_labels.py docstring 已明确；物理清理后 meta 已是新值，**npz 仍为旧值需同步**）。

## 通用教训（复用价值最高）

1. **破坏性批量操作三件套**：dry-run 预览 → 双侧存在校验 → 审计日志。缺一不可。
2. **跨盘移动**永远用 `shutil.move`，别用 `os.replace`。
3. **增量索引**只重建受影响键（dir_index listdir 单目录），别全量重算；但**指针型索引**（name_index 路径）必须全量重建以清失效盘符。
4. **长任务/间歇性环境故障**：`run_in_background` / detached 进程；验证用只读工具（Grep/Read）先行，不阻塞。
5. **审计日志写覆盖 = 只留最后一轮**：权威全集记在 plan 文件（move_srcs 列）；增量刷新以 plan 为主源。
6. **语料污染要物理根除**：label_override.csv 只改训练契约；真正清理必须移动文件 + 重建全部缓存，否则缓存与磁盘不一致会持续污染后续提取。

## 产物清单
- `review_queue.csv` / `summary.json`（build_label_governance_queue.py）
- `review_verdicts.csv` / `label_corrections.csv` / `review_results.json`（parse_review_results.py）
- `corpus_conflicts.csv` / `label_override.csv` / `cleaned_meta.csv` / `corpus_clean_report.json`（clean_corpus_labels.py）
- `move_plan_preview.csv`（probe，只读）
- `move_executed.csv`（归位审计，仅最后一轮；权威全集=move_srcs）
- 重建后的 `dir_index.pkl` / `name_index.pkl` / `meta.csv`

## 相关脚本
`probe_conflict_move_plan.py` · `apply_conflict_moves.py` · `rebuild_corpus_cache.py` · `clean_corpus_labels.py` · `build_label_governance_queue.py` · `parse_review_results.py` · `copy_priority_samples.py` · `copy_undecided_conflict_samples.py` · `vt_recheck.py` · `avast_scan.py` · `run_avast_scan.bat`
