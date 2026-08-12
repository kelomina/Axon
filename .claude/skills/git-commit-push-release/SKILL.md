---
name: git-commit-push-release
description: 提交工作区修改并推送到 GitHub 仓库 + 把模型等大文件发布到 GitHub Releases。覆盖：提交范围治理（排除构建产物/临时文件/密钥）、推送到 origin、用 GitHub API（gh CLI 缺失时）创建 Release 并上传多个大文件资产、校验资产上传状态、可选补充 SHA-256 校验值。提炼自 2026-08-12 一次真实提交+发布（647 文件提交、4×41MB 模型发布）。
---

# Git 提交 + 推送 + GitHub Releases 发布

本项目 **模型/数据文件走 GitHub Releases，不走 git**（`.gitignore` 排除 `models/`、`*.pt`、`*.pth`、`reports/`、`*.npz`）。发布流程 = **git 提交推送**（源码）+ **GitHub API 上传资产**（大文件）。

**⚠️ 环境要点**：本机 **未安装 `gh` CLI**，GitHub API 凭据从 **Windows Credential Manager** 提取（git 配置了 `credential.helper=manager`）。

## 何时使用

- 用户指令含「提交修改」「推送到仓库」「发布/推送到 releases」等。
- 需要把 gitignored 的大文件（模型 .pt、onnx 等）交付给外部。

## 安全红线（必须遵守）

1. **绝不打印凭据/token**。检查 token 用存在性断言（`[ -n "$GH_TOKEN" ] && echo set`），`env | grep -i token` 会被权限分类器拒绝（会把 token 值打进出差记录）。
2. 提取 token 后**立即 `unset`**（同一条命令内用完即清）。
3. **发布是不可逆的对外操作**：多个候选文件时先 `AskUserQuestion` 确认要发布哪些，再执行。
4. 提交前**扫描暂存区**：排除构建产物（Rust `target/`、`build-bench/`）、FFI 探测残留（`.koffi-*/`）、临时目录（`_scratch_attr/`）、密钥文件。

## 完整流程

### Phase A 提交范围治理（`git add` 之前）

1. **看全貌**：`git status --porcelain`，按顶层目录归组（`awk -F/ '{print $1}' | sort | uniq -c`）。
2. **审计 .gitignore**：坏行（如 `*.json!manifests/**/*.json` 两行误并一行 → json 匹配语义全丢）必须恢复；补充忽略杂项：
   ```
   # Rust build artifacts (any target/ dir)
   target/
   **/target/

   # Session scratch / FFI probe leftovers
   .koffi-*/
   _scratch_attr/
   build-bench/
   resources/
   ```
3. **扫描暂存区**（`git add -A` 后）：
   - 密钥/敏感：`git diff --cached --name-only | grep -iE 'key|token|secret|\.env|credential|password|api_key'`
   - 二进制/构建产物：`grep -iE '\.(exe|dll|obj|pdb|so|rlib|o|pyc|npy|pt|bin)$'`
   - 大文件异常：`git diff --cached --numstat | awk '$1+$2 > 4000'`
4. **注意**：历史上已入库的 `target/` 文件（如 `git ls-files | grep -c target/` = 180）不会因新 ignore 规则自动移除——只要本轮未改动即可接受；若要清理需单独 `git rm --cached`（会改变历史语义，需用户确认）。

### Phase B 提交 + 推送

1. 提交前跑相关冒烟测试：`PYTHONPATH=src python -m pytest <改动相关测试> -q`（本项目 src 在 `PYTHONPATH` 上，直接 `pytest` 会 `ModuleNotFoundError`）。
2. 提交信息模板（中文，涵盖本轮要点 + 累积工作）：
   ```
   perf: 训练/推理提速（torch.compile reduce-overhead）+ 累积 loop 工作
   <空行>
   训练零风险杠杆（fp32 地板，精度不变）：...
   （改动清单、关键数字）
   <空行>
   同时纳入此前累积的 ... 未提交工作、... 示例、... manifests 与脚本。
   ```
3. 推送前看分歧：`git rev-list --left-right --count origin/master...HEAD`（0 后 = 领先数；`0  5` = 落后 0 领先 5）。
4. 推送：`git push origin master`。凭据走 Credential Manager，无需显式 token。

### Phase C 检查凭据（gh 缺失时）

```
# 提取 token（值只进变量，不落输出）：
TOKEN=$(printf 'protocol=https\nhost=github.com\n\n' | git credential fill 2>/dev/null | sed -n 's/^password=//p')
# 验证 token + 查既有 releases（只读）：
curl -s -H "Authorization: Bearer $TOKEN" "https://api.github.com/repos/<owner>/<repo>/releases"
# 用完：
unset TOKEN
```

### Phase D 创建 Release

1. **JSON body 用文件承载**（`--data-binary @file`），避免 heredoc 内联在 PowerShell/Git Bash 下中文/引号转义出 `Problems parsing JSON`。
2. 创建（返回 `id`，后续上传用）：
   ```
   curl -s -X POST -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
     --data-binary @release_body.json \
     "https://api.github.com/repos/<owner>/<repo>/releases"
   ```
3. body 里放 `tag_name` / `name` / `body`（Markdown 表格列文件与说明）/ `draft:false` / `prerelease:false`。
4. **先 AskUserQuestion 确认发布文件**——不可逆操作 + 候选多文件时用户决定权优先。

### Phase E 上传资产（大文件）

```
for f in "path/to/model1.pt" "path/to/model2.pt"; do
  name=$(basename "$f")
  curl -s -X POST -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/octet-stream" \
    --data-binary "@$f" \
    "https://uploads.github.com/repos/<owner>/<repo>/releases/<RELEASE_ID>/assets?name=$name"
done
```
- 响应含 `state: uploaded` 即成功；**每次循环独立 curl**（一个命令传多个文件会只传第一个或失败）。
- 大文件（41MB×4）需给足超时（`timeout: 300000`）。

### Phase F 校验 + 补充 SHA-256

1. **校验资产状态**：`GET /releases/<id>`，确认 `assets` 数量与每个 `state: uploaded`。API 偶发限流返回空 → `sleep 2` 重试。
2. **SHA-256 校验值**：本地 `sha256sum <file>` 逐个算，然后 **PATCH release body** 把校验值追加进去（正文末尾加 `### SHA-256` 表）。
3. 用户常要「给 Release 补充 SHA-256」——直接把校验值写进 release body 比单独传 checksum 文件更直观。

### Phase G 收尾

- 清理临时文件（`release_body.json` 等，注意别删被 gitignore 的目录本身）。
- `git status --porcelain` 确认干净。
- 汇报：commit hash、推送范围、release URL、资产清单 + SHA-256 表。

## 通用教训

1. **坏行 .gitignore 是隐藏地雷**：两行误并一行会让忽略规则静默失效，先 `git diff .gitignore` 审计。
2. **Rust `target/`、`build-bench/` 等构建产物**会随 `git add -A` 进暂存——必须显式忽略或选择性 add。
3. **PowerShell/Git Bash 里内联 JSON**（尤其含中文/引号）易 `Problems parsing JSON`；一律 `--data-binary @file`。
4. **token 生命周期**：单命令内提取→使用→`unset`，绝不打印；跨命令持久化 token 是隐患。
5. **验证要有 HTTP 状态码**：`curl -w "HTTP %{http_code}\n"` 让空响应/限流可见，别只解析 json。

## 产物清单

- 提交 `commit` + 推送后的远程 HEAD
- Release（tag / name / body / assets）
- 每资产 SHA-256 校验值（写在 release body）

## 相关命令

`git status` · `git add -A` · `git diff --cached` · `git commit` · `git push` · `git credential fill`（token 提取）· `curl` GitHub REST + uploads API · `sha256sum`
