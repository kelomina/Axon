#!/usr/bin/env python3
"""批量给指定目录中的低熵 PE 文件施加 UPX 壳的自动化脚本 (高性能稳定多进程版)。

用于针对二分类恶意软件检测模型生成加壳良性数据增强样本，提升模型的泛化能力。

主要特性：
1. 自动计算文件字节熵 (Shannon Entropy, 0~8 bits/byte)。
2. 过滤符合条件的低熵文件（默认熵 <= 6.0）。
3. 使用 UPX 压缩文件，严格避免覆盖原始文件。
4. 加壳成功后计算加壳后文件的 SHA256，重命名为 `<SHA256>.<ext>`。
5. 采用滑动窗口受控多进程池 (ProcessPoolExecutor)，避免海量任务一次性入列导致 OS 句柄溢出。
6. 维护 `upx_manifest.json` 索引记录，实现秒级断点续传（重启后直接跳过已完成的文件）。
"""

import os
import sys
import stat
import math
import json
import shutil
import argparse
import hashlib
import subprocess
import collections
import concurrent.futures
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor


def _safe_unlink(file_path: Path):
    """强制删除文件，避免因只读属性引发权限拒绝异常。"""
    try:
        if file_path.exists():
            os.chmod(file_path, stat.S_IWRITE)
            file_path.unlink()
    except Exception:
        pass


def calculate_entropy(file_path: Path) -> float:
    """计算文件的 Shannon 字节熵 (0.0 ~ 8.0 bits/byte)。"""
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        if not data:
            return 0.0
        counts = collections.Counter(data)
        total = len(data)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            entropy -= p * math.log2(p)
        return entropy
    except Exception as e:
        print(f"[!] 读取文件计算熵失败 {file_path}: {e}", file=sys.stderr, flush=True)
        return 9.0  # 失败时返回超高熵以跳过


def calculate_sha256(file_path: Path) -> str:
    """计算文件的 SHA256 哈希值。"""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def is_pe_file(file_path: Path) -> bool:
    """检查文件是否为 Windows PE 可执行文件 (MZ 头校验)。"""
    try:
        with open(file_path, "rb") as f:
            header = f.read(2)
            return header == b"MZ"
    except Exception:
        return False


def find_upx_executable(custom_path: str = None) -> str:
    """查找系统中 UPX 可执行文件的路径。"""
    if custom_path:
        custom_exe = Path(custom_path)
        if custom_exe.exists() and custom_exe.is_file():
            return str(custom_exe)
        found = shutil.which(custom_path)
        if found:
            return found
        raise FileNotFoundError(f"未找到指定的 UPX 可执行文件: {custom_path}")
    
    found = shutil.which("upx")
    if found:
        return found
    
    # 尝试常见的默认路径
    possible_paths = [
        Path("C:/GreenPrograms/upx_pacther/upx.exe"),
        Path("C:/GreenPrograms/upx_pacther/upx.EXE"),
        Path("C:/Program Files/upx/upx.exe"),
        Path("C:/tools/upx/upx.exe"),
    ]
    for p in possible_paths:
        if p.exists():
            return str(p)
            
    raise FileNotFoundError("未在环境变量 PATH 中找到 `upx` 命令。请确保已安装 UPX，或通过 `--upx-path` 参数指定 UPX 的可执行文件路径。")


def process_file(
    file_path_str: str,
    output_dir_str: str,
    max_entropy: float,
    upx_bin: str,
    upx_compression_level: str,
    force_pack: bool
) -> dict:
    """进程池单 Task 执行函数：检查熵 -> 复制临时文件 -> UPX 加壳 -> 计算加壳后 SHA256 -> 重命名。"""
    file_path = Path(file_path_str)
    output_dir = Path(output_dir_str)

    res = {
        "file": str(file_path),
        "filename": file_path.name,
        "status": "skipped",
        "reason": "",
        "entropy": 0.0,
        "packed_filename": None,
        "packed_sha256": None,
    }

    if not is_pe_file(file_path):
        res["reason"] = "非 PE 文件 (无 MZ 头部)"
        return res

    entropy = calculate_entropy(file_path)
    res["entropy"] = entropy
    if entropy > max_entropy:
        res["reason"] = f"高熵文件 ({entropy:.2f} > {max_entropy:.2f})"
        return res

    orig_ext = file_path.suffix.lower()
    if not orig_ext:
        orig_ext = ".exe"

    # 临时文件在 output_dir 下隔离，文件名加上 pid 防碰撞
    temp_output_path = output_dir / f"_temp_packing_{os.getpid()}_{file_path.name}"
    
    try:
        shutil.copy2(file_path, temp_output_path)
    except Exception as e:
        res["status"] = "failed"
        res["reason"] = f"复制文件失败: {e}"
        return res

    try:
        # 构造 UPX 加壳命令
        cmd = [upx_bin, f"-{upx_compression_level}"]
        if force_pack:
            cmd.append("--force")
        cmd.append(str(temp_output_path))

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120
        )

        if proc.returncode != 0:
            res["status"] = "failed"
            res["reason"] = f"UPX 加壳失败 (Exit Code {proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
            _safe_unlink(temp_output_path)
            return res

        # 计算加壳后文件的 SHA256
        packed_sha256 = calculate_sha256(temp_output_path)
        packed_filename = f"{packed_sha256}{orig_ext}"
        final_target_path = output_dir / packed_filename

        # 若加壳后的文件已在输出目录中存在，删除临时文件并记录跳过
        if final_target_path.exists():
            _safe_unlink(temp_output_path)
            res["status"] = "already_exists"
            res["reason"] = f"加壳后目标文件已存在: {packed_filename}"
            res["packed_filename"] = packed_filename
            res["packed_sha256"] = packed_sha256
            return res

        # 重命名为加壳后的 SHA256
        temp_output_path.rename(final_target_path)
        res["status"] = "success"
        res["packed_filename"] = packed_filename
        res["packed_sha256"] = packed_sha256
        return res

    except subprocess.TimeoutExpired:
        res["status"] = "failed"
        res["reason"] = "UPX 命令运行超时 (120s)"
        _safe_unlink(temp_output_path)
        return res
    except Exception as e:
        res["status"] = "failed"
        res["reason"] = f"处理异常: {e}"
        _safe_unlink(temp_output_path)
        return res


def main():
    cpu_cnt = os.cpu_count() or 8
    default_processes = max(8, cpu_cnt)

    parser = argparse.ArgumentParser(
        description="批量给指定目录中的低熵 PE 文件加 UPX 壳并重命名为加壳后的 SHA256 (多进程池版)。"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=r"H:\私人\良性文件\待加入白名单",
        help="输入的原始文件目录路径 (默认: H:\\私人\\良性文件\\待加入白名单)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="加壳后文件的保存输出目录 (若未指定，默认为输入目录同级的 `<input-dir>_upx`)"
    )
    parser.add_argument(
        "--max-entropy",
        type=float,
        default=6.0,
        help="判定为低熵文件的最高熵阈值 (0.0~8.0，默认: 6.0 bits/byte)"
    )
    parser.add_argument(
        "--upx-path",
        type=str,
        default=None,
        help="UPX 可执行文件的路径 (若不在环境变量中需指定)"
    )
    parser.add_argument(
        "--upx-level",
        type=str,
        default="9",
        choices=["1", "2", "3", "4", "5", "6", "7", "8", "9", "best"],
        help="UPX 压缩级别 (1-9 或 best，默认: 9)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="对 UPX 加壳启用 --force 参数"
    )
    parser.add_argument(
        "--processes", "--workers", "--threads",
        dest="processes",
        type=int,
        default=default_processes,
        help=f"进程池并发进程数 (默认最小 8 进程，当前系统推荐: {default_processes})"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="递归扫描子目录"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"[!] 错误: 输入目录不存在: {input_dir}", file=sys.stderr, flush=True)
        sys.exit(1)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_dir.parent / f"{input_dir.name}_upx"

    output_dir.mkdir(parents=True, exist_ok=True)

    # 自动清理上一次中断遗留的临时文件
    for leftover in output_dir.glob("_temp_packing_*"):
        _safe_unlink(leftover)

    # 检查/加载断点续传 Manifest 文件
    manifest_path = output_dir / "upx_manifest.json"
    manifest = {}
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            print(f"[*] 加载断点续传记录清单，已包含 {len(manifest)} 个处理记录", flush=True)
        except Exception:
            manifest = {}

    try:
        upx_bin = find_upx_executable(args.upx_path)
        print(f"[*] 使用 UPX 可执行文件: {upx_bin}", flush=True)
    except Exception as e:
        print(f"[!] 错误: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    print(f"[*] 输入目录: {input_dir}", flush=True)
    print(f"[*] 输出目录: {output_dir}", flush=True)
    print(f"[*] 低熵判定阈值: <= {args.max_entropy:.2f} bits/byte", flush=True)
    print(f"[*] 进程池大小: {args.processes} 个进程 (最小 8 进程配置)", flush=True)
    print(f"[*] 开始扫描输入目录文件...", flush=True)

    if args.recursive:
        all_files = [p for p in input_dir.rglob("*") if p.is_file()]
    else:
        all_files = [p for p in input_dir.iterdir() if p.is_file()]

    stats = {
        "total": len(all_files),
        "success": 0,
        "already_exists": 0,
        "skipped": 0,
        "failed": 0,
    }

    # 秒级过滤：如果 Manifest 中已有记录且目标文件存在，直接跳过，无需重复跑 UPX
    to_process = []
    for f in all_files:
        filename = f.name
        if filename in manifest:
            packed_name = manifest[filename]
            if (output_dir / packed_name).exists():
                stats["already_exists"] += 1
                continue
        to_process.append(f)

    print(f"[*] 样本扫描完毕: 总计 {len(all_files)} 个文件，已加速跳过 {stats['already_exists']} 个完成项，剩余 {len(to_process)} 个文件提交进程池...", flush=True)

    def save_manifest():
        try:
            temp_m = manifest_path.with_suffix(".tmp")
            with open(temp_m, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
            temp_m.replace(manifest_path)
        except Exception:
            pass

    if to_process:
        save_counter = 0
        max_pending = args.processes * 4
        file_iter = iter(to_process)
        pending_futures = {}

        with ProcessPoolExecutor(max_workers=args.processes) as executor:
            # 填满初始活动队列
            for f in file_iter:
                fut = executor.submit(
                    process_file,
                    str(f),
                    str(output_dir),
                    args.max_entropy,
                    upx_bin,
                    args.upx_level,
                    args.force
                )
                pending_futures[fut] = f
                if len(pending_futures) >= max_pending:
                    break

            processed_count = 0
            while pending_futures:
                done_set, _ = concurrent.futures.wait(
                    pending_futures.keys(),
                    return_when=concurrent.futures.FIRST_COMPLETED
                )

                for fut in done_set:
                    orig_file = pending_futures.pop(fut)
                    processed_count += 1
                    res = fut.result()

                    status = res["status"]
                    stats[status] = stats.get(status, 0) + 1

                    filename = res["filename"]
                    packed_filename = res.get("packed_filename")

                    if packed_filename and status in ("success", "already_exists"):
                        manifest[filename] = packed_filename
                        save_counter += 1

                    if status == "success":
                        print(f"[{processed_count}/{len(to_process)}] [成功] {filename} (熵:{res['entropy']:.2f}) -> {res['packed_filename']}", flush=True)
                    elif status == "already_exists":
                        print(f"[{processed_count}/{len(to_process)}] [跳过-已存在] {filename} -> {res['packed_filename']}", flush=True)
                    elif status == "skipped":
                        print(f"[{processed_count}/{len(to_process)}] [跳过] {filename}: {res['reason']}", flush=True)
                    else: # failed
                        print(f"[{processed_count}/{len(to_process)}] [失败] {filename}: {res['reason']}", file=sys.stderr, flush=True)

                    if save_counter >= 50:
                        save_manifest()
                        save_counter = 0

                    # 动态补满活动队列
                    try:
                        next_f = next(file_iter)
                        new_fut = executor.submit(
                            process_file,
                            str(next_f),
                            str(output_dir),
                            args.max_entropy,
                            upx_bin,
                            args.upx_level,
                            args.force
                        )
                        pending_futures[new_fut] = next_f
                    except StopIteration:
                        pass

        save_manifest()

    print("\n" + "=" * 50, flush=True)
    print("处理完成统计摘要:", flush=True)
    print(f"  总扫描文件数: {stats['total']}", flush=True)
    print(f"  加壳成功数  : {stats['success']}", flush=True)
    print(f"  重复/已存在 : {stats['already_exists']}", flush=True)
    print(f"  不符合条件  : {stats['skipped']}", flush=True)
    print(f"  失败文件数  : {stats['failed']}", flush=True)
    print(f"  输出结果目录: {output_dir}", flush=True)
    print("=" * 50, flush=True)


if __name__ == "__main__":
    main()
