@echo off
rem 桌面会话运行 Avast 逐文件扫描（72 个未判定冲突样本）
rem 双击本文件，或在桌面终端运行：run_avast_scan.bat
cd /d %~dp0..
set PY=python
%PY% scripts\avast_scan.py %*
echo.
echo 若上表出现大量 crash，说明 ashCmd 在非桌面会话不可用；
echo 请改为在桌面 PowerShell 直接运行上面命令行。
pause
