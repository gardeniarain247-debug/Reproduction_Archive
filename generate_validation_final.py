from pathlib import Path
from datetime import datetime, timezone
import subprocess
import hashlib
import json
import sys
import re

ROOT = Path(r"D:\latex\IJFS")
OUT = ROOT / "validation_final"
OUT.mkdir(exist_ok=True)

# 检测测试文件夹
if (ROOT / "tests").is_dir():
    TEST_DIR = "tests"
else:
    TEST_DIR = "."

# 确认存在测试脚本
test_files = list((ROOT / TEST_DIR).glob("test_*.py"))
if not test_files:
    raise FileNotFoundError(
        "No test_*.py files found. Check TEST_DIR."
    )

# 使用 Spyder 当前环境中的 Python
command = [
    sys.executable,
    "-m", "unittest",
    "discover",
    "-s", TEST_DIR,
    "-p", "test_*.py",
    "-v"
]

# 在日志中保留命令和运行环境
header = (
    f"UTC: {datetime.now(timezone.utc).isoformat()}\n"
    f"Python: {sys.executable}\n"
    f"Working directory: {ROOT}\n"
    f"Command: {' '.join(command)}\n"
    f"Test files discovered: {len(test_files)}\n\n"
)

result = subprocess.run(
    command,
    cwd=str(ROOT),
    capture_output=True,
    text=True,
    encoding="utf-8",
    errors="replace"
)

log = header + result.stdout + "\n" + result.stderr

log_file = OUT / "TEST_RESULTS_final.txt"
log_file.write_text(log, encoding="utf-8")

# 解析真实测试结果
match = re.search(r"Ran (\d+) tests? in", log)
tests = int(match.group(1)) if match else None

fail_match = re.search(
    r"FAILED\s*\(([^)]*)\)", log
)

passed = (
    result.returncode == 0
    and tests is not None
    and re.search(r"(?m)^OK\s*$", log) is not None
)

# 计算本次测试代码的整体指纹
source_files = sorted(
    p for p in ROOT.rglob("*.py")
    if "validation_final" not in p.parts
    and "__pycache__" not in p.parts
)

source_hashes = {
    str(p.relative_to(ROOT)).replace("\\", "/"):
    hashlib.sha256(p.read_bytes()).hexdigest()
    for p in source_files
}

hash_file = OUT / "TEST_SOURCE_SHA256_final.json"
hash_file.write_text(
    json.dumps(source_hashes, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

summary = {
    "validation_type": "automated_tests",
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "python_executable": sys.executable,
    "tests_run": tests,
    "tests_passed": tests if passed else None,
    "all_passed": passed,
    "returncode": result.returncode,
    "failure_summary": (
        fail_match.group(1) if fail_match else None
    ),
    "real_full_data_rerun": False,
    "test_log": log_file.name,
    "test_source_hash_manifest": hash_file.name
}

summary_file = OUT / "VALIDATION_final.json"
summary_file.write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("=" * 55)
print("Tests run:", tests)
print("All passed:", passed)
print("Exit code:", result.returncode)
print("Log:", log_file)
print("Summary:", summary_file)
print("Source hashes:", hash_file)
print("=" * 55)

if not passed:
    print("Tests did not all pass. Review the log.")
