"""清理本地运行日志"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"


def main() -> int:
    log_dir = LOG_DIR.resolve()
    project_root = PROJECT_ROOT.resolve()
    if project_root not in log_dir.parents:
        raise RuntimeError(f"日志目录越界: {log_dir}")

    if not log_dir.exists():
        print(f"日志目录不存在: {log_dir}")
        return 0

    removed = 0
    for path in sorted(log_dir.glob("*.log*")):
        resolved = path.resolve()
        if log_dir not in resolved.parents:
            raise RuntimeError(f"日志文件越界: {resolved}")
        if resolved.is_file():
            resolved.unlink()
            removed += 1
            print(f"已删除: {resolved}")

    print(f"清理完成，共删除 {removed} 个日志文件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
