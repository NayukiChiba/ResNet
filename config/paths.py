from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
# 数据集和输出目录
DATASET_DIR = ROOT_DIR / "datasets"
OUTPUTS_DIR = ROOT_DIR / "outputs"

# 可视化目录
VIS_DIR = OUTPUTS_DIR / "visualizations"

# 日志目录
LOGS_DIR = OUTPUTS_DIR / "logs"

# checkpoint目录
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"


def ensure_dir_exists(dir_path: Path):
    """确保目录存在，如果不存在则创建"""
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_model_vis_dir(model_name: str) -> Path:
    """获取模型的可视化目录"""
    return ensure_dir_exists(VIS_DIR / model_name)


def get_model_checkpoint_dir(model_name: str) -> Path:
    """获取模型的checkpoint目录"""
    return ensure_dir_exists(CHECKPOINTS_DIR / model_name)


def get_model_log_dir(model_name: str) -> Path:
    """获取模型的日志目录"""
    return ensure_dir_exists(LOGS_DIR / model_name)


def get_best_model_path(model_name: str) -> Path:
    """获取模型的最佳checkpoint路径"""
    checkpoint_dir = get_model_checkpoint_dir(model_name)
    best_model_path = checkpoint_dir / "best_model.pth"
    return best_model_path


def get_last_model_path(model_name: str) -> Path:
    """获取模型的最新checkpoint路径"""
    checkpoint_dir = get_model_checkpoint_dir(model_name)
    last_model_path = checkpoint_dir / "last_model.pth"
    return last_model_path


if __name__ == "__main__":
    print(ROOT_DIR)
    print(get_model_checkpoint_dir("resnet50"))
