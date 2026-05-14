"""
数据加载与预处理默认超参数
"""


class DataParams:
    """
    数据加载与预处理默认超参数
    """

    BATCH_SIZE: int = 128
    NUM_WORKERS: int = 4
    PIN_MEMORY: bool = True  # 是否将数据加载到GPU的内存中以加速训练
    VAL_SPLIT: float = 0.1

    ENABLE_DATA_AUGMENTATION: bool = True  # 是否启用数据增强
    ROTATION_DEGREES: int = 15  # 数据增强中的旋转角度范围
    TRANSLATION_RATIO: float = 0.1  # 数据增强中的平移比例
