from .dataset_ct import DatasetCT # 导入 CTGSDataset

def get_dataset(data_path, type):
    """
    Args:
        data_path (str): 数据根目录路径
        type (str): 'train' 或 'test'，指定加载哪个数据集
    Returns:
        DatasetCT: 数据集对象
    """
    assert type in ['train', 'test'], "type must be 'train' or 'test'"
    return DatasetCT(data_path, type)