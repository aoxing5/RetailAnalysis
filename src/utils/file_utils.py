import pandas as pd
import os

def read_csv_file(file_path: str) -> pd.DataFrame:
    """读取CSV文件并返回DataFrame"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件未找到: {file_path}")
    return pd.read_csv(file_path)

def save_dataframe_to_csv(df: pd.DataFrame, file_path: str, index: bool = False):
    """将DataFrame保存为CSV文件"""
    df.to_csv(file_path, index=index)
    
def create_directory(dir_path: str):
    """创建目录，如果不存在的话"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def save_to_file(data, file_path):
    """将数据保存到文件"""
    try:
        if isinstance(data, pd.DataFrame):
            data.to_csv(file_path, index=False, encoding='utf-8')
        elif isinstance(data, dict):
            with open(file_path, 'w', encoding='utf-8') as f:
                for key, value in data.items():
                    f.write(f"{key}: {value}\n")
        elif isinstance(data, list):
            with open(file_path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(f"{item}\n")
        else:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(str(data))
    except Exception as e:
        raise IOError(f"保存文件 {file_path} 时出错: {str(e)}")