import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path: str) -> pd.DataFrame:
    """加载零售交易数据"""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        raise Exception(f"文件未找到: {file_path}")
    except Exception as e:
        raise Exception(f"加载数据时发生错误: {str(e)}")

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """数据预处理"""
    # 转换日期列
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    
    # 填充缺失值
    df.fillna(method='ffill', inplace=True)
    
    # 标准化数值列
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df

def save_preprocessed_data(df: pd.DataFrame, output_path: str):
    """保存预处理后的数据"""
    df.to_csv(output_path, index=False)