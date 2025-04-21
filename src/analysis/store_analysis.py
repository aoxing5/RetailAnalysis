import pandas as pd
import os

class StoreAnalysis:
    def __init__(self, dataframe):
        self.df = dataframe

    def store_performance(self):
        """分析每个店铺的整体表现"""
        if 'Store_ID' in self.df.columns:
            store_stats = self.df.groupby('Store_ID').agg({
                'Total_Cost': ['count', 'sum', 'mean'],
                'Customer_Name': 'nunique' if 'Customer_Name' in self.df.columns else 'count',
                'Product': 'nunique' if 'Product' in self.df.columns else 'count'
            })
            return store_stats
        else:
            raise ValueError("数据框中缺少 'Store_ID' 列")

    def store_sales_ranking(self):
        """获取店铺销售额排名"""
        store_stats = self.store_performance()
        return store_stats.nlargest(5, ('Total_Cost', 'sum'))

    def store_daily_performance(self):
        """分析每个店铺的日均表现"""
        if 'Date' in self.df.columns:
            store_daily = self.df.groupby(['Store_ID', 'Date']).agg({
                'Transaction_ID': 'count',
                'Total_Cost': 'sum'
            }).reset_index()
            return store_daily.groupby('Store_ID').agg({
                'Transaction_ID': 'mean',
                'Total_Cost': 'mean'
            }).round(2)
        else:
            raise ValueError("数据框中缺少 'Date' 列")