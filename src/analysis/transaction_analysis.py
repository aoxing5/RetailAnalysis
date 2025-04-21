from typing import Any
import pandas as pd

class TransactionAnalysis:
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe

    def basic_transaction_info(self) -> None:
        """基础交易信息分析"""
        print("\n交易数量分析")
        print(f"总交易数: {len(self.df)}")
        print(f"唯一交易ID数: {self.df['Transaction_ID'].nunique()}")

    def time_analysis(self) -> None:
        """时间分析"""
        self.df['Year'] = pd.to_datetime(self.df['Date']).dt.year
        self.df['Month'] = pd.to_datetime(self.df['Date']).dt.month
        self.df['Day'] = pd.to_datetime(self.df['Date']).dt.day
        self.df['Hour'] = pd.to_datetime(self.df['Date']).dt.hour
        self.df['Weekday'] = pd.to_datetime(self.df['Date']).dt.dayofweek

        print("\n年度交易统计:")
        yearly_stats = self.df.groupby('Year')['Transaction_ID'].count()
        print(yearly_stats)

        print("\n月度交易统计:")
        monthly_stats = self.df.groupby('Month')['Transaction_ID'].count()
        print(monthly_stats)

        print("\n星期交易统计:")
        weekday_stats = self.df.groupby('Weekday')['Transaction_ID'].count()
        print(weekday_stats)

        print("\n小时交易统计:")
        hourly_stats = self.df.groupby('Hour')['Transaction_ID'].count()
        print(hourly_stats)

    def promotion_analysis(self) -> None:
        """促销分析"""
        if 'Promotion' in self.df.columns:
            promo_stats = self.df.groupby('Promotion')['Transaction_ID'].count()
            print("\n促销活动统计:")
            print(promo_stats)

    def run_analysis(self) -> None:
        """运行所有交易分析"""
        self.basic_transaction_info()
        self.time_analysis()
        self.promotion_analysis()