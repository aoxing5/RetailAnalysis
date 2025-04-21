import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

class PatternAnalysis:
    def __init__(self, df):
        self.df = df

    def rfm_analysis(self):
        if all(col in self.df.columns for col in ['Customer_Name', 'Date', 'Total_Cost']):
            last_date = self.df['Date'].max()
            rfm = self.df.groupby('Customer_Name').agg({
                'Date': lambda x: (last_date - x.max()).days,  # Recency
                'Transaction_ID': 'count',  # Frequency
                'Total_Cost': 'sum'  # Monetary
            }).rename(columns={
                'Date': 'Recency',
                'Transaction_ID': 'Frequency',
                'Total_Cost': 'Monetary'
            })
            return rfm
        else:
            raise ValueError("缺少必要的列进行RFM分析")

    def basket_analysis(self, basket_data, min_support=0.02, min_confidence=0.3):
        """购物篮分析"""
        if 'Transaction_ID' not in basket_data.columns or 'Product' not in basket_data.columns:
            raise ValueError("数据框中缺少必要的列进行购物篮分析")

        # 创建购物篮矩阵
        basket_matrix = pd.crosstab(basket_data['Transaction_ID'], basket_data['Product'])
        
        # 添加这一行新代码 - 将矩阵转换为布尔类型
        basket_matrix = basket_matrix > 0
        
        # 使用 apriori 算法进行频繁项集挖掘
        frequent_itemsets = apriori(basket_matrix, min_support=min_support, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

        return frequent_itemsets, rules