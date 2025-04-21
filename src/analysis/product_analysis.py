import pandas as pd

class ProductAnalysis:
    def __init__(self, df):
        self.df = df

    def basic_product_stats(self):
        """基础商品统计"""
        product_basic_stats = self.df.groupby('Product').agg({
            'Transaction_ID': 'count',  # 交易次数
            'Total_Cost': ['sum', 'mean', 'std'],  # 销售金额统计
            'Customer_Name': 'nunique'  # 购买客户数
        }).round(2)

        product_basic_stats.columns = ['交易次数', '总销售额', '平均销售额', '销售额标准差', '购买客户数']
        return product_basic_stats

    def sales_ranking(self, product_basic_stats):
        """销售排名分析"""
        top_sales = product_basic_stats.nlargest(10, '总销售额')
        top_transactions = product_basic_stats.nlargest(10, '交易次数')
        top_customers = product_basic_stats.nlargest(10, '购买客户数')
        return top_sales, top_transactions, top_customers

    def price_analysis(self):
        """价格分析"""
        product_price = self.df.groupby('Product')['Total_Cost'].agg(['mean', 'std', 'min', 'max']).round(2)
        product_price['price_variation'] = product_price['std'] / product_price['mean']
        return product_price.nlargest(10, 'price_variation')

    def monthly_trend(self, product_basic_stats):
        """商品时间维度分析"""
        monthly_product = self.df.groupby(['Product', 'Month']).agg({
            'Total_Cost': ['count', 'sum'],
            'Customer_Name': 'nunique'
        }).round(2)

        top_products = product_basic_stats.nlargest(5, '总销售额').index
        monthly_trend = monthly_product.loc[top_products]
        return monthly_trend

    def repurchase_rate(self):
        """商品复购率分析"""
        product_repurchase = self.df.groupby('Product').agg({
            'Customer_Name': lambda x: len(x[x.duplicated()]) / len(x.unique())
        }).round(3)
        return product_repurchase.nlargest(10, 'Customer_Name')