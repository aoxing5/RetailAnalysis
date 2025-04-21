import os
import logging
import pandas as pd
import ast
from src.analysis.retail_analyzer import RetailAnalyzer
from src.analysis.product_analysis import ProductAnalysis
from src.analysis.transaction_analysis import TransactionAnalysis
from src.analysis.pattern_analysis import PatternAnalysis
from src.analysis.visualization import (plot_seasonal_trends,
                                  plot_basket_analysis, plot_rfm_analysis,
                                  plot_monthly_trends,
                                  # 保留对应新建文本文档.txt中的图表
                                  plot_price_quantity_scatter, plot_transaction_scatter,
                                  plot_product_distribution_histogram,
                                  plot_product_wordcloud, plot_customer_wordcloud,
                                  plot_product_trends,
                                  plot_category_comparison,
                                  plot_seasonal_category_heatmap,
                                  plot_hourly_sales)
from src.utils.file_utils import save_to_file

def main():
    logging.info("启动零售分析程序")
    try:
        # 加载数据文件
        data_file = os.path.join('data', 'Retail_Transactions_Dataset.csv')
        analyzer = RetailAnalyzer(data_file=data_file)
        df = analyzer.df  # 加载的数据

        # 随机抽样 10,000 条数据
        df = df.sample(n=10000, random_state=42)

        # 转换 Product 列为列表并展平
        def flatten_product(x):
            try:
                if isinstance(x, str):
                    products = ast.literal_eval(x)
                else:
                    products = x
                return [str(p).strip() for p in products if p]
            except:
                return ['Unknown']

        # 处理 Product 列
        df['Product'] = df['Product'].apply(flatten_product)

        # 优化数据类型
        df['Transaction_ID'] = df['Transaction_ID'].astype('int32')
        df['Total_Items'] = df['Total_Items'].astype('int16')
        df['Total_Cost'] = df['Total_Cost'].astype('float32')

        # 转换日期列为日期时间类型
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        # 为可视化准备数据：添加季节列和月份列
        df['Month'] = df['Date'].dt.month
        df['Season'] = pd.cut(df['Month'], bins=[0, 3, 6, 9, 12], 
                              labels=['春季', '夏季', '秋季', '冬季'], include_lowest=True)

        # 先进行基础分析
        basic_info = analyzer.basic_info()
        transaction_info = analyzer.transaction_analysis()

        # 商品分析前展开数据
        exploded_df = df.explode('Product')
        # 确保展开的数据框也有Month列
        if 'Month' not in exploded_df.columns and 'Date' in exploded_df.columns:
            exploded_df['Month'] = exploded_df['Date'].dt.month
            
        product_analyzer = ProductAnalysis(exploded_df)
        product_stats = product_analyzer.basic_product_stats()
        sales_ranking = product_analyzer.sales_ranking(product_stats)
        price_analysis = product_analyzer.price_analysis()
        
        # 获取复购率数据
        repurchase_rate = product_analyzer.repurchase_rate()
        
        # 月度趋势分析
        monthly_trend = product_analyzer.monthly_trend(product_stats)

        # 模式分析
        pattern_analyzer = PatternAnalysis(df)
        rfm = pattern_analyzer.rfm_analysis()
        
        # 购物篮分析使用展开后的数据
        basket_data = exploded_df[['Transaction_ID', 'Product']]
        frequent_itemsets, rules = pattern_analyzer.basket_analysis(basket_data, min_support=0.01)

        # 保存分析结果
        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)
        save_to_file(basic_info, os.path.join(output_dir, 'basic_info.txt'))
        save_to_file(transaction_info, os.path.join(output_dir, 'transaction_info.txt'))
        save_to_file(product_stats, os.path.join(output_dir, 'product_stats.txt'))
        save_to_file(sales_ranking, os.path.join(output_dir, 'sales_ranking.txt'))
        save_to_file(price_analysis, os.path.join(output_dir, 'price_analysis.txt'))
        save_to_file(rfm, os.path.join(output_dir, 'rfm_analysis.txt'))
        save_to_file(frequent_itemsets, os.path.join(output_dir, 'frequent_itemsets.txt'))
        save_to_file(rules, os.path.join(output_dir, 'association_rules.txt'))

        # 创建图形输出目录
        figures_dir = os.path.join(output_dir, 'figures')
        os.makedirs(figures_dir, exist_ok=True)

        # 生成可视化图表
        logging.info("开始生成可视化图表...")
        
        try:
            # 1. 商品价格与销量关系散点图
            logging.info("生成商品价格与销量关系散点图...")
            plot_price_quantity_scatter(exploded_df, figures_dir)
            
            # 2. 商品价格分布直方图
            logging.info("生成商品价格分布直方图...")
            plot_product_distribution_histogram(exploded_df, figures_dir)
            
            # 3. RFM特征相关性热力图
            logging.info("生成RFM分析图表...")
            plot_rfm_analysis(rfm, figures_dir)
            
            # 4. 季节销售趋势
            logging.info("生成季节销售趋势图...")
            plot_seasonal_trends(exploded_df, 'Season', 'Total_Cost', figures_dir)
            
            # 5. 热门商品词云图
            logging.info("生成商品词云图...")
            plot_product_wordcloud(exploded_df, figures_dir)
            
            # 6. 购物篮分析热力图
            logging.info("生成购物篮分析热力图...")
            top_transactions = exploded_df['Transaction_ID'].value_counts().nlargest(20).index.tolist()
            top_products = exploded_df.groupby('Product')['Total_Cost'].sum().nlargest(10).reset_index()
            top_products_list = top_products['Product'].tolist()
            basket_viz_data = exploded_df[
                (exploded_df['Transaction_ID'].isin(top_transactions)) & 
                (exploded_df['Product'].isin(top_products_list))
            ]
            plot_basket_analysis(basket_viz_data, figures_dir)
            
            # 7. 热门商品月度销售趋势
            logging.info("生成月度销售趋势图...")
            plot_monthly_trends(monthly_trend, figures_dir)
            
            # 8. 商品类别销售分析比较
            logging.info("生成销售类别比较图...")
            plot_category_comparison(exploded_df, figures_dir)
            
            # 9. 交易金额与商品数量关系散点图
            logging.info("生成交易金额与商品数量关系散点图...")
            plot_transaction_scatter(df, figures_dir)
            
            # 10. 客户消费词云图
            logging.info("生成客户词云图...")
            plot_customer_wordcloud(exploded_df, figures_dir)
            
            # 11. 热门商品销售额排名
            logging.info("生成商品销售排名图...")
            top_products = exploded_df.groupby('Product')['Total_Cost'].sum().nlargest(10).reset_index()
            plot_product_trends(top_products, 'Product', 'Total_Cost', figures_dir)
            
            # 12. 季节-商品类别销售热力图
            logging.info("生成季节-类别热力图...")
            plot_seasonal_category_heatmap(exploded_df, figures_dir)
            
            # 13. 24小时销售分析
            logging.info("生成24小时销售分析...")
            plot_hourly_sales(exploded_df, figures_dir)
            
        except Exception as e:
            logging.error(f"生成可视化图表过程中出现错误: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            
        logging.info("所有分析完成，结果已保存到 {} 目录".format(output_dir))
        logging.info("可视化图表已保存到 {} 目录".format(figures_dir))
    except Exception as e:
        logging.error(f"分析过程中出现错误: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join('logs', 'analysis.log'), encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    # 确保日志目录存在
    os.makedirs('logs', exist_ok=True)
    main()