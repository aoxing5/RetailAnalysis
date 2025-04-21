import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import platform
import matplotlib as mpl
from wordcloud import WordCloud
import matplotlib.patches as mpatches


def configure_chinese_font():
    system = platform.system()
    if system == 'Windows':
        # Windows系统通常可用的中文字体
        font_options = ['SimHei', 'Microsoft YaHei', 'SimSun', 'NSimSun', 'FangSong', 'KaiTi']
        for font in font_options:
            try:
                plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
                plt.rcParams['axes.unicode_minus'] = False
                test_str = '测试中文'
                fig = plt.figure(figsize=(1, 1))
                plt.text(0.5, 0.5, test_str)
                plt.close(fig)
                print(f"使用中文字体: {font}")
                return font
            except:
                continue
    else:
        # Linux/Mac系统
        font_options = ['WenQuanYi Zen Hei', 'Hiragino Sans GB', 'Heiti TC', 'STHeiti']
        for font in font_options:
            try:
                plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
                plt.rcParams['axes.unicode_minus'] = False
                return font
            except:
                continue
    
    # 如果没有合适的中文字体，我们会使用默认字体，但可能无法正确显示中文
    return None

chinese_font = configure_chinese_font()

# 启动时配置中文字体
chinese_font = configure_chinese_font()


def plot_seasonal_trends(data: pd.DataFrame, season_col: str, sales_col: str, output_dir: str):
    """绘制季节销售趋势图"""
    plt.figure(figsize=(10, 6))
    seasonal_sales = data.groupby(season_col, observed=True)[sales_col].sum().reset_index()
    ax = sns.barplot(x=season_col, y=sales_col, data=seasonal_sales, hue=season_col, legend=False)
    
    plt.title('季节销售趋势')
    plt.xlabel('季节')
    plt.ylabel('销售总额')
    
    for i in ax.containers:
        ax.bar_label(i, fmt='%.0f')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'seasonal_trends.png'))
    plt.close()


def plot_product_trends(data: pd.DataFrame, product_col: str, sales_col: str, output_dir: str):
    """生成商品销售趋势图"""
    plt.figure(figsize=(14, 7))
    
    colors = sns.color_palette('viridis', len(data))
    bars = plt.bar(data[product_col], data[sales_col], color=colors)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.title('热门商品销售额排名', fontsize=15)
    plt.xlabel('商品', fontsize=12)
    plt.ylabel('销售额', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'product_trends.png'), dpi=300)
    plt.close()


def plot_basket_analysis(data: pd.DataFrame, output_dir: str):
    """生成购物篮分析热力图"""
    basket_data = pd.crosstab(data['Transaction_ID'], data['Product'])
    
    # 为了更好的可视化效果，将数据二值化
    basket_binary = (basket_data > 0).astype(int)
    
    plt.figure(figsize=(14, 10))
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#FFFFFF', '#3498db'])
    heatmap = sns.heatmap(basket_binary, cmap=custom_cmap, cbar_kws={'label': '商品是否存在'})
    
    plt.title('购物篮分析热力图 (热门交易 x 热门商品)', fontsize=15)
    plt.xlabel('商品', fontsize=12)
    plt.ylabel('交易ID', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'basket_analysis_heatmap.png'), dpi=300)
    plt.close()


def plot_rfm_analysis(rfm_data: pd.DataFrame, output_dir: str):
    """可视化RFM分析"""
    # RFM 热力图
    plt.figure(figsize=(12, 8))
    correlation_matrix = rfm_data[['Recency', 'Frequency', 'Monetary']].corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    sns.heatmap(correlation_matrix, annot=True, mask=mask, cmap='coolwarm', vmin=-1, vmax=1, 
                annot_kws={"size": 14}, square=True, linewidths=.5)
    
    plt.title('RFM 特征相关性热力图', fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rfm_correlation.png'), dpi=300)
    plt.close()


def plot_monthly_trends(monthly_data: pd.DataFrame, output_dir: str):
    """月度销售趋势可视化"""
    # 检查数据结构并重置多级索引
    try:
        monthly_data_reset = monthly_data.reset_index()
        
        # 获取Total_Cost和sum所在的列
        sum_col = None
        for col in monthly_data_reset.columns:
            if isinstance(col, tuple) and col[0] == 'Total_Cost' and col[1] == 'sum':
                sum_col = col
                break
        
        if sum_col is None:
            # 如果找不到特定的列结构，尝试自动检测
            for col in monthly_data_reset.columns:
                if isinstance(col, tuple) and 'sum' in col:
                    sum_col = col
                    break
        
        # 创建适合绘图的数据格式
        plot_data = pd.DataFrame({
            'Product': monthly_data_reset['Product'],
            'Month': monthly_data_reset['Month'],
            'Sales': monthly_data_reset[sum_col]
        })
        
        # 绘制线图
        plt.figure(figsize=(14, 8))
        
        # 分组计算每个产品每月的销售额
        sns.lineplot(x='Month', y='Sales', hue='Product', data=plot_data, marker='o', linewidth=2)
        
        plt.title('热门商品月度销售趋势', fontsize=15)
        plt.xlabel('月份', fontsize=12)
        plt.ylabel('销售额', fontsize=12)
        plt.grid(linestyle='--', alpha=0.7)
        plt.legend(title='商品', title_fontsize=12, fontsize=10, loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'monthly_trends.png'), dpi=300)
        plt.close()
    except Exception as e:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f"无法生成月度趋势图: {str(e)}", 
                ha='center', va='center', fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'monthly_trends_error.png'), dpi=300)
        plt.close()
        print(f"月度趋势图生成错误: {str(e)}")


def extract_product_category(product_name):
    """从商品名称中提取更有意义的类别"""
    if not isinstance(product_name, str):
        return "其他"
    
    # 常见商品类别关键词映射
    category_keywords = {
        "食品": ["food", "snack", "meal", "drink", "beverage", "果", "菜", "肉", "奶", "茶", "酒", "咖啡", "水", "饮料", "零食"],
        "电子产品": ["phone", "computer", "laptop", "电脑", "手机", "相机", "充电器", "耳机", "音响", "电视", "显示器"],
        "家居用品": ["furniture", "home", "kitchen", "bathroom", "bed", "chair", "table", "sofa", "desk", "家居", "厨房", "卫生间", "床", "椅", "桌"],
        "服装": ["clothing", "shirt", "pants", "dress", "jacket", "coat", "shoes", "socks", "衣", "裤", "鞋", "袜", "帽"],
        "美妆": ["cosmetic", "makeup", "lipstick", "foundation", "mascara", "护肤", "美容", "化妆", "面膜"],
        "书籍": ["book", "novel", "magazine", "书", "杂志", "小说"],
        "办公用品": ["office", "pen", "paper", "notebook", "pencil", "文具", "笔", "纸", "本"],
        "运动健身": ["sport", "fitness", "exercise", "gym", "运动", "健身"]
    }
    
    product_lower = product_name.lower()
    
    # 遍历关键词映射
    for category, keywords in category_keywords.items():
        for keyword in keywords:
            if keyword.lower() in product_lower:
                return category
    
    # 如果没有匹配的关键词，使用第一个单词作为类别
    words = product_lower.split()
    if words:
        return words[0].capitalize()
    else:
        return "其他"


def plot_price_quantity_scatter(data: pd.DataFrame, output_dir: str):
    """商品价格与销量关系散点图"""
    try:
        # 计算每个商品的平均价格和销售量
        product_stats = data.groupby('Product').agg({
            'Total_Cost': 'mean',  # 平均价格
            'Product': 'count'  # 销售量
        }).rename(columns={'Product': 'Sales_Count'}).reset_index()
        
        plt.figure(figsize=(12, 8))
        
        # 创建散点图
        scatter = plt.scatter(
            product_stats['Total_Cost'], 
            product_stats['Sales_Count'],
            c=product_stats['Total_Cost'],  # 颜色映射基于价格
            cmap='viridis',
            alpha=0.7,
            s=80  # 点的大小
        )
        
        # 添加回归线
        sns.regplot(
            x='Total_Cost', 
            y='Sales_Count', 
            data=product_stats, 
            scatter=False, 
            color='red',
            line_kws={"linestyle": "--"}
        )
        
        # 添加颜色条
        plt.colorbar(scatter, label='平均价格')
        
        # 标记一些重要的点
        for idx, row in product_stats.nlargest(5, 'Sales_Count').iterrows():
            plt.annotate(
                row['Product'],
                (row['Total_Cost'], row['Sales_Count']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
            )
        
        plt.title('商品价格与销量关系散点图', fontsize=15)
        plt.xlabel('平均价格', fontsize=12)
        plt.ylabel('销售量', fontsize=12)
        plt.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'price_quantity_scatter.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"生成商品价格与销量关系散点图时出错: {str(e)}")


def plot_transaction_scatter(data: pd.DataFrame, output_dir: str):
    """交易金额与商品数量关系散点图"""
    try:
        plt.figure(figsize=(12, 8))
        
        # 创建散点图
        scatter = plt.scatter(
            data['Total_Items'], 
            data['Total_Cost'],
            c=data['Total_Cost'],  # 颜色映射基于价格
            cmap='magma',
            alpha=0.7,
            s=60  # 点的大小
        )
        
        # 添加回归线
        sns.regplot(
            x='Total_Items', 
            y='Total_Cost', 
            data=data, 
            scatter=False, 
            color='red',
            line_kws={"linestyle": "--"}
        )
        
        # 添加颜色条
        plt.colorbar(scatter, label='交易金额')
        
        plt.title('交易金额与商品数量关系散点图', fontsize=15)
        plt.xlabel('商品数量', fontsize=12)
        plt.ylabel('交易金额', fontsize=12)
        plt.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'transaction_scatter.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"生成交易金额与商品数量关系散点图时出错: {str(e)}")


def plot_product_wordcloud(data: pd.DataFrame, output_dir: str):
    """商品词云图"""
    try:
        # 计算每个商品的频率
        product_freq = data['Product'].value_counts().to_dict()
        
        # 只保留频率最高的50个商品（减少词云密度）
        if len(product_freq) > 50:
            top_products = dict(sorted(product_freq.items(), key=lambda x: x[1], reverse=True)[:50])
            product_freq = top_products
        
        # 创建词云
        wordcloud = WordCloud(
            width=1000, 
            height=600,
            background_color='white',
            max_words=50,  # 减少单词数量
            colormap='viridis',
            font_path='simhei.ttf' if os.path.exists('simhei.ttf') else None,
            contour_width=1,
            contour_color='steelblue',
            prefer_horizontal=0.9,  # 90%的词水平显示
            min_font_size=10,  # 最小字体大小
            max_font_size=120,  # 最大字体大小
            scale=3,  # 增加缩放比例
            relative_scaling=0.5,  # 减小词频和字体大小的相关性
            margin=10,  # 增加词与词之间的距离
            collocations=False  # 避免重复词语
        ).generate_from_frequencies(product_freq)
        
        plt.figure(figsize=(16, 9))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('热门商品词云图', fontsize=22, pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'product_wordcloud.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"生成商品词云图时出错: {str(e)}")


def plot_customer_wordcloud(data: pd.DataFrame, output_dir: str):
    """客户词云图（基于消费金额）"""
    try:
        # 计算每个客户的消费总额
        customer_spending = data.groupby('Customer_Name')['Total_Cost'].sum().to_dict()
        
        # 只保留消费金额最高的40个客户（减少词云密度）
        if len(customer_spending) > 40:
            top_customers = dict(sorted(customer_spending.items(), key=lambda x: x[1], reverse=True)[:40])
            customer_spending = top_customers
        
        # 创建词云
        wordcloud = WordCloud(
            width=1000, 
            height=600,
            background_color='white',
            max_words=40,  # 减少单词数量
            colormap='plasma',
            font_path='simhei.ttf' if os.path.exists('simhei.ttf') else None,
            contour_width=1,
            contour_color='steelblue',
            prefer_horizontal=0.9,  # 90%的词水平显示
            min_font_size=10,  # 最小字体大小
            max_font_size=120,  # 最大字体大小
            scale=3,  # 增加缩放比例
            relative_scaling=0.5,  # 减小词频和字体大小的相关性
            margin=10,  # 增加词与词之间的距离
            collocations=False  # 避免重复词语
        ).generate_from_frequencies(customer_spending)
        
        plt.figure(figsize=(16, 9))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('客户消费词云图（字体大小基于消费金额）', fontsize=22, pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'customer_wordcloud.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"生成客户词云图时出错: {str(e)}")


def plot_product_distribution_histogram(data: pd.DataFrame, output_dir: str):
    """商品价格分布直方图"""
    try:
        plt.figure(figsize=(12, 8))
        
        # 创建直方图
        sns.histplot(
            data['Total_Cost'], 
            bins=30, 
            kde=True, 
            color='skyblue',
            line_kws={'linewidth': 2}
        )
        
        # 添加均值线
        mean_price = data['Total_Cost'].mean()
        plt.axvline(mean_price, color='red', linestyle='--', linewidth=2, label=f'均值: {mean_price:.2f}')
        
        # 添加中位数线
        median_price = data['Total_Cost'].median()
        plt.axvline(median_price, color='green', linestyle='--', linewidth=2, label=f'中位数: {median_price:.2f}')
        
        plt.title('商品价格分布直方图', fontsize=15)
        plt.xlabel('价格', fontsize=12)
        plt.ylabel('频率', fontsize=12)
        plt.legend()
        plt.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'price_distribution_histogram.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"生成商品价格分布直方图时出错: {str(e)}")


def plot_category_comparison(data: pd.DataFrame, output_dir: str):
    """销售类别比较图 - 同时展示销售额和销售量"""
    try:
        # 使用智能分类方法
        data['Product_Category'] = data['Product'].apply(extract_product_category)
        
        # 按类别分组计算销售额和销售量
        category_stats = data.groupby('Product_Category').agg({
            'Total_Cost': 'sum',  # 销售额
            'Product': 'count'  # 销售量
        }).rename(columns={'Product': 'Sales_Count'})
        
        # 选择前8个类别
        top_categories = category_stats.nlargest(8, 'Total_Cost')
        
        # 创建子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # 左侧图 - 销售额饼图
        wedges, texts, autotexts = ax1.pie(
            top_categories['Total_Cost'],
            labels=None,
            autopct='%1.1f%%',
            startangle=90,
            colors=plt.cm.tab20c(np.linspace(0, 1, len(top_categories))),
            wedgeprops={'edgecolor': 'w', 'linewidth': 1}
        )
        
        # 设置饼图标题和图例
        ax1.set_title('销售额占比', fontsize=16)
        ax1.legend(
            wedges,
            top_categories.index,
            title="商品类别",
            loc="center left",
            bbox_to_anchor=(0.9, 0, 0.5, 1)
        )
        
        # 右侧图 - 销售量条形图
        bars = ax2.bar(
            top_categories.index,
            top_categories['Sales_Count'],
            color=plt.cm.tab20c(np.linspace(0, 1, len(top_categories)))
        )
        
        # 添加数据标签
        for bar in bars:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width()/2.,
                height + 5,
                f'{int(height)}',
                ha='center',
                va='bottom',
                fontsize=10
            )
        
        # 设置条形图标题和标签
        ax2.set_title('销售量排名', fontsize=16)
        ax2.set_ylabel('销售数量 (件)', fontsize=12)
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=45, ha='right')
        
        # 添加总体标题
        fig.suptitle('商品类别销售分析比较', fontsize=20, fontweight='bold')
        
        # 添加注释说明
        fig.text(
            0.5, 0.01,
            '该图表展示了销售额最高的8个商品类别的销售额占比与销售数量对比',
            ha='center',
            fontsize=12,
            bbox=dict(facecolor='lavender', alpha=0.5, boxstyle='round,pad=0.5')
        )
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(output_dir, 'category_comparison.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"生成销售类别比较图时出错: {str(e)}")


def plot_seasonal_category_heatmap(data: pd.DataFrame, output_dir: str):
    """绘制季节-商品类别销售热力图"""
    seasonal_category = data.pivot_table(
        values='Total_Cost',
        index='Season',
        columns='Product',
        aggfunc='sum',
        observed=True
    )
    
    plt.figure(figsize=(15, 8))
    sns.heatmap(seasonal_category, cmap='YlOrRd', annot=True, fmt='.0f')
    
    plt.title('季节-商品类别销售热力图')
    plt.xlabel('商品类别')
    plt.ylabel('季节')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'seasonal_category_heatmap.png'))
    plt.close()


def plot_hourly_sales(data: pd.DataFrame, output_dir: str):
    """分析24小时内各时段的商品销售情况"""
    try:
        # 确保日期列是datetime类型
        if 'Date' not in data.columns:
            print("数据中缺少Date列，无法进行时间分析")
            return
            
        # 添加小时列
        data['Hour'] = data['Date'].dt.hour
        
        # 按小时和商品分组计算销售情况
        hourly_sales = data.groupby(['Hour', 'Product']).agg({
            'Total_Cost': ['sum', 'count']
        }).reset_index()
        
        hourly_sales.columns = ['Hour', 'Product', 'Total_Sales', 'Sales_Count']
        
        # 获取每个小时销售额最高的商品
        top_products_by_hour = hourly_sales.sort_values('Total_Sales', ascending=False).groupby('Hour').first().reset_index()
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 20))
        
        # 1. 销售额分析
        bars1 = ax1.barh(top_products_by_hour['Hour'].astype(str) + '时 - ' + top_products_by_hour['Product'], 
                        top_products_by_hour['Total_Sales'],
                        color='skyblue')
        ax1.set_title('24小时内各时段销售额最高的商品', fontsize=14, pad=20)
        ax1.set_xlabel('销售额 (元)', fontsize=12)
        ax1.set_ylabel('时段 - 商品', fontsize=12)
        
        # 在条形上添加数值标签
        for bar in bars1:
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2,
                    f'¥{width:,.0f}',
                    ha='left', va='center', fontsize=10)
        
        # 2. 销售量分析
        bars2 = ax2.barh(top_products_by_hour['Hour'].astype(str) + '时 - ' + top_products_by_hour['Product'],
                        top_products_by_hour['Sales_Count'],
                        color='lightgreen')
        ax2.set_title('24小时内各时段销售量最高的商品', fontsize=14, pad=20)
        ax2.set_xlabel('销售数量 (件)', fontsize=12)
        ax2.set_ylabel('时段 - 商品', fontsize=12)
        
        # 在条形上添加数值标签
        for bar in bars2:
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2,
                    f'{int(width)}件',
                    ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'hourly_sales.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存详细数据到文本文件
        with open(os.path.join(output_dir, 'hourly_sales.txt'), 'w', encoding='utf-8') as f:
            f.write("24小时销售分析报告\n")
            f.write("=" * 50 + "\n\n")
            for _, row in top_products_by_hour.sort_values('Hour').iterrows():
                f.write(f"时段：{row['Hour']:02d}:00-{row['Hour']:02d}:59\n")
                f.write(f"最畅销商品：{row['Product']}\n")
                f.write(f"销售额：¥{row['Total_Sales']:,.2f}\n")
                f.write(f"销售数量：{int(row['Sales_Count'])}件\n")
                f.write("-" * 50 + "\n")
                
    except Exception as e:
        print(f"生成24小时销售分析图表时出错: {str(e)}")
        import traceback
        print(traceback.format_exc())