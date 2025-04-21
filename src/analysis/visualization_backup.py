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

# 配置中文字体
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

# 启动时配置中文字体
chinese_font = configure_chinese_font()

def plot_seasonal_trends(data: pd.DataFrame, season_col: str, sales_col: str, output_dir: str):
    """生成季节销售趋势图"""
    plt.figure(figsize=(12, 6))
    seasonal_sales = data.groupby(season_col)[sales_col].sum().reset_index()
    ax = sns.barplot(x=season_col, y=sales_col, data=seasonal_sales, palette='viridis')
    
    # 添加数值标签
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='bottom',
                   fontsize=11, color='black')
    
    plt.title('季节销售趋势', fontsize=15)
    plt.xlabel('季节', fontsize=12)
    plt.ylabel('销售额', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'seasonal_trends.png'), dpi=300)
    plt.close()

def plot_promotion_effect(data: pd.DataFrame, promotion_col: str, sales_col: str, output_dir: str):
    """生成促销活动效果图"""
    plt.figure(figsize=(12, 6))
    promotion_effect = data.groupby(promotion_col)[sales_col].sum().reset_index()
    ax = sns.barplot(x=promotion_col, y=sales_col, data=promotion_effect, palette='magma')
    
    # 添加数值标签
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='bottom',
                   fontsize=11, color='black')
    
    plt.title('促销活动效果', fontsize=15)
    plt.xlabel('促销活动', fontsize=12)
    plt.ylabel('销售额', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'promotion_effect.png'), dpi=300)
    plt.close()

def plot_customer_activity(data: pd.DataFrame, customer_col: str, sales_col: str, output_dir: str):
    """生成客户活跃度图"""
    plt.figure(figsize=(12, 6))
    customer_activity = data.groupby(customer_col)[sales_col].sum().reset_index()
    sns.histplot(customer_activity[sales_col], bins=30, kde=True, color='purple')
    plt.title('客户消费分布', fontsize=15)
    plt.xlabel('消费金额', fontsize=12)
    plt.ylabel('客户数量', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'customer_activity.png'), dpi=300)
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
    # 将RFM值标准化到0-1范围
    rfm_normalized = rfm_data.copy()
    for col in ['Recency', 'Frequency', 'Monetary']:
        if col in rfm_normalized.columns:
            rfm_normalized[col] = (rfm_normalized[col] - rfm_normalized[col].min()) / (rfm_normalized[col].max() - rfm_normalized[col].min())
    
    # 1. 3D散点图
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(
        rfm_normalized['Recency'].values, 
        rfm_normalized['Frequency'].values, 
        rfm_normalized['Monetary'].values,
        c=rfm_normalized['Monetary'].values,
        cmap='viridis',
        s=50,
        alpha=0.7
    )
    
    plt.colorbar(scatter, ax=ax, label='标准化货币价值')
    ax.set_xlabel('近期度 (Recency)')
    ax.set_ylabel('频率 (Frequency)')
    ax.set_zlabel('金额 (Monetary)')
    ax.set_title('RFM 分析 3D 散点图')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rfm_3d_scatter.png'), dpi=300)
    plt.close()
    
    # 2. RFM 热力图
    plt.figure(figsize=(12, 8))
    correlation_matrix = rfm_data[['Recency', 'Frequency', 'Monetary']].corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    sns.heatmap(correlation_matrix, annot=True, mask=mask, cmap='coolwarm', vmin=-1, vmax=1, 
                annot_kws={"size": 14}, square=True, linewidths=.5)
    
    plt.title('RFM 特征相关性热力图', fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rfm_correlation.png'), dpi=300)
    plt.close()

def plot_repurchase_rate(repurchase_data: pd.DataFrame, output_dir: str):
    """商品复购率可视化"""
    plt.figure(figsize=(14, 7))
    
    # 对复购率数据进行排序
    sorted_data = repurchase_data.sort_values('Customer_Name', ascending=False)
    
    # 创建水平条形图
    ax = sns.barplot(y=sorted_data.index, x='Customer_Name', data=sorted_data, palette='Blues_d')
    
    # 添加数据标签
    for i, v in enumerate(sorted_data['Customer_Name']):
        ax.text(v + 0.01, i, f'{v:.2f}', va='center', fontsize=10)
    
    plt.title('商品复购率排名', fontsize=15)
    plt.xlabel('复购率', fontsize=12)
    plt.ylabel('商品', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'repurchase_rate.png'), dpi=300)
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

def plot_association_rules_network(rules: pd.DataFrame, output_dir: str):
    """关联规则网络图可视化"""
    if rules.empty:
        return
    
    # 创建有向图
    G = nx.DiGraph()
    
    # 限制只显示前15条规则
    if len(rules) > 15:
        rules = rules.nlargest(15, 'lift')
    
    # 添加节点和边
    for idx, row in rules.iterrows():
        antecedents = list(row['antecedents'])
        consequents = list(row['consequents'])
        
        # 如果有多个前项，将它们连接起来
        if len(antecedents) > 1:
            ant_name = ' & '.join(antecedents)
        else:
            ant_name = antecedents[0]
            
        # 如果有多个后项，将它们连接起来
        if len(consequents) > 1:
            cons_name = ' & '.join(consequents)
        else:
            cons_name = consequents[0]
            
        # 添加节点
        G.add_node(ant_name)
        G.add_node(cons_name)
        
        # 添加边
        G.add_edge(ant_name, cons_name, weight=row['lift'], confidence=row['confidence'])
    
    # 创建图形
    plt.figure(figsize=(14, 10))
    
    # 设置节点位置
    pos = nx.spring_layout(G, seed=42)
    
    # 获取边的权重用于宽度
    edge_widths = [G[u][v]['weight'] * 0.5 for u, v in G.edges()]
    
    # 获取边的置信度用于颜色
    edge_colors = [G[u][v]['confidence'] for u, v in G.edges()]
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue', alpha=0.8)
    
    # 绘制边
    edges = nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, 
                                 edge_cmap=plt.cm.Blues, arrows=True, arrowsize=15, 
                                 connectionstyle='arc3,rad=0.1')
    
    # 绘制标签
    nx.draw_networkx_labels(G, pos, font_size=10, font_family=chinese_font if chinese_font else 'sans-serif')
    
    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=min(edge_colors), vmax=max(edge_colors)))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('置信度 (Confidence)')
    
    plt.title('商品关联规则网络图', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'association_rules_network.png'), dpi=300)
    plt.close()

def plot_product_sales_bar(data: pd.DataFrame, output_dir: str):
    """商品销售量柱状图"""
    try:
        # 按商品分组计算销售量
        product_sales = data.groupby('Product').size().reset_index(name='销售量')
        product_sales = product_sales.sort_values('销售量', ascending=False).head(10)
        
        plt.figure(figsize=(14, 8))
        
        # 创建柱状图
        ax = sns.barplot(x='Product', y='销售量', data=product_sales, palette='viridis')
        
        # 添加数据标签
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha='center', va='bottom', fontsize=10)
            
        plt.title('热门商品销售量排名（Top 10）', fontsize=15)
        plt.xlabel('商品名称', fontsize=12)
        plt.ylabel('销售量（件）', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, 'product_sales_bar.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"生成商品销售量柱状图时出错: {str(e)}")

def plot_customer_spending_bar(data: pd.DataFrame, output_dir: str):
    """客户消费排名柱状图"""
    try:
        # 按客户分组计算总消费金额
        customer_spending = data.groupby('Customer_Name')['Total_Cost'].sum().reset_index()
        customer_spending = customer_spending.sort_values('Total_Cost', ascending=False).head(10)
        
        plt.figure(figsize=(14, 8))
        
        # 创建柱状图
        ax = sns.barplot(x='Customer_Name', y='Total_Cost', data=customer_spending, palette='coolwarm')
        
        # 添加数据标签
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.1f}', 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha='center', va='bottom', fontsize=10)
            
        plt.title('顶级客户消费排名（Top 10）', fontsize=15)
        plt.xlabel('客户姓名', fontsize=12)
        plt.ylabel('消费金额', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, 'customer_spending_bar.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"生成客户消费排名柱状图时出错: {str(e)}")

def plot_seasonal_sales_pie(data: pd.DataFrame, season_col: str, sales_col: str, output_dir: str):
    """季节销售占比饼图"""
    try:
        seasonal_sales = data.groupby(season_col)[sales_col].sum()
        
        plt.figure(figsize=(10, 8))
        
        # 创建饼图
        explode = (0.1, 0, 0, 0)  # 突出第一季度
        colors = plt.cm.viridis(np.linspace(0, 1, len(seasonal_sales)))
        
        patches, texts, autotexts = plt.pie(
            seasonal_sales, 
            labels=seasonal_sales.index,
            explode=explode,
            autopct='%1.1f%%',
            shadow=True,
            startangle=90,
            colors=colors
        )
        
        # 设置字体大小
        for text in texts:
            text.set_fontsize(12)
        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_color('white')
        
        plt.axis('equal')  # 确保饼图是圆的
        plt.title('季节销售额占比', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'seasonal_sales_pie.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"生成季节销售占比饼图时出错: {str(e)}")

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

def plot_improved_product_category_pie(data: pd.DataFrame, output_dir: str):
    """改进的商品类别销售占比饼图"""
    try:
        # 使用更智能的分类方法
        data['Product_Category'] = data['Product'].apply(extract_product_category)
        
        # 按类别分组计算销售额
        category_sales = data.groupby('Product_Category')['Total_Cost'].sum()
        
        # 保留销售额占比大于1%的类别，其余归为"其他"
        total_sales = category_sales.sum()
        significant_categories = category_sales[category_sales / total_sales >= 0.01]
        other_sales = total_sales - significant_categories.sum()
        
        if other_sales > 0:
            significant_categories['其他'] = other_sales
        
        plt.figure(figsize=(14, 10))
        
        # 创建更丰富的颜色映射
        colors = plt.cm.tab20c(np.linspace(0, 1, len(significant_categories)))
        
        # 爆炸效果 - 突出主要类别
        explode = [0.05] * len(significant_categories)
        max_index = significant_categories.values.argmax()
        explode[max_index] = 0.2  # 突出销售额最高的类别
        
        # 创建饼图
        wedges, texts, autotexts = plt.pie(
            significant_categories, 
            labels=None,  # 不直接在饼图上显示标签，而是使用图例
            explode=explode,
            autopct='%1.1f%%',
            pctdistance=0.85,
            shadow=True,
            startangle=90,
            colors=colors,
            wedgeprops={'edgecolor': 'w', 'linewidth': 1.5}
        )
        
        # 设置百分比标签的样式
        for autotext in autotexts:
            autotext.set_fontsize(11)
            autotext.set_color('black')
            autotext.set_fontweight('bold')
        
        # 添加标题和中心圆
        centre_circle = plt.Circle((0, 0), 0.5, fc='white')
        plt.gca().add_artist(centre_circle)
        plt.title('商品类别销售占比分析', fontsize=20, pad=20, fontweight='bold')
        
        # 添加图例
        plt.legend(
            wedges,
            significant_categories.index,
            title="商品类别",
            loc="center left",
            bbox_to_anchor=(0.85, 0, 0.5, 1),
            fontsize=12
        )
        
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'improved_product_category_pie.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"生成改进的商品类别销售占比饼图时出错: {str(e)}")

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

def plot_product_category_pie(data: pd.DataFrame, output_dir: str):
    """商品类别销售占比饼图"""
    try:
        # 尝试从Product名称中提取类别，简单方法是使用第一个单词
        data['Product_Category'] = data['Product'].apply(lambda x: str(x).split()[0] if isinstance(x, str) and ' ' in x else x)
        
        # 按类别分组计算销售额
        category_sales = data.groupby('Product_Category')['Total_Cost'].sum()
        
        # 只保留前12个类别（增加显示类别数量），其他合并为"其他"
        if len(category_sales) > 12:
            top_categories = category_sales.nlargest(12)
            others_sum = category_sales.sum() - top_categories.sum()
            # 只有当"其他"占比超过3%时才显示
            if others_sum / category_sales.sum() > 0.03:
                top_categories['其他'] = others_sum
            category_sales = top_categories
        
        plt.figure(figsize=(12, 10))
        
        # 创建颜色映射
        colors = plt.cm.tab20(np.linspace(0, 1, len(category_sales)))
        
        # 创建饼图
        wedges, texts, autotexts = plt.pie(
            category_sales, 
            labels=category_sales.index,
            autopct='%1.1f%%',
            shadow=False,
            startangle=140,
            colors=colors,
            pctdistance=0.85
        )
        
        # 设置字体大小
        for text in texts:
            text.set_fontsize(12)
        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_color('black')
            
        # 添加圆环效果
        plt.annotate(
            '商品类别',
            xy=(0, 0),
            xytext=(0, 0),
            fontsize=16,
            ha='center',
            va='center'
        )
        
        # 添加图例，使图表更清晰
        plt.legend(
            wedges,
            category_sales.index,
            title="商品类别",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1)
        )
            
        plt.axis('equal')  # 确保饼图是圆的
        plt.title('商品类别销售占比', fontsize=18, pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'product_category_pie.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"生成商品类别销售占比饼图时出错: {str(e)}")

def plot_sales_by_category_horizontal(data: pd.DataFrame, output_dir: str):
    """销售类别水平条形图 - 更直观的销售类别可视化"""
    try:
        # 使用智能分类方法
        data['Product_Category'] = data['Product'].apply(extract_product_category)
        
        # 按类别分组计算销售额和销售量
        category_stats = data.groupby('Product_Category').agg({
            'Total_Cost': 'sum',  # 销售额
            'Product': 'count'  # 销售量
        }).rename(columns={'Product': 'Sales_Count'})
        
        # 计算每个类别的平均价格
        category_stats['Avg_Price'] = category_stats['Total_Cost'] / category_stats['Sales_Count']
        
        # 对数据进行排序并选择前10个类别
        top_categories = category_stats.sort_values('Total_Cost', ascending=False).head(10)
        
        # 创建一个水平条形图
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # 创建水平条形图
        bars = ax.barh(
            top_categories.index,
            top_categories['Total_Cost'],
            color=plt.cm.viridis(np.linspace(0, 0.8, len(top_categories)))
        )
        
        # 添加数据标签
        for i, bar in enumerate(bars):
            width = bar.get_width()
            items_count = top_categories['Sales_Count'].iloc[i]
            avg_price = top_categories['Avg_Price'].iloc[i]
            
            # 在条形上显示销售额
            ax.text(
                width + width*0.02,
                bar.get_y() + bar.get_height()/2,
                f'¥{width:.1f}',
                va='center',
                fontweight='bold'
            )
            
            # 在条形内部显示销售量和平均价格
            if width > top_categories['Total_Cost'].max() * 0.05:  # 只在较长的条形上显示更多信息
                ax.text(
                    width / 2,
                    bar.get_y() + bar.get_height()/2,
                    f'数量: {int(items_count)} | 均价: ¥{avg_price:.2f}',
                    va='center',
                    ha='center',
                    color='white',
                    fontweight='bold'
                )
        
        # 添加网格线
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # 设置标题和标签
        ax.set_title('销售类别分析 - 按销售额排序（Top 10）', fontsize=18, pad=20)
        ax.set_xlabel('销售额 (¥)', fontsize=14)
        ax.set_ylabel('商品类别', fontsize=14)
        
        # 将y轴标签设置得更醒目
        for tick in ax.get_yticklabels():
            tick.set_fontweight('bold')
            tick.set_fontsize(12)
        
        # 添加总计信息
        total_sales = top_categories['Total_Cost'].sum()
        total_items = top_categories['Sales_Count'].sum()
        
        ax.text(
            0.98, 0.02,
            f'总销售额: ¥{total_sales:.2f}\n总销售量: {int(total_items)}件',
            transform=ax.transAxes,
            ha='right',
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5')
        )
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sales_by_category_horizontal.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"生成销售类别水平条形图时出错: {str(e)}")

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
    """季节-类别热力图 - 展示不同季节各类别的销售情况"""
    try:
        # 确保数据包含季节和类别信息
        if 'Season' not in data.columns:
            print("数据中缺少Season列，无法生成季节-类别热力图")
            return
        
        # 使用智能分类方法
        data['Product_Category'] = data['Product'].apply(extract_product_category)
        
        # 计算每个季节每个类别的销售额
        seasonal_category = data.pivot_table(
            index='Season', 
            columns='Product_Category',
            values='Total_Cost',
            aggfunc='sum',
            fill_value=0
        )
        
        # 只保留销售额排名前10的类别
        top_categories = data.groupby('Product_Category')['Total_Cost'].sum().nlargest(10).index
        seasonal_category = seasonal_category[seasonal_category.columns.intersection(top_categories)]
        
        # 对数据进行归一化处理，使热力图更明显
        seasonal_category_norm = seasonal_category.div(seasonal_category.max(axis=1), axis=0)
        
        # 创建热力图
        plt.figure(figsize=(15, 8))
        ax = sns.heatmap(
            seasonal_category_norm,
            annot=seasonal_category.round(1),  # 显示实际销售额
            fmt='.1f',
            cmap='YlGnBu',
            linewidths=.5,
            cbar_kws={'label': '销售额占比（按季节归一化）'}
        )
        
        # 设置标题和标签
        plt.title('季节-商品类别销售热力图', fontsize=16, pad=20)
        plt.xlabel('商品类别', fontsize=12)
        plt.ylabel('季节', fontsize=12)
        
        # 调整标签大小
        plt.xticks(fontsize=10, rotation=45, ha='right')
        plt.yticks(fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'seasonal_category_heatmap.png'), dpi=300)
        plt.close()
        
        # 生成堆叠条形图，展示每个季节的类别组成
        plt.figure(figsize=(15, 10))
        
        seasonal_category.plot(
            kind='bar',
            stacked=True,
            figsize=(15, 10),
            colormap='tab20'
        )
        
        plt.title('各季节商品类别销售额构成', fontsize=16, pad=15)
        plt.xlabel('季节', fontsize=12)
        plt.ylabel('销售额', fontsize=12)
        plt.legend(title='商品类别', bbox_to_anchor=(1.01, 1), loc='upper left')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 添加总计数据标签
        for i, season in enumerate(seasonal_category.index):
            total = seasonal_category.loc[season].sum()
            plt.text(
                i,
                total + total*0.02,
                f'总计: ¥{total:.1f}',
                ha='center',
                fontweight='bold'
            )
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'seasonal_category_stacked.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"生成季节-类别热力图时出错: {str(e)}")