# 零售数据分析系统

这是一个基于Python的零售数据分析系统,用于分析零售交易数据,生成各类分析报告和可视化图表。


## 数据来源

本项目使用的数据集来自 Kaggle:
[Retail Transactions Dataset](https://www.kaggle.com/datasets/prasad22/retail-transactions-dataset/data)


## 功能特点

- 基础数据分析

  - 交易基本信息统计
  - 商品销售分析
  - 客户购买行为分析
- 高级分析

  - RFM客户分群分析
  - 购物篮分析
  - 商品关联规则挖掘
- 可视化分析

  - 商品价格与销量关系散点图
  - 商品价格分布直方图
  - 季节销售趋势图
  - 热门商品词云图
  - 购物篮分析热力图
  - 月度销售趋势图
  - 商品类别销售分析
  - 交易金额散点图
  - 客户消费词云图
  - 商品销售排名图
  - 季节-商品类别热力图
  - 24小时销售分析图

## 项目结构

```
RetailAnalysis/
├── data/                   # 数据文件目录
│   └── Retail_Transactions_Dataset.csv # 零售交易数据集
├── src/                    # 源代码目录
│   ├── analysis/          # 分析模块
│   │   ├── retail_analyzer.py     # 负责加载数据、基础清洗和初步分析
│   │   ├── product_analysis.py    # 负责商品层面的分析，如销售额、价格、排名等
│   │   ├── transaction_analysis.py# 负责交易层面的分析，如交易模式、时间序列等
│   │   ├── pattern_analysis.py    # 负责模式挖掘，如RFM分析、购物篮分析、关联规则
│   │   └── visualization.py       # 负责生成各种可视化图表
│   ├── utils/             # 工具模块
│   │   ├── data_preprocessing.py  # 包含数据预处理相关的辅助函数
│   │   ├── file_utils.py          # 包含文件操作（如保存结果）的辅助函数
│   │   └── logging_utils.py       # 包含日志记录配置的辅助函数
│   └── main.py            # 主分析流程控制脚本，调用各分析和工具模块
├── output/                 # 输出目录
│   ├── figures/           # 图表输出目录
│   └── *.txt              # 文本格式的分析报告
├── logs/                   # 日志目录
│   └── analysis.log       # 分析过程的日志文件
├── requirements.txt        # 项目依赖的Python包列表
└── run_analysis.py        # 项目启动脚本，检查依赖并运行主分析程序
```



## 环境要求

- Python 3.12
- 依赖包:
  - pandas >= 1.3.0
  - numpy >= 1.20.0
  - matplotlib >= 3.4.0
  - seaborn >= 0.11.0
  - mlxtend >= 0.19.0
  - networkx >= 2.6.0
  - scikit-learn >= 0.24.0
  - wordcloud >= 1.8.0



## 安装使用

1. 克隆项目到本地:

```bash
git clone https://github.com/yourusername/RetailAnalysis.git
cd RetailAnalysis
```

2. 安装依赖:

```bash
pip install -r requirements.txt
```

3. 准备数据:

- 将零售交易数据集文件 `Retail_Transactions_Dataset.csv` 放入 `data` 目录

4. 运行分析:

```bash
python run_analysis.py
```

5. 查看结果:

- 分析报告保存在 `output` 目录
- 可视化图表保存在 `output/figures` 目录
- 运行日志保存在 `logs` 目录



## 数据要求

输入数据集需要包含以下字段:

- Transaction_ID: 交易ID
- Date: 交易日期
- Product: 商品列表
- Total_Items: 商品总数
- Total_Cost: 交易总金额



## 主要功能模块

### retail_analyzer.py

- 基础数据分析
- 数据预处理
- 数据统计

### product_analysis.py

- 商品销售分析
- 价格分析
- 商品排名

### transaction_analysis.py

- 交易模式分析
- 时间序列分析

### pattern_analysis.py

- RFM分析
- 购物篮分析
- 关联规则挖掘

### visualization.py

- 数据可视化
- 图表生成



## 注意事项

1. 数据文件大小限制:

   - 建议数据量不超过100万条记录
   - 默认随机抽样10000条记录进行分析
2. 内存要求:

   - 建议可用内存不少于8GB
   - 大数据量分析时需要更多内存
3. 运行时间:

   - 完整分析过程约需5-10分钟
   - 可视化渲染约需2-3分钟



## 维护者

[@https://github.com/aoxing5](https://github.com/yourusername)



## 许可证

[MIT](LICENSE) © aoxing5
