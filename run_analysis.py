#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
零售数据分析系统启动脚本
- 检查所需依赖
- 如果需要，自动安装缺失的依赖
- 运行分析程序
"""

import os
import sys
import subprocess
import importlib
import logging

# 添加颜色设置
class Colors:
    WHITE = '\033[37m'
    RESET = '\033[0m'

def print_white(text):
    """打印白色文本"""
    print(f"{Colors.WHITE}{text}{Colors.RESET}")

# 自定义日志格式化器
class WhiteColorFormatter(logging.Formatter):
    def format(self, record):
        return f"{Colors.WHITE}{super().format(record)}{Colors.RESET}"

# 配置日志
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = WhiteColorFormatter('%(asctime)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

def check_and_install_dependencies():
    """检查并安装缺少的依赖"""
    required_packages = [
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'mlxtend',
        'networkx',
        'sklearn',
        'wordcloud'
    ]
    
    missing_packages = []
    
    # 检查每个包是否已安装
    for package in required_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            missing_packages.append(package)
    
    # 如果有缺失的包，询问用户是否安装
    if missing_packages:
        logger.info(f"检测到缺少以下依赖包: {', '.join(missing_packages)}")
        
        try:
            choice = input(f"{Colors.WHITE}是否自动安装这些依赖? (y/n): {Colors.RESET}").lower().strip()
            if choice == 'y' or choice == 'yes':
                logger.info("开始安装缺失的依赖...")
                
                # 使用pip安装缺失的包
                subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
                
                logger.info("所有依赖安装完成")
                return True
            else:
                logger.warning("未安装依赖，程序可能无法正常运行")
                return False
        except Exception as e:
            logger.error(f"安装依赖时出错: {str(e)}")
            return False
    
    return True

def check_data_file():
    """检查数据文件是否存在"""
    data_file = os.path.join('data', 'Retail_Transactions_Dataset.csv')
    if not os.path.exists(data_file):
        logger.error(f"数据文件不存在: {data_file}")
        logger.info("请确保数据文件位于data目录下，并命名为'Retail_Transactions_Dataset.csv'")
        logger.info("您可以从Kaggle下载数据集: https://www.kaggle.com/datasets/prasad22/retail-transactions-dataset/data")
        return False
    return True

def run_analysis():
    """运行分析程序"""
    try:
        logger.info("启动零售数据分析...")
        
        # 确保日志目录存在
        os.makedirs('logs', exist_ok=True)
        
        # 添加当前目录到Python路径，确保能正确导入模块
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # 运行主程序
        from src.main import main
        main()
        
        logger.info("分析完成!")
        
        # 获取图表目录的绝对路径
        figures_dir = os.path.abspath(os.path.join('output', 'figures'))
        logger.info(f"分析结果保存在 output 目录")
        logger.info(f"生成的图表保存在 {figures_dir}")
        
        return True
    except Exception as e:
        logger.error(f"运行分析时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    # 移除之前的所有处理器
    logger.handlers.clear()
    # 添加新的处理器
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    print_white("=" * 60)
    print_white("          零售数据分析系统")
    print_white("=" * 60)
    
    # 检查并安装依赖
    if check_and_install_dependencies():
        # 检查数据文件
        if check_data_file():
            # 运行分析
            if run_analysis():
                print_white("\n分析成功完成! 您可以在output目录查看分析结果。")
            else:
                print_white("\n分析过程中出错。请查看日志获取详细信息。")
        else:
            print_white("\n数据文件检查失败。请确保数据文件正确放置。")
    else:
        print_white("\n依赖安装失败。请手动安装所需依赖后再运行程序。")
    
    print_white("\n按回车键退出...")
    input() 