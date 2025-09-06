#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QCA分析模块
使用rpy2连接R的QCA包进行模糊集定性比较分析(fsQCA)
"""

import pandas as pd
import numpy as np
import streamlit as st
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

try:
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects import Formula
    from rpy2.robjects.vectors import StrVector
    
    # 启用pandas转换
    pandas2ri.activate()
    
    # 导入R包
    qca = importr('QCA')
    base = importr('base')
    stats = importr('stats')
    
    R_AVAILABLE = True
    print("✓ QCA模块初始化成功")
    
except ImportError as e:
    R_AVAILABLE = False
    print(f"❌ QCA模块初始化失败: {e}")
    print("请确保已安装rpy2包和R的QCA包")


def check_r_connection():
    """检查R连接状态"""
    if not R_AVAILABLE:
        return False, "R连接不可用"
    
    try:
        # 测试R版本
        r_version = robjects.r('R.version.string')[0]
        return True, f"R连接正常，版本: {r_version}"
    except Exception as e:
        return False, f"R连接测试失败: {str(e)}"


def perform_necessity_analysis(data, condition_vars, outcome_var, threshold=0.9):
    """
    执行必要性分析
    
    参数:
    - data: 包含数据的DataFrame
    - condition_vars: 条件变量列表
    - outcome_var: 结果变量名
    - threshold: 必要性阈值
    
    返回:
    - necessity_results: 必要性分析结果DataFrame
    """
    if not R_AVAILABLE:
        st.error("R连接不可用，无法执行QCA分析")
        return pd.DataFrame()
    
    try:
        # 准备数据
        analysis_data = data[condition_vars + [outcome_var]].copy()
        
        # 转换为R数据框
        r_data = pandas2ri.py2rpy(analysis_data)
        
        # 构建R代码进行必要性分析
        r_code = f"""
        # 设置数据
        data <- {r_data.r_repr()}
        
        # 执行必要性分析
        necessity_result <- pof(data${outcome_var}, data[,c({', '.join([f'"{var}"' for var in condition_vars])})], 
                               relation = "necessity", threshold = {threshold})
        
        # 提取结果
        result_df <- data.frame(
            condition = names(necessity_result$inclN),
            inclusion = as.numeric(necessity_result$inclN),
            coverage = as.numeric(necessity_result$covN),
            relevance = as.numeric(necessity_result$relN)
        )
        
        result_df
        """
        
        # 执行R代码
        result = robjects.r(r_code)
        
        # 转换为pandas DataFrame
        necessity_results = pandas2ri.rpy2py(result)
        
        # 添加解释
        necessity_results['必要性解释'] = necessity_results.apply(
            lambda row: '强必要性' if row['inclusion'] >= threshold else '弱必要性', axis=1
        )
        
        return necessity_results
        
    except Exception as e:
        st.error(f"必要性分析执行失败: {str(e)}")
        return pd.DataFrame()


def perform_sufficiency_analysis(data, condition_vars, outcome_var, 
                                freq_threshold=1.0, pri_consistency=0.7, consistency=0.8):
    """
    执行充分性分析
    
    参数:
    - data: 包含数据的DataFrame
    - condition_vars: 条件变量列表
    - outcome_var: 结果变量名
    - freq_threshold: 频率阈值
    - pri_consistency: 原始一致性阈值
    - consistency: 一致性阈值
    
    返回:
    - sufficiency_results: 充分性分析结果DataFrame
    """
    if not R_AVAILABLE:
        st.error("R连接不可用，无法执行QCA分析")
        return pd.DataFrame()
    
    try:
        # 准备数据
        analysis_data = data[condition_vars + [outcome_var]].copy()
        
        # 转换为R数据框
        r_data = pandas2ri.py2rpy(analysis_data)
        
        # 构建R代码进行充分性分析
        r_code = f"""
        # 设置数据
        data <- {r_data.r_repr()}
        
        # 执行充分性分析
        sufficiency_result <- pof(data${outcome_var}, data[,c({', '.join([f'"{var}"' for var in condition_vars])})], 
                                 relation = "sufficiency", threshold = {consistency})
        
        # 提取结果
        result_df <- data.frame(
            condition = names(sufficiency_result$inclS),
            inclusion = as.numeric(sufficiency_result$inclS),
            coverage = as.numeric(sufficiency_result$covS),
            relevance = as.numeric(sufficiency_result$relS)
        )
        
        result_df
        """
        
        # 执行R代码
        result = robjects.r(r_code)
        
        # 转换为pandas DataFrame
        sufficiency_results = pandas2ri.rpy2py(result)
        
        # 添加解释
        sufficiency_results['充分性解释'] = sufficiency_results.apply(
            lambda row: '强充分性' if row['inclusion'] >= consistency else '弱充分性', axis=1
        )
        
        return sufficiency_results
        
    except Exception as e:
        st.error(f"充分性分析执行失败: {str(e)}")
        return pd.DataFrame()


def perform_truth_table_analysis(data, condition_vars, outcome_var, 
                                freq_threshold=1.0, consistency=0.8):
    """
    执行真值表分析
    
    参数:
    - data: 包含数据的DataFrame
    - condition_vars: 条件变量列表
    - outcome_var: 结果变量名
    - freq_threshold: 频率阈值
    - consistency: 一致性阈值
    
    返回:
    - truth_table: 真值表DataFrame
    """
    if not R_AVAILABLE:
        st.error("R连接不可用，无法执行QCA分析")
        return pd.DataFrame()
    
    try:
        # 准备数据
        analysis_data = data[condition_vars + [outcome_var]].copy()
        
        # 转换为R数据框
        r_data = pandas2ri.py2rpy(analysis_data)
        
        # 构建R代码进行真值表分析
        r_code = f"""
        # 设置数据
        data <- {r_data.r_repr()}
        
        # 执行真值表分析
        truth_table <- truthTable(data, outcome = "{outcome_var}", 
                                 conditions = c({', '.join([f'"{var}"' for var in condition_vars])}),
                                 incl.cut = {consistency}, n.cut = {freq_threshold})
        
        # 提取真值表
        tt_df <- data.frame(truth_table$tt)
        tt_df$OUT <- as.numeric(tt_df$OUT)
        tt_df$n <- as.numeric(tt_df$n)
        tt_df$incl <- as.numeric(tt_df$incl)
        tt_df$PRI <- as.numeric(tt_df$PRI)
        
        tt_df
        """
        
        # 执行R代码
        result = robjects.r(r_code)
        
        # 转换为pandas DataFrame
        truth_table = pandas2ri.rpy2py(result)
        
        # 添加解释
        truth_table['结果解释'] = truth_table.apply(
            lambda row: '高绩效' if row['OUT'] == 1 else '低绩效', axis=1
        )
        
        return truth_table
        
    except Exception as e:
        st.error(f"真值表分析执行失败: {str(e)}")
        return pd.DataFrame()


def perform_minimization(data, condition_vars, outcome_var, 
                        freq_threshold=1.0, consistency=0.8, 
                        include="", exclude=""):
    """
    执行最小化算法
    
    参数:
    - data: 包含数据的DataFrame
    - condition_vars: 条件变量列表
    - outcome_var: 结果变量名
    - freq_threshold: 频率阈值
    - consistency: 一致性阈值
    - include: 必须包含的项
    - exclude: 必须排除的项
    
    返回:
    - minimization_results: 最小化结果DataFrame
    """
    if not R_AVAILABLE:
        st.error("R连接不可用，无法执行QCA分析")
        return pd.DataFrame()
    
    try:
        # 准备数据
        analysis_data = data[condition_vars + [outcome_var]].copy()
        
        # 转换为R数据框
        r_data = pandas2ri.py2rpy(analysis_data)
        
        # 构建R代码进行最小化分析
        r_code = f"""
        # 设置数据
        data <- {r_data.r_repr()}
        
        # 执行最小化分析
        minimization_result <- minimize(data, outcome = "{outcome_var}", 
                                       conditions = c({', '.join([f'"{var}"' for var in condition_vars])}),
                                       incl.cut = {consistency}, n.cut = {freq_threshold},
                                       include = "{include}", exclude = "{exclude}")
        
        # 提取结果
        if (length(minimization_result$solution) > 0) {{
            solution_df <- data.frame(
                solution = names(minimization_result$solution),
                consistency = as.numeric(minimization_result$solution),
                coverage = as.numeric(minimization_result$coverage),
                unique_coverage = as.numeric(minimization_result$unique_coverage)
            )
        }} else {{
            solution_df <- data.frame(
                solution = character(0),
                consistency = numeric(0),
                coverage = numeric(0),
                unique_coverage = numeric(0)
            )
        }}
        
        solution_df
        """
        
        # 执行R代码
        result = robjects.r(r_code)
        
        # 转换为pandas DataFrame
        minimization_results = pandas2ri.rpy2py(result)
        
        if not minimization_results.empty:
            # 添加解释
            minimization_results['路径解释'] = minimization_results.apply(
                lambda row: f"组态路径: {row['solution']}", axis=1
            )
        
        return minimization_results
        
    except Exception as e:
        st.error(f"最小化分析执行失败: {str(e)}")
        return pd.DataFrame()


def perform_complete_qca_analysis(data, condition_vars, outcome_var, 
                                 freq_threshold=1.0, consistency=0.8):
    """
    执行完整的QCA分析流程
    
    参数:
    - data: 包含数据的DataFrame
    - condition_vars: 条件变量列表
    - outcome_var: 结果变量名
    - freq_threshold: 频率阈值
    - consistency: 一致性阈值
    
    返回:
    - results: 包含所有分析结果的字典
    """
    results = {}
    
    # 1. 必要性分析
    st.info("正在执行必要性分析...")
    necessity_results = perform_necessity_analysis(data, condition_vars, outcome_var, consistency)
    results['necessity'] = necessity_results
    
    # 2. 充分性分析
    st.info("正在执行充分性分析...")
    sufficiency_results = perform_sufficiency_analysis(data, condition_vars, outcome_var, 
                                                      freq_threshold, consistency, consistency)
    results['sufficiency'] = sufficiency_results
    
    # 3. 真值表分析
    st.info("正在执行真值表分析...")
    truth_table = perform_truth_table_analysis(data, condition_vars, outcome_var, 
                                              freq_threshold, consistency)
    results['truth_table'] = truth_table
    
    # 4. 最小化分析
    st.info("正在执行最小化分析...")
    minimization_results = perform_minimization(data, condition_vars, outcome_var, 
                                               freq_threshold, consistency)
    results['minimization'] = minimization_results
    
    return results


# 测试函数
def test_qca_module():
    """测试QCA模块功能"""
    print("开始测试QCA模块...")
    
    # 检查R连接
    is_connected, message = check_r_connection()
    print(f"R连接状态: {message}")
    
    if not is_connected:
        return False
    
    # 创建测试数据
    test_data = pd.DataFrame({
        'A': [0.8, 0.6, 0.9, 0.3, 0.7],
        'B': [0.7, 0.8, 0.4, 0.9, 0.6],
        'C': [0.6, 0.7, 0.8, 0.5, 0.9],
        'Y': [0.8, 0.7, 0.9, 0.4, 0.8]
    })
    
    print("测试数据创建成功")
    
    # 测试必要性分析
    try:
        necessity = perform_necessity_analysis(test_data, ['A', 'B', 'C'], 'Y')
        print(f"必要性分析测试成功，结果行数: {len(necessity)}")
    except Exception as e:
        print(f"必要性分析测试失败: {e}")
    
    return True


if __name__ == "__main__":
    test_qca_module()
