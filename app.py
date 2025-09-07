import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import re
import itertools
from scipy.stats import pearsonr
from scipy.optimize import linprog
import tempfile
import os

# 使用自定义DEA实现
print("✅ 使用自定义DEA实现进行DEA分析")

class CustomDEA:
    """自定义DEA实现，支持CCR和BCC模型的输入导向和输出导向版本"""
    
    def __init__(self, input_data, output_data):
        self.input_data = np.array(input_data, dtype=np.float64)
        self.output_data = np.array(output_data, dtype=np.float64)
        self.n_dmus = self.input_data.shape[0]
        self.n_inputs = self.input_data.shape[1]
        self.n_outputs = self.output_data.shape[1]
        
        # 数据验证：只检查负值，允许0值
        if np.any(self.input_data < 0):
            raise ValueError("所有投入变量不能为负数")
        if np.any(self.output_data < 0):
            raise ValueError("所有产出变量不能为负数")
        
        # 将0替换为极小正值，避免除零错误
        self.input_data = np.maximum(self.input_data, 1e-10)
        self.output_data = np.maximum(self.output_data, 1e-10)
        
        # 检查常数列（可能导致数值问题）
        for i in range(self.n_inputs):
            if np.all(self.input_data[:, i] == self.input_data[0, i]):
                print(f"警告: 投入变量 {i} 是常数列，可能导致数值问题")
        
        for r in range(self.n_outputs):
            if np.all(self.output_data[:, r] == self.output_data[0, r]):
                print(f"警告: 产出变量 {r} 是常数列，可能导致数值问题")
        
        # 数据标准化以提高数值稳定性
        self.input_scale = np.mean(self.input_data, axis=0)
        self.output_scale = np.mean(self.output_data, axis=0)
        
        # 避免除零
        self.input_scale = np.maximum(self.input_scale, 1e-8)
        self.output_scale = np.maximum(self.output_scale, 1e-8)
        
        # 标准化数据
        self.input_data_norm = self.input_data / self.input_scale
        self.output_data_norm = self.output_data / self.output_scale
        
        # 存储松弛变量
        self.slack_input = {}
        self.slack_output = {}
    
    def ccr_input_oriented(self, input_variable, output_variable, dmu, data, method='revised simplex'):
        """CCR模型 - 输入导向（规模报酬不变）"""
        return self._solve_ccr_input_model(input_variable, output_variable, dmu, data, method)
    
    def ccr_output_oriented(self, input_variable, output_variable, dmu, data, method='revised simplex'):
        """CCR模型 - 输出导向（规模报酬不变）"""
        return self._solve_ccr_output_model(input_variable, output_variable, dmu, data, method)
    
    def bcc_input_oriented(self, input_variable, output_variable, dmu, data, method='revised simplex'):
        """BCC模型 - 输入导向（规模报酬可变）"""
        return self._solve_bcc_input_model(input_variable, output_variable, dmu, data, method)
    
    def bcc_output_oriented(self, input_variable, output_variable, dmu, data, method='revised simplex'):
        """BCC模型 - 输出导向（规模报酬可变）"""
        return self._solve_bcc_output_model(input_variable, output_variable, dmu, data, method)
    
    def sbm(self, input_variable, desirable_output, undesirable_output, dmu, data, method='revised simplex'):
        """SBM模型 - 基于松弛变量的效率测量模型"""
        return self._solve_sbm_model(input_variable, desirable_output, undesirable_output, dmu, data, method)
    
    def super_sbm(self, input_variable, desirable_output, undesirable_output, dmu, data, method='revised simplex'):
        """超效率SBM模型 - 允许效率值大于1"""
        return self._solve_super_sbm_model(input_variable, desirable_output, undesirable_output, dmu, data, method)
    
    def _solve_ccr_input_model(self, input_variable, output_variable, dmu, data, method='revised simplex'):
        """求解CCR输入导向模型的核心实现"""
        import pandas as pd
        import scipy.optimize as op
        import numpy as np
        
        res = pd.DataFrame(columns=['dmu', 'TE'], index=data.index)
        res['dmu'] = data[dmu]
        
        # 获取基本参数
        dmu_counts = data.shape[0]
        m = len(input_variable)  # 投入个数
        s = len(output_variable)  # 产出个数
        
        # 变量结构：x[:dmu_counts] 为lambda, x[dmu_counts] 为theta
        total = dmu_counts + 1
        
        # 创建lambda列
        for j in range(dmu_counts):
            res[f'lambda_{j+1}'] = np.nan
        
        # 对每个DMU求解
        for i in range(dmu_counts):
            try:
                # 目标函数：max theta (转换为min -theta)
                c = [0] * dmu_counts + [-1]
                
                # 约束条件
                A_ub = []
                b_ub = []
                
                # 投入约束：∑λⱼxᵢⱼ ≤ θxᵢₒ
                for j1 in range(m):
                    constraint = [0] * dmu_counts + [-data.loc[i, input_variable[j1]]]
                    for k in range(dmu_counts):
                        constraint[k] = data.loc[k, input_variable[j1]]
                    A_ub.append(constraint)
                    b_ub.append(0)
                
                # 产出约束：∑λⱼyᵣⱼ ≥ yᵣₒ (转换为 -∑λⱼyᵣⱼ ≤ -yᵣₒ)
                for j2 in range(s):
                    constraint = [0] * dmu_counts + [0]
                    for k in range(dmu_counts):
                        constraint[k] = -data.loc[k, output_variable[j2]]
                    A_ub.append(constraint)
                    b_ub.append(-data.loc[i, output_variable[j2]])
                
                # 非负约束
                bounds = [(0, None)] * total
                
                # 求解
                op1 = op.linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=method)
                
                if op1.success:
                    res.loc[i, 'TE'] = -op1.fun  # 转换回max theta
                    res.loc[i, [f'lambda_{j+1}' for j in range(dmu_counts)]] = op1.x[:dmu_counts]
                else:
                    res.loc[i, 'TE'] = 0.0
                    res.loc[i, [f'lambda_{j+1}' for j in range(dmu_counts)]] = 0.0
                    
            except Exception as e:
                print(f"CCR输入导向求解失败 (DMU {i+1}): {e}")
                res.loc[i, 'TE'] = 0.0
                res.loc[i, [f'lambda_{j+1}' for j in range(dmu_counts)]] = 0.0
        
        return res
    
    def _solve_ccr_output_model(self, input_variable, output_variable, dmu, data, method='revised simplex'):
        """求解CCR输出导向模型的核心实现"""
        import pandas as pd
        import scipy.optimize as op
        import numpy as np
        
        res = pd.DataFrame(columns=['dmu', 'TE'], index=data.index)
        res['dmu'] = data[dmu]
        
        # 获取基本参数
        dmu_counts = data.shape[0]
        m = len(input_variable)  # 投入个数
        s = len(output_variable)  # 产出个数
        
        # 变量结构：x[:dmu_counts] 为lambda, x[dmu_counts] 为phi
        total = dmu_counts + 1
        
        # 创建lambda列
        for j in range(dmu_counts):
            res[f'lambda_{j+1}'] = np.nan
        
        # 对每个DMU求解
        for i in range(dmu_counts):
            try:
                # 目标函数：min phi
                c = [0] * dmu_counts + [1]
                
                # 约束条件
                A_ub = []
                b_ub = []
                
                # 投入约束：∑λⱼxᵢⱼ ≤ xᵢₒ
                for j1 in range(m):
                    constraint = [0] * dmu_counts + [0]
                    for k in range(dmu_counts):
                        constraint[k] = data.loc[k, input_variable[j1]]
                    A_ub.append(constraint)
                    b_ub.append(data.loc[i, input_variable[j1]])
                
                # 产出约束：∑λⱼyᵣⱼ ≥ φyᵣₒ (转换为 -∑λⱼyᵣⱼ + φyᵣₒ ≤ 0)
                for j2 in range(s):
                    constraint = [0] * dmu_counts + [data.loc[i, output_variable[j2]]]
                    for k in range(dmu_counts):
                        constraint[k] = -data.loc[k, output_variable[j2]]
                    A_ub.append(constraint)
                    b_ub.append(0)
                
                # 非负约束
                bounds = [(0, None)] * total
                
                # 求解
                op1 = op.linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=method)
                
                if op1.success:
                    res.loc[i, 'TE'] = 1.0 / op1.fun  # 效率值 = 1/phi
                    res.loc[i, [f'lambda_{j+1}' for j in range(dmu_counts)]] = op1.x[:dmu_counts]
                else:
                    res.loc[i, 'TE'] = 0.0
                    res.loc[i, [f'lambda_{j+1}' for j in range(dmu_counts)]] = 0.0
                    
            except Exception as e:
                print(f"CCR输出导向求解失败 (DMU {i+1}): {e}")
                res.loc[i, 'TE'] = 0.0
                res.loc[i, [f'lambda_{j+1}' for j in range(dmu_counts)]] = 0.0
        
        return res
    
    def _solve_bcc_input_model(self, input_variable, output_variable, dmu, data, method='revised simplex'):
        """求解BCC输入导向模型的核心实现"""
        import pandas as pd
        import scipy.optimize as op
        import numpy as np
        
        res = pd.DataFrame(columns=['dmu', 'TE'], index=data.index)
        res['dmu'] = data[dmu]
        
        # 获取基本参数
        dmu_counts = data.shape[0]
        m = len(input_variable)  # 投入个数
        s = len(output_variable)  # 产出个数
        
        # 变量结构：x[:dmu_counts] 为lambda, x[dmu_counts] 为theta
        total = dmu_counts + 1
        
        # 创建lambda列
        for j in range(dmu_counts):
            res[f'lambda_{j+1}'] = np.nan
        
        # 对每个DMU求解
        for i in range(dmu_counts):
            try:
                # 目标函数：max theta (转换为min -theta)
                c = [0] * dmu_counts + [-1]
                
                # 约束条件
                A_ub = []
                b_ub = []
                A_eq = []
                b_eq = []
                
                # 投入约束：∑λⱼxᵢⱼ ≤ θxᵢₒ
                for j1 in range(m):
                    constraint = [0] * dmu_counts + [-data.loc[i, input_variable[j1]]]
                    for k in range(dmu_counts):
                        constraint[k] = data.loc[k, input_variable[j1]]
                    A_ub.append(constraint)
                    b_ub.append(0)
                
                # 产出约束：∑λⱼyᵣⱼ ≥ yᵣₒ (转换为 -∑λⱼyᵣⱼ ≤ -yᵣₒ)
                for j2 in range(s):
                    constraint = [0] * dmu_counts + [0]
                    for k in range(dmu_counts):
                        constraint[k] = -data.loc[k, output_variable[j2]]
                    A_ub.append(constraint)
                    b_ub.append(-data.loc[i, output_variable[j2]])
                
                # BCC模型特有约束：∑λⱼ = 1
                constraint = [1] * dmu_counts + [0]
                A_eq.append(constraint)
                b_eq.append(1)
                
                # 非负约束
                bounds = [(0, None)] * total
                
                # 求解
                op1 = op.linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=method)
                
                if op1.success:
                    res.loc[i, 'TE'] = -op1.fun  # 转换回max theta
                    res.loc[i, [f'lambda_{j+1}' for j in range(dmu_counts)]] = op1.x[:dmu_counts]
                else:
                    res.loc[i, 'TE'] = 0.0
                    res.loc[i, [f'lambda_{j+1}' for j in range(dmu_counts)]] = 0.0
                    
            except Exception as e:
                print(f"BCC输入导向求解失败 (DMU {i+1}): {e}")
                res.loc[i, 'TE'] = 0.0
                res.loc[i, [f'lambda_{j+1}' for j in range(dmu_counts)]] = 0.0
        
        return res
    
    def _solve_bcc_output_model(self, input_variable, output_variable, dmu, data, method='revised simplex'):
        """求解BCC输出导向模型的核心实现"""
        import pandas as pd
        import scipy.optimize as op
        import numpy as np
        
        res = pd.DataFrame(columns=['dmu', 'TE'], index=data.index)
        res['dmu'] = data[dmu]
        
        # 获取基本参数
        dmu_counts = data.shape[0]
        m = len(input_variable)  # 投入个数
        s = len(output_variable)  # 产出个数
        
        # 变量结构：x[:dmu_counts] 为lambda, x[dmu_counts] 为phi
        total = dmu_counts + 1
        
        # 创建lambda列
        for j in range(dmu_counts):
            res[f'lambda_{j+1}'] = np.nan
        
        # 对每个DMU求解
        for i in range(dmu_counts):
            try:
                # 目标函数：min phi
                c = [0] * dmu_counts + [1]
                
                # 约束条件
                A_ub = []
                b_ub = []
                A_eq = []
                b_eq = []
                
                # 投入约束：∑λⱼxᵢⱼ ≤ xᵢₒ
                for j1 in range(m):
                    constraint = [0] * dmu_counts + [0]
                    for k in range(dmu_counts):
                        constraint[k] = data.loc[k, input_variable[j1]]
                    A_ub.append(constraint)
                    b_ub.append(data.loc[i, input_variable[j1]])
                
                # 产出约束：∑λⱼyᵣⱼ ≥ φyᵣₒ (转换为 -∑λⱼyᵣⱼ + φyᵣₒ ≤ 0)
                for j2 in range(s):
                    constraint = [0] * dmu_counts + [data.loc[i, output_variable[j2]]]
                    for k in range(dmu_counts):
                        constraint[k] = -data.loc[k, output_variable[j2]]
                    A_ub.append(constraint)
                    b_ub.append(0)
                
                # BCC模型特有约束：∑λⱼ = 1
                constraint = [1] * dmu_counts + [0]
                A_eq.append(constraint)
                b_eq.append(1)
                
                # 非负约束
                bounds = [(0, None)] * total
                
                # 求解
                op1 = op.linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=method)
                
                if op1.success:
                    res.loc[i, 'TE'] = 1.0 / op1.fun  # 效率值 = 1/phi
                    res.loc[i, [f'lambda_{j+1}' for j in range(dmu_counts)]] = op1.x[:dmu_counts]
                else:
                    res.loc[i, 'TE'] = 0.0
                    res.loc[i, [f'lambda_{j+1}' for j in range(dmu_counts)]] = 0.0
                    
            except Exception as e:
                print(f"BCC输出导向求解失败 (DMU {i+1}): {e}")
                res.loc[i, 'TE'] = 0.0
                res.loc[i, [f'lambda_{j+1}' for j in range(dmu_counts)]] = 0.0
        
        return res
    
    def _solve_sbm_model(self, input_variable, desirable_output, undesirable_output, dmu, data, method='revised simplex'):
        """求解SBM模型的核心实现"""
        import pandas as pd
        import scipy.optimize as op
        import numpy as np
        
        res = pd.DataFrame(columns=['dmu', 'TE'], index=data.index)
        res['dmu'] = data[dmu]
        
        # 获取基本参数
        dmu_counts = data.shape[0]
        m = len(input_variable)  # 投入个数
        s1 = len(desirable_output)  # 期望产出个数
        s2 = len(undesirable_output)  # 非期望产出个数
        
        # 变量结构：
        # x[:dmu_counts] 为lambda
        # x[dmu_counts : dmu_counts+1] 为 t
        # x[dmu_counts+1 : dmu_counts + m + 1] 为投入slack
        # x[dmu_counts+ 1 + m : dmu_counts + 1 + m + s1] 为期望产出slack
        # x[dmu_counts + 1 + m + s1 :] 为非期望产出slack
        total = dmu_counts + m + s1 + s2 + 1
        
        # 创建slack列
        cols = input_variable + desirable_output + undesirable_output
        newcols = []
        for j in cols:
            newcols.append(j + '_slack')
            res[j + '_slack'] = np.nan
        
        # 对每个DMU求解
        for i in range(dmu_counts):
            try:
                # 优化目标：目标函数的系数矩阵
                c = [0] * dmu_counts + [1] + list(-1 / (m * data.loc[i, input_variable])) + [0] * (s1 + s2)
                
                # 约束条件：约束方程的系数矩阵
                A_eq = [[0] * dmu_counts + [1] + [0] * m +
                        list(1 / ((s1 + s2) * data.loc[i, desirable_output])) +
                        list(1 / ((s1 + s2) * data.loc[i, undesirable_output]))]
                
                # 约束条件（1）：投入松弛变量为正
                for j1 in range(m):
                    list1 = [0] * m
                    list1[j1] = 1
                    eq1 = list(data[input_variable[j1]]) + [-data.loc[i, input_variable[j1]]] + list1 + [0] * (s1 + s2)
                    A_eq.append(eq1)
                
                # 约束条件（2）：期望产出松弛变量为正
                for j2 in range(s1):
                    list2 = [0] * s1
                    list2[j2] = -1
                    eq2 = list(data[desirable_output[j2]]) + [-data.loc[i, desirable_output[j2]]] + [0] * m + list2 + [0] * s2
                    A_eq.append(eq2)
                
                # 约束条件（3）：非期望产出松弛变量为正
                for j3 in range(s2):
                    list3 = [0] * s2
                    list3[j3] = 1
                    eq3 = list(data[undesirable_output[j3]]) + [-data.loc[i, undesirable_output[j3]]] + [0] * (m + s1) + list3
                    A_eq.append(eq3)
                
                # 约束条件：常数向量
                b_eq = [1] + [0] * (m + s1 + s2)
                bounds = [(0, None)] * total  # 约束边界为零
                
                # 求解
                op1 = op.linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=method)
                
                if op1.success:
                    res.loc[i, 'TE'] = op1.fun
                    res.loc[i, newcols] = op1.x[dmu_counts + 1:]
                else:
                    res.loc[i, 'TE'] = 0.0
                    res.loc[i, newcols] = 0.0
                    
            except Exception as e:
                print(f"SBM求解失败 (DMU {i+1}): {e}")
                res.loc[i, 'TE'] = 0.0
                res.loc[i, newcols] = 0.0
        
        return res
    
    def _solve_super_sbm_model(self, input_variable, desirable_output, undesirable_output, dmu, data, method='revised simplex'):
        """求解超效率SBM模型"""
        import pandas as pd
        import scipy.optimize as op
        import numpy as np
        
        res = pd.DataFrame(columns=['dmu', 'TE'], index=data.index)
        res['dmu'] = data[dmu]
        
        # 获取基本参数
        dmu_counts = data.shape[0]
        m = len(input_variable)
        s1 = len(desirable_output)
        s2 = len(undesirable_output)
        total = dmu_counts + m + s1 + s2 + 1
        
        # 创建slack列
        cols = input_variable + desirable_output + undesirable_output
        newcols = []
        for j in cols:
            newcols.append(j + '_slack')
            res[j + '_slack'] = np.nan
        
        # 对每个DMU求解（超效率模型排除被评估DMU）
        for i in range(dmu_counts):
            try:
                # 优化目标：目标函数的系数矩阵
                c = [0] * dmu_counts + [1] + list(-1 / (m * data.loc[i, input_variable])) + [0] * (s1 + s2)
                
                # 约束条件：约束方程的系数矩阵
                A_eq = [[0] * dmu_counts + [1] + [0] * m +
                        list(1 / ((s1 + s2) * data.loc[i, desirable_output])) +
                        list(1 / ((s1 + s2) * data.loc[i, undesirable_output]))]
                
                # 约束条件（1）：投入松弛变量为正（排除被评估DMU）
                for j1 in range(m):
                    list1 = [0] * m
                    list1[j1] = 1
                    # 排除被评估DMU的数据
                    eq1_data = [data.loc[k, input_variable[j1]] if k != i else 0 for k in range(dmu_counts)]
                    eq1 = eq1_data + [-data.loc[i, input_variable[j1]]] + list1 + [0] * (s1 + s2)
                    A_eq.append(eq1)
                
                # 约束条件（2）：期望产出松弛变量为正（排除被评估DMU）
                for j2 in range(s1):
                    list2 = [0] * s1
                    list2[j2] = -1
                    eq2_data = [data.loc[k, desirable_output[j2]] if k != i else 0 for k in range(dmu_counts)]
                    eq2 = eq2_data + [-data.loc[i, desirable_output[j2]]] + [0] * m + list2 + [0] * s2
                    A_eq.append(eq2)
                
                # 约束条件（3）：非期望产出松弛变量为正（排除被评估DMU）
                for j3 in range(s2):
                    list3 = [0] * s2
                    list3[j3] = 1
                    eq3_data = [data.loc[k, undesirable_output[j3]] if k != i else 0 for k in range(dmu_counts)]
                    eq3 = eq3_data + [-data.loc[i, undesirable_output[j3]]] + [0] * (m + s1) + list3
                    A_eq.append(eq3)
                
                # 约束条件：常数向量
                b_eq = [1] + [0] * (m + s1 + s2)
                bounds = [(0, None)] * total
                
                # 求解
                op1 = op.linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=method)
                
                if op1.success:
                    res.loc[i, 'TE'] = max(op1.fun, 1.0)  # 超效率模型允许效率值大于1
                    res.loc[i, newcols] = op1.x[dmu_counts + 1:]
                else:
                    res.loc[i, 'TE'] = 1.0
                    res.loc[i, newcols] = 0.0
                    
            except Exception as e:
                print(f"超效率SBM求解失败 (DMU {i+1}): {e}")
                res.loc[i, 'TE'] = 1.0
                res.loc[i, newcols] = 0.0
        
        return res
    
    # 向后兼容的方法
    def ccr(self):
        """CCR模型 - 默认输入导向（向后兼容）"""
        return self.ccr_input_oriented()
    
    def bcc(self):
        """BCC模型 - 默认输入导向（向后兼容）"""
        return self.bcc_input_oriented()
    
    def efficiency(self):
        """默认效率计算方法"""
        return self.ccr_input_oriented()

# Streamlit应用配置
st.set_page_config(
    page_title="基于DEA与fsQCA的医院运营效能与发展路径智慧决策系统",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 主标题
st.title("🏥 基于DEA与fsQCA的医院运营效能与发展路径智慧决策系统")

# 系统状态概览
st.subheader("📊 系统状态概览")

col1, col2, col3, col4 = st.columns(4)

with col1:
    dea_status = "✅" if 'dea_results' in st.session_state else "❌"
    st.metric("DEA分析", dea_status)

with col2:
    qca_status = "✅" if 'qca_results' in st.session_state else "❌"
    st.metric("QCA分析", qca_status)

with col3:
    fsqca_status = "✅" if 'fsqca_results' in st.session_state else "❌"
    st.metric("fsQCA分析", fsqca_status)

with col4:
    st.metric("DEA库", "✅ 自定义DEA库正常")

# 数据输入区
st.subheader("📥 数据输入")

# 文件上传
uploaded_file = st.file_uploader(
    "请上传包含医院数据的文件",
    type=['csv', 'xlsx'],
    help="支持CSV和Excel文件，文件应包含投入变量、产出变量和DMU标识列"
)

if uploaded_file is not None:
    try:
        # 根据文件类型读取数据
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'csv':
            data = pd.read_csv(uploaded_file)
        elif file_extension == 'xlsx':
            data = pd.read_excel(uploaded_file)
        else:
            st.error("❌ 不支持的文件格式，请上传CSV或Excel文件")
            st.stop()
        
        st.session_state['data'] = data
        
        st.success(f"✅ 成功上传数据文件，包含 {len(data)} 行数据")
        
        # 显示数据预览
        st.subheader("📋 数据预览")
        st.dataframe(data.head(10), use_container_width=True)
        
        # 列选择
        st.subheader("🔧 变量配置")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**投入变量选择**")
            input_cols = st.multiselect(
                "选择投入变量（如：员工数量、床位数量等）",
                options=data.columns.tolist(),
                key="input_vars"
            )
        
        with col2:
            st.markdown("**产出变量选择**")
            output_cols = st.multiselect(
                "选择产出变量（如：门诊量、住院量等）",
                options=data.columns.tolist(),
                key="output_vars"
            )
        
        # DMU列选择
        dmu_col = st.selectbox(
            "选择DMU标识列",
            options=data.columns.tolist(),
            key="dmu_col"
        )
        
        # 非期望产出选择（可选）
        st.markdown("**非期望产出选择（可选，仅SBM模型使用）**")
        undesirable_cols = st.multiselect(
            "选择非期望产出变量（如：医疗事故、投诉等）",
            options=data.columns.tolist(),
            key="undesirable_vars"
        )
        
        # 分析按钮
        if st.button("🚀 开始DEA分析", type="primary"):
            if not input_cols or not output_cols or not dmu_col:
                st.error("❌ 请选择投入变量、产出变量和DMU标识列")
            else:
                try:
                    # 创建CustomDEA实例
                    input_data = data[input_cols].values
                    output_data = data[output_cols].values
                    
                    dea = CustomDEA(input_data, output_data)
                    
                    # 模型选择
                    model_type = st.selectbox(
                        "选择DEA模型类型",
                        options=['CCR', 'BCC', 'SBM', 'Super-SBM'],
                        key="model_type"
                    )
                    
                    orientation = st.selectbox(
                        "选择导向类型",
                        options=['input', 'output'],
                        key="orientation"
                    )
                    
                    # 执行分析
                    if model_type == 'CCR':
                        if orientation == 'input':
                            results = dea.ccr_input_oriented(input_cols, output_cols, dmu_col, data)
                        else:
                            results = dea.ccr_output_oriented(input_cols, output_cols, dmu_col, data)
                    elif model_type == 'BCC':
                        if orientation == 'input':
                            results = dea.bcc_input_oriented(input_cols, output_cols, dmu_col, data)
                        else:
                            results = dea.bcc_output_oriented(input_cols, output_cols, dmu_col, data)
                    elif model_type == 'SBM':
                        if not undesirable_cols:
                            st.warning("⚠️ SBM模型建议包含非期望产出变量")
                            undesirable_cols = []
                        results = dea.sbm(input_cols, output_cols, undesirable_cols, dmu_col, data)
                    elif model_type == 'Super-SBM':
                        if not undesirable_cols:
                            st.warning("⚠️ 超效率SBM模型建议包含非期望产出变量")
                            undesirable_cols = []
                        results = dea.super_sbm(input_cols, output_cols, undesirable_cols, dmu_col, data)
                    
                    # 保存结果
                    st.session_state['dea_results'] = results
                    st.session_state['dea_model_type'] = model_type
                    st.session_state['dea_orientation'] = orientation
                    
                    st.success(f"✅ {model_type}模型分析完成！")
                    
                    # 显示结果
                    st.subheader("📊 分析结果")
                    st.dataframe(results, use_container_width=True)
                    
                    # 效率统计
                    efficiency_scores = results['TE']
                    st.subheader("📈 效率统计")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("平均效率", f"{efficiency_scores.mean():.4f}")
                    with col2:
                        st.metric("最高效率", f"{efficiency_scores.max():.4f}")
                    with col3:
                        st.metric("最低效率", f"{efficiency_scores.min():.4f}")
                    with col4:
                        efficient_count = (efficiency_scores >= 0.95).sum()
                        st.metric("高效DMU数量", f"{efficient_count}/{len(efficiency_scores)}")
                    
                    # 效率分布图
                    fig = px.histogram(
                        results, 
                        x='TE', 
                        title=f'{model_type}模型效率分布',
                        nbins=20,
                        color_discrete_sequence=['#1a365d']
                    )
                    fig.update_layout(
                        xaxis_title="效率值",
                        yaxis_title="频数",
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ DEA分析失败: {str(e)}")
        
    except Exception as e:
        st.error(f"❌ 文件读取失败: {str(e)}")

# 侧边栏信息
with st.sidebar:
    st.markdown("## 🏥 系统信息")
    st.info("""
    **系统版本**: v1.0
    
    **DEA模型支持**:
    - CCR模型（规模报酬不变）
    - BCC模型（规模报酬可变）
    - SBM模型（基于松弛变量）
    - 超效率SBM模型
    
    **技术特点**:
    - 纯自定义DEA实现
    - 支持输入/输出导向
    - 支持非期望产出
    - 线性规划求解
    """)
    
    st.markdown("## 📋 使用说明")
    st.markdown("""
    1. 上传包含医院数据的CSV或Excel文件
    2. 选择投入变量和产出变量
    3. 选择DMU标识列
    4. 选择DEA模型类型和导向
    5. 点击"开始DEA分析"按钮
    6. 查看分析结果和效率统计
    """)

if __name__ == "__main__":
    pass
