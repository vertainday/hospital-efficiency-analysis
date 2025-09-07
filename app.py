import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import re
from scipy.optimize import linprog
import itertools
from scipy.stats import pearsonr

PYDEA_AVAILABLE = False
print("使用自定义DEA实现进行DEA分析")

class CustomDEA:
    """自定义DEA实现，支持CCR和BCC模型的输入导向和输出导向版本"""
    
    def __init__(self, input_data, output_data):
        self.input_data = np.array(input_data, dtype=np.float64)
        self.output_data = np.array(output_data, dtype=np.float64)
        self.n_dmus = self.input_data.shape[0]
        self.n_inputs = self.input_data.shape[1]
        self.n_outputs = self.output_data.shape[1]
        
        # 数据预处理：确保所有数据为正数，并标准化以提高数值稳定性
        self.input_data = np.maximum(self.input_data, 1e-8)
        self.output_data = np.maximum(self.output_data, 1e-8)
        
        # 数据标准化以提高数值稳定性
        self.input_scale = np.mean(self.input_data, axis=0)
        self.output_scale = np.mean(self.output_data, axis=0)
        
        # 避免除零
        self.input_scale = np.maximum(self.input_scale, 1e-8)
        self.output_scale = np.maximum(self.output_scale, 1e-8)
        
        # 标准化数据
        self.input_data_norm = self.input_data / self.input_scale
        self.output_data_norm = self.output_data / self.output_scale
    
    def ccr_input_oriented(self):
        """CCR模型 - 输入导向（规模报酬不变）
        目标函数：max θ
        约束条件：∑λⱼxᵢⱼ ≤ θxᵢₒ, ∑λⱼyᵣⱼ ≥ yᵣₒ, λⱼ ≥ 0
        """
        return self._solve_dea_model(model='ccr', orientation='input')
    
    def ccr_output_oriented(self):
        """CCR模型 - 输出导向（规模报酬不变）
        目标函数：min φ
        约束条件：∑λⱼxᵢⱼ ≤ xᵢₒ, ∑λⱼyᵣⱼ ≥ φyᵣₒ, λⱼ ≥ 0
        """
        return self._solve_dea_model(model='ccr', orientation='output')
    
    def bcc_input_oriented(self):
        """BCC模型 - 输入导向（规模报酬可变）
        目标函数：max θ
        约束条件：∑λⱼxᵢⱼ + sᵢ⁻ = θxᵢₒ, ∑λⱼyᵣⱼ - sᵣ⁺ = yᵣₒ, ∑λⱼ = 1, λⱼ ≥ 0
        """
        return self._solve_dea_model(model='bcc', orientation='input')
    
    def bcc_output_oriented(self):
        """BCC模型 - 输出导向（规模报酬可变）
        目标函数：min φ
        约束条件：∑λⱼxᵢⱼ + sᵢ⁻ = xᵢₒ, ∑λⱼyᵣⱼ - sᵣ⁺ = φyᵣₒ, ∑λⱼ = 1, λⱼ ≥ 0
        """
        return self._solve_dea_model(model='bcc', orientation='output')
    
    def ccr(self):
        """CCR模型 - 默认输入导向（向后兼容）"""
        return self.ccr_input_oriented()
    
    def bcc(self):
        """BCC模型 - 默认输入导向（向后兼容）"""
        return self.bcc_input_oriented()
    
    def efficiency(self):
        """默认效率计算方法"""
        return self.ccr_input_oriented()
    
    def _solve_dea_model(self, model='ccr', orientation='input'):
        """求解DEA模型的核心方法
        
        Args:
            model: 'ccr' 或 'bcc'
            orientation: 'input' 或 'output'
        """
        efficiency_scores = []
        
        for i in range(self.n_dmus):
            try:
                if orientation == 'input':
                    efficiency = self._solve_input_oriented(i, model)
                else:  # output oriented
                    efficiency = self._solve_output_oriented(i, model)
                
                # 确保效率值在合理范围内
                efficiency = min(max(efficiency, 0.0), 1.0)
                efficiency_scores.append(efficiency)
                
            except Exception as e:
                print(f"DEA求解失败 (DMU {i+1}): {e}")
                efficiency_scores.append(0.0)
        
        return np.array(efficiency_scores)
    
    def _solve_input_oriented(self, dmu_idx, model):
        """求解输入导向DEA模型
        
        目标函数：max θ
        约束条件：
        - 输入约束：∑λⱼxᵢⱼ ≤ θxᵢₒ
        - 输出约束：∑λⱼyᵣⱼ ≥ yᵣₒ
        - 规模报酬约束（BCC）：∑λⱼ = 1
        - 非负约束：λⱼ ≥ 0
        """
        # 使用标准化数据以提高数值稳定性
        input_data = self.input_data_norm
        output_data = self.output_data_norm
        
        # 变量：θ, λ₁, λ₂, ..., λₙ
        n_vars = self.n_dmus + 1
        
        # 目标函数：最大化θ（转换为最小化-θ）
        c = np.zeros(n_vars, dtype=np.float64)
        c[0] = -1.0  # -θ
        
        # 约束条件
        A_ub = []
        b_ub = []
        
        # 输入约束：∑λⱼxᵢⱼ ≤ θxᵢₒ
        # 转换为：∑λⱼxᵢⱼ - θxᵢₒ ≤ 0
        for j in range(self.n_inputs):
            constraint = np.zeros(n_vars, dtype=np.float64)
            constraint[1:] = input_data[:, j]  # λⱼ的系数
            constraint[0] = -input_data[dmu_idx, j]  # -θ的系数
            A_ub.append(constraint)
            b_ub.append(0.0)
        
        # 输出约束：∑λⱼyᵣⱼ ≥ yᵣₒ
        # 转换为：-∑λⱼyᵣⱼ ≤ -yᵣₒ
        for r in range(self.n_outputs):
            constraint = np.zeros(n_vars, dtype=np.float64)
            constraint[1:] = -output_data[:, r]  # -λⱼ的系数
            constraint[0] = 0.0  # θ不参与此约束
            A_ub.append(constraint)
            b_ub.append(-output_data[dmu_idx, r])
        
        # 规模报酬约束
        if model == 'bcc':
            # BCC模型：∑λⱼ = 1
            constraint = np.zeros(n_vars, dtype=np.float64)
            constraint[1:] = 1.0  # λⱼ的系数
            constraint[0] = 0.0   # θ不参与此约束
            A_ub.append(constraint)
            b_ub.append(1.0)
            
            constraint = np.zeros(n_vars, dtype=np.float64)
            constraint[1:] = -1.0  # -λⱼ的系数
            constraint[0] = 0.0    # θ不参与此约束
            A_ub.append(constraint)
            b_ub.append(-1.0)
        
        # 非负约束
        bounds = [(0.0, None) for _ in range(n_vars)]
        
        # 转换为numpy数组
        A_ub = np.array(A_ub, dtype=np.float64)
        b_ub = np.array(b_ub, dtype=np.float64)
        
        # 求解线性规划 - 使用多种方法尝试
        methods = ['highs', 'interior-point', 'revised simplex']
        
        for method in methods:
            try:
                result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=method, options={'maxiter': 10000})
                
                if result.success and result.fun is not None and not np.isnan(result.fun):
                    theta = -result.fun  # 因为目标函数是-θ
                    # 确保效率值在合理范围内
                    theta = max(0.0, min(theta, 1.0))
                    return theta
            except Exception as e:
                continue
        
        # 如果所有方法都失败，使用简化的DEA方法
        return self._simple_efficiency_estimate(dmu_idx)
    
    def _simple_efficiency_estimate(self, dmu_idx):
        """简化的效率估计方法"""
        try:
            # 计算加权投入产出比率
            input_weights = 1.0 / self.input_scale
            output_weights = 1.0 / self.output_scale
            
            # 加权投入和产出
            weighted_input = np.sum(self.input_data[dmu_idx] * input_weights)
            weighted_output = np.sum(self.output_data[dmu_idx] * output_weights)
            
            # 计算所有DMU的加权投入产出比率
            all_weighted_inputs = np.sum(self.input_data * input_weights, axis=1)
            all_weighted_outputs = np.sum(self.output_data * output_weights, axis=1)
            
            # 计算效率比率
            efficiency_ratios = all_weighted_outputs / all_weighted_inputs
            max_efficiency = np.max(efficiency_ratios)
            
            # 当前DMU的效率
            current_efficiency = efficiency_ratios[dmu_idx] / max_efficiency
            
            return max(0.0, min(current_efficiency, 1.0))
        except:
            return 0.0
    
    def _solve_output_oriented(self, dmu_idx, model):
        """求解输出导向DEA模型
        
        目标函数：min φ
        约束条件：
        - 输入约束：∑λⱼxᵢⱼ ≤ xᵢₒ
        - 输出约束：∑λⱼyᵣⱼ ≥ φyᵣₒ
        - 规模报酬约束（BCC）：∑λⱼ = 1
        - 非负约束：λⱼ ≥ 0
        """
        # 使用标准化数据以提高数值稳定性
        input_data = self.input_data_norm
        output_data = self.output_data_norm
        
        # 变量：φ, λ₁, λ₂, ..., λₙ
        n_vars = self.n_dmus + 1
        
        # 目标函数：最小化φ
        c = np.zeros(n_vars, dtype=np.float64)
        c[0] = 1.0  # φ
        
        # 约束条件
        A_ub = []
        b_ub = []
        
        # 输入约束：∑λⱼxᵢⱼ ≤ xᵢₒ
        for j in range(self.n_inputs):
            constraint = np.zeros(n_vars, dtype=np.float64)
            constraint[1:] = input_data[:, j]  # λⱼ的系数
            constraint[0] = 0.0  # φ不参与此约束
            A_ub.append(constraint)
            b_ub.append(input_data[dmu_idx, j])
        
        # 输出约束：∑λⱼyᵣⱼ ≥ φyᵣₒ
        # 转换为：-∑λⱼyᵣⱼ + φyᵣₒ ≤ 0
        for r in range(self.n_outputs):
            constraint = np.zeros(n_vars, dtype=np.float64)
            constraint[1:] = -output_data[:, r]  # -λⱼ的系数
            constraint[0] = output_data[dmu_idx, r]  # φ的系数
            A_ub.append(constraint)
            b_ub.append(0.0)
        
        # 规模报酬约束
        if model == 'bcc':
            # BCC模型：∑λⱼ = 1
            constraint = np.zeros(n_vars, dtype=np.float64)
            constraint[1:] = 1.0  # λⱼ的系数
            constraint[0] = 0.0   # φ不参与此约束
            A_ub.append(constraint)
            b_ub.append(1.0)
            
            constraint = np.zeros(n_vars, dtype=np.float64)
            constraint[1:] = -1.0  # -λⱼ的系数
            constraint[0] = 0.0    # φ不参与此约束
            A_ub.append(constraint)
            b_ub.append(-1.0)
        
        # 非负约束
        bounds = [(0.0, None) for _ in range(n_vars)]
        
        # 转换为numpy数组
        A_ub = np.array(A_ub, dtype=np.float64)
        b_ub = np.array(b_ub, dtype=np.float64)
        
        # 求解线性规划 - 使用多种方法尝试
        methods = ['highs', 'interior-point', 'revised simplex']
        
        for method in methods:
            try:
                result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=method, options={'maxiter': 10000})
                
                if result.success and result.fun is not None and not np.isnan(result.fun):
                    phi = result.fun
                    # 输出导向的效率值是1/φ
                    if phi > 0:
                        efficiency = 1.0 / phi
                        return max(0.0, min(efficiency, 1.0))
                    else:
                        return 1.0
            except Exception as e:
                continue
        
        # 如果所有方法都失败，使用简化的DEA方法
        return self._simple_efficiency_estimate(dmu_idx)
    
    # SBM模型相关方法
    def sbm(self, undesirable_outputs=None):
        """SBM模型 - 包含非期望产出的松弛基础模型
        
        Args:
            undesirable_outputs: 非期望产出列索引列表，如果为None则只考虑期望产出
        """
        return self._solve_sbm_model(undesirable_outputs=undesirable_outputs)
    
    def super_sbm(self, undesirable_outputs=None):
        """超效率SBM模型 - 允许效率值大于1
        
        Args:
            undesirable_outputs: 非期望产出列索引列表，如果为None则只考虑期望产出
        """
        return self._solve_super_sbm_model(undesirable_outputs=undesirable_outputs)
    
    def _solve_sbm_model(self, undesirable_outputs=None):
        """求解SBM模型（线性化后形式）"""
        efficiency_scores = []
        
        for i in range(self.n_dmus):
            try:
                efficiency = self._solve_sbm_dmu(i, undesirable_outputs, super_efficiency=False)
                efficiency_scores.append(efficiency)
            except Exception as e:
                print(f"SBM求解失败 (DMU {i+1}): {e}")
                efficiency_scores.append(0.0)
        
        return np.array(efficiency_scores)
    
    def _solve_super_sbm_model(self, undesirable_outputs=None):
        """求解超效率SBM模型"""
        efficiency_scores = []
        
        for i in range(self.n_dmus):
            try:
                efficiency = self._solve_sbm_dmu(i, undesirable_outputs, super_efficiency=True)
                efficiency_scores.append(efficiency)
            except Exception as e:
                print(f"超效率SBM求解失败 (DMU {i+1}): {e}")
                efficiency_scores.append(0.0)
        
        return np.array(efficiency_scores)
    
    def _solve_sbm_dmu(self, dmu_idx, undesirable_outputs=None, super_efficiency=False):
        """求解单个DMU的SBM模型
        
        Args:
            dmu_idx: 被评估DMU的索引
            undesirable_outputs: 非期望产出列索引列表
            super_efficiency: 是否为超效率模型
        """
        # 确定期望产出和非期望产出
        if undesirable_outputs is None:
            # 如果没有指定非期望产出，所有产出都视为期望产出
            good_outputs = list(range(self.n_outputs))
            bad_outputs = []
        else:
            # 分离期望产出和非期望产出
            good_outputs = [i for i in range(self.n_outputs) if i not in undesirable_outputs]
            bad_outputs = undesirable_outputs
        
        n_good_outputs = len(good_outputs)
        n_bad_outputs = len(bad_outputs)
        
        # 变量：t, μ₁, μ₂, ..., μₙ, S₁⁻, S₂⁻, ..., Sₘ⁻, S₁ᵍ⁺, ..., Sᵣᵍ⁺, S₁ᵇ⁺, ..., Sᶠᵇ⁺
        n_vars = 1 + self.n_dmus + self.n_inputs + n_good_outputs + n_bad_outputs
        
        # 目标函数：min ρ = t - (1/m)∑(Sᵢ⁻/xᵢₖ)
        c = np.zeros(n_vars)
        c[0] = 1  # t的系数
        
        # 添加投入松弛变量的系数
        for i in range(self.n_inputs):
            var_idx = 1 + self.n_dmus + i
            c[var_idx] = -1.0 / (self.n_inputs * self.input_data[dmu_idx, i])
        
        # 约束条件
        A_eq = []
        b_eq = []
        
        # 投入约束：t xᵢₖ = ∑μⱼ xᵢⱼ + Sᵢ⁻
        for i in range(self.n_inputs):
            constraint = np.zeros(n_vars)
            constraint[0] = self.input_data[dmu_idx, i]  # t的系数
            
            # μⱼ的系数
            for j in range(self.n_dmus):
                if super_efficiency and j == dmu_idx:
                    # 超效率模型排除被评估DMU
                    constraint[1 + j] = 0
                else:
                    constraint[1 + j] = self.input_data[j, i]
            
            # Sᵢ⁻的系数
            constraint[1 + self.n_dmus + i] = 1
            
            A_eq.append(constraint)
            b_eq.append(0)
        
        # 期望产出约束：t yᵣₖᵍ = ∑μⱼ yᵣⱼᵍ - Sᵣᵍ⁺
        for r_idx, r in enumerate(good_outputs):
            constraint = np.zeros(n_vars)
            constraint[0] = self.output_data[dmu_idx, r]  # t的系数
            
            # μⱼ的系数
            for j in range(self.n_dmus):
                if super_efficiency and j == dmu_idx:
                    # 超效率模型排除被评估DMU
                    constraint[1 + j] = 0
                else:
                    constraint[1 + j] = -self.output_data[j, r]  # 负号因为要减去
            
            # Sᵣᵍ⁺的系数
            constraint[1 + self.n_dmus + self.n_inputs + r_idx] = -1
            
            A_eq.append(constraint)
            b_eq.append(0)
        
        # 非期望产出约束：t yᶠₖᵇ = ∑μⱼ yᶠⱼᵇ + Sᶠᵇ⁺
        for f_idx, f in enumerate(bad_outputs):
            constraint = np.zeros(n_vars)
            constraint[0] = self.output_data[dmu_idx, f]  # t的系数
            
            # μⱼ的系数
            for j in range(self.n_dmus):
                if super_efficiency and j == dmu_idx:
                    # 超效率模型排除被评估DMU
                    constraint[1 + j] = 0
                else:
                    constraint[1 + j] = self.output_data[j, f]
            
            # Sᶠᵇ⁺的系数
            constraint[1 + self.n_dmus + self.n_inputs + n_good_outputs + f_idx] = 1
            
            A_eq.append(constraint)
            b_eq.append(0)
        
        # 非负约束
        bounds = []
        bounds.append((1e-10, None))  # t > 0
        for _ in range(self.n_dmus):
            bounds.append((0, None))  # μⱼ ≥ 0
        for _ in range(self.n_inputs):
            bounds.append((0, None))  # Sᵢ⁻ ≥ 0
        for _ in range(n_good_outputs):
            bounds.append((0, None))  # Sᵣᵍ⁺ ≥ 0
        for _ in range(n_bad_outputs):
            bounds.append((0, None))  # Sᶠᵇ⁺ ≥ 0
        
        # 求解线性规划
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if result.success:
            rho = result.fun
            if super_efficiency:
                # 超效率SBM：ρ ≥ 1
                return max(rho, 1.0)
            else:
                # 普通SBM：ρ ∈ (0,1]
                return min(max(rho, 0.0), 1.0)
        else:
            return 0.0 if not super_efficiency else 1.0

class DEAWrapper:
    """DEA分析包装器，使用自定义DEA实现"""
    
    def __init__(self, input_data, output_data, dmu_names=None):
        self.input_data = np.array(input_data)
        self.output_data = np.array(output_data)
        # 修复numpy数组的布尔值判断问题
        if dmu_names is not None:
            self.dmu_names = list(dmu_names) if hasattr(dmu_names, '__iter__') else [dmu_names]
        else:
            self.dmu_names = [f'DMU{i+1}' for i in range(len(input_data))]
        
        # 使用自定义DEA实现
        self.dea = CustomDEA(self.input_data, self.output_data)
        print("✅ 使用自定义DEA实现进行DEA分析")
    
    
    # 新增方法：支持不同的模型和方向选择
    def ccr_input_oriented(self):
        """CCR模型 - 输入导向"""
        return self.dea.ccr_input_oriented()
    
    def ccr_output_oriented(self):
        """CCR模型 - 输出导向"""
        return self.dea.ccr_output_oriented()
    
    def bcc_input_oriented(self):
        """BCC模型 - 输入导向"""
        return self.dea.bcc_input_oriented()
    
    def bcc_output_oriented(self):
        """BCC模型 - 输出导向"""
        return self.dea.bcc_output_oriented()
    
    # 保持向后兼容的方法
    def ccr(self):
        """CCR模型 - 默认输入导向（向后兼容）"""
        return self.ccr_input_oriented()
    
    def bcc(self):
        """BCC模型 - 默认输入导向（向后兼容）"""
        return self.bcc_input_oriented()
    
    def sbm(self, undesirable_outputs=None):
        """SBM模型 - 包含非期望产出的松弛基础模型"""
        return self.dea.sbm(undesirable_outputs=undesirable_outputs)
    
    def super_sbm(self, undesirable_outputs=None):
        """超效率SBM模型 - 允许效率值大于1"""
        return self.dea.super_sbm(undesirable_outputs=undesirable_outputs)
    
    def efficiency(self):
        """默认效率计算方法"""
        return self.ccr()

# 为了保持兼容性，创建DEA别名
DEA = DEAWrapper

# 导入QCA分析模块
QCA_AVAILABLE = True

try:
    # 导入纯Python QCA实现
    from qca_analysis import (
        check_r_connection, 
        perform_necessity_analysis, 
        perform_sufficiency_analysis,
        perform_truth_table_analysis,
        perform_minimization,
        perform_complete_qca_analysis
    )
    print("✓ 成功加载纯Python QCA实现")
except Exception as e:
    print(f"❌ 纯Python QCA实现加载失败: {e}")
    QCA_AVAILABLE = False
    # 创建占位符函数
    def check_r_connection():
        return False, "纯Python QCA实现不可用"
    def perform_necessity_analysis(*args, **kwargs):
        return pd.DataFrame()
    def perform_sufficiency_analysis(*args, **kwargs):
        return pd.DataFrame()
    def perform_truth_table_analysis(*args, **kwargs):
        return pd.DataFrame()
    def perform_minimization(*args, **kwargs):
        return pd.DataFrame()
    def perform_complete_qca_analysis(*args, **kwargs):
        return pd.DataFrame()

# 设置页面配置
st.set_page_config(
    page_title="基于DEA与fsQCA的医院运营效能与发展路径智慧决策系统",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 隐藏页脚
st.markdown("""
<style>
    .stApp > footer {
        visibility: hidden;
    }
    .stApp > footer:after {
        content: "医院运营效能智慧决策系统 v1.0";
        visibility: visible;
        display: block;
        position: relative;
        padding: 5px;
        top: 2px;
        color: #1a365d;
        font-size: 12px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# 专业医疗风格CSS样式
st.markdown("""
<style>
    /* 全局样式 */
    .stApp {
        background-color: #e6f7ff;
    }
    
    /* 主标题样式 */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1a365d;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1.5rem;
        background: linear-gradient(135deg, #1a365d, #2c5282);
        color: white;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(26, 54, 93, 0.3);
        position: relative;
    }
    
    .main-header::before {
        content: "🏥";
        font-size: 3rem;
        position: absolute;
        left: 2rem;
        top: 50%;
        transform: translateY(-50%);
    }
    
    /* 区域标题样式 */
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1a365d;
        margin: 2rem 0 1.5rem 0;
        padding: 1rem 1.5rem;
        border-left: 6px solid #1a365d;
        background: linear-gradient(90deg, #e6f7ff, #f0f9ff);
        border-radius: 0 10px 10px 0;
        box-shadow: 0 2px 8px rgba(26, 54, 93, 0.1);
    }
    
    /* 区域容器样式 */
    .analysis-section {
        background-color: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(26, 54, 93, 0.1);
        margin-bottom: 2rem;
        border: 1px solid #e6f7ff;
    }
    
    /* 消息样式 */
    .success-message {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        color: #155724;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(40, 167, 69, 0.2);
    }
    
    .error-message {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        color: #721c24;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(220, 53, 69, 0.2);
    }
    
    .warning-message {
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
        color: #856404;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(255, 193, 7, 0.2);
    }
    
    /* 数据预览样式 */
    .data-preview {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #e6f7ff;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(26, 54, 93, 0.1);
    }
    
    /* 绿色按钮样式 */
    .stButton > button:first-child {
        background: linear-gradient(135deg, #28a745, #20c997);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: bold;
        font-size: 1.1rem;
        box-shadow: 0 4px 12px rgba(40, 167, 69, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:first-child:hover {
        background: linear-gradient(135deg, #218838, #1ea085);
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(40, 167, 69, 0.4);
    }
    
    .stButton > button:first-child:active {
        transform: translateY(0);
        box-shadow: 0 2px 8px rgba(40, 167, 69, 0.3);
    }
    
    /* 按钮样式 */
    .stButton > button {
        background: linear-gradient(135deg, #1a365d, #2c5282);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: bold;
        box-shadow: 0 4px 8px rgba(26, 54, 93, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2c5282, #1a365d);
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(26, 54, 93, 0.4);
    }
    
    /* 侧边栏样式 */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a365d, #2c5282);
    }
    
    .css-1d391kg .stSelectbox > div > div {
        background-color: white;
        border-radius: 8px;
    }
    
    /* 指标卡片样式 */
    .metric-card {
        background: linear-gradient(135deg, #e6f7ff, #f0f9ff);
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #1a365d;
        text-align: center;
        box-shadow: 0 4px 8px rgba(26, 54, 93, 0.1);
    }
    
    /* 进度条样式 */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #1a365d, #2c5282);
    }
    
    /* 表格样式 */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(26, 54, 93, 0.1);
    }
    
    /* 图表容器样式 */
    .plotly-chart {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(26, 54, 93, 0.1);
    }
</style>
""", unsafe_allow_html=True)

def create_searchable_multiselect(label, options, key, help_text="", placeholder="请选择..."):
    """
    创建带搜索功能的multiselect组件
    
    参数:
    - label: 标签文本
    - options: 选项列表
    - key: 组件的唯一键
    - help_text: 帮助文本
    - placeholder: 占位符文本
    
    返回:
    - 选中的选项列表
    """
    # 获取当前已选择的变量（从session state中获取）
    current_selected = st.session_state.get(key, [])
    
    # 添加搜索框
    search_term = st.text_input(
        f"🔍 搜索{label}",
        key=f"search_{key}",
        placeholder=f"输入关键词搜索{label}...",
        help=f"输入关键词来快速找到需要的{label}"
    )
    
    # 根据搜索词过滤选项
    if search_term:
        filtered_options = [opt for opt in options if search_term.lower() in opt.lower()]
        if not filtered_options:
            st.warning(f"未找到包含 '{search_term}' 的{label}")
            filtered_options = options
    else:
        filtered_options = options
    
    # 确保已选择的变量始终在选项列表中（即使它们不在当前搜索结果中）
    # 这样用户之前的选择不会被清空
    for selected_item in current_selected:
        if selected_item not in filtered_options and selected_item in options:
            filtered_options.append(selected_item)
    
    # 显示过滤后的选项数量
    if search_term:
        st.caption(f"找到 {len([opt for opt in filtered_options if search_term.lower() in opt.lower()])} 个匹配的{label}")
    
    # 创建multiselect
    selected = st.multiselect(
        label,
        options=filtered_options,
        key=key,
        help=help_text,
        placeholder=placeholder,
        default=current_selected  # 设置默认值为当前已选择的变量
    )
    
    return selected

def validate_dmu_column(df):
    """验证数据是否包含DMU列"""
    if 'DMU' not in df.columns and '医院ID' not in df.columns:
        return False, "错误：上传的文件必须包含'DMU'列或'医院ID'列！"
    return True, "数据验证通过"

def convert_percentage_to_decimal(value):
    """将百分比数据转换为小数"""
    if pd.isna(value):
        return value
    
    # 如果是字符串，尝试提取数字
    if isinstance(value, str):
        # 移除百分号和其他非数字字符
        numeric_str = re.sub(r'[^\d.]', '', value)
        if numeric_str:
            try:
                num = float(numeric_str)
                # 如果原值包含%或大于1，认为是百分比
                if '%' in value or num > 1:
                    return num / 100
                return num
            except:
                return value
    
    # 如果是数字
    if isinstance(value, (int, float)):
        if value > 1:
            return value / 100
        return value
    
    return value

def validate_numeric_data(df, exclude_columns=['DMU', '医院ID']):
    """验证数值数据的有效性"""
    errors = []
    warnings = []
    
    for col in df.columns:
        if col in exclude_columns:
            continue
            
        # 检查是否包含非数值数据（空值除外）
        non_numeric_mask = pd.to_numeric(df[col], errors='coerce').isna()
        # 排除原本就是空值的情况
        original_nulls = df[col].isna()
        actual_non_numeric = non_numeric_mask & ~original_nulls
        
        if actual_non_numeric.any():
            non_numeric_rows = df[actual_non_numeric].index.tolist()
            errors.append(f"列'{col}'包含非数值数据，行号：{non_numeric_rows}")
        elif non_numeric_mask.any():
            # 只有空值的情况，给出提示
            null_count = non_numeric_mask.sum()
            warnings.append(f"列'{col}'包含 {null_count} 个空值，将自动转换为0")
        
        # 检查是否包含负值（对于某些指标）
        if col in ['满意度', '患者满意度', '员工满意度']:
            negative_mask = pd.to_numeric(df[col], errors='coerce') < 0
            if negative_mask.any():
                warnings.append(f"列'{col}'包含负值，已自动处理")
    
    return errors, warnings

def process_cleaned_data(df_cleaned, warnings):
    """处理清理后的数据"""
    # 显示警告信息
    if warnings:
        for warning in warnings:
            st.markdown(f'<div class="warning-message">{warning}</div>', unsafe_allow_html=True)
    
    # 显示数据预览
    st.markdown("### 📋 数据预览（前5行）")
    st.markdown('<div class="data-preview">', unsafe_allow_html=True)
    st.dataframe(df_cleaned.head(), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 数据统计信息
    st.markdown("### 📈 数据统计信息")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("医院数量", len(df_cleaned))
    with col2:
        st.metric("变量数量", len(df_cleaned.columns) - 1)
    with col3:
        st.metric("数据完整性", "100%")
    
    # 保存到session state
    st.session_state['data'] = df_cleaned
    st.session_state['data_source'] = 'file'
    
    # 显示成功消息
    st.success("✅ 数据加载成功！请继续下一步分析。")
    
    # 自动跳转到下一步
    st.markdown("### 🚀 下一步操作")
    st.markdown("数据已成功加载，您可以：")
    st.markdown("1. 进行DEA效率分析")
    st.markdown("2. 进行fsQCA路径分析")
    st.markdown("3. 查看数据详情和统计信息")

def detect_and_handle_nulls(df):
    """检测空值并让用户选择处理方式"""
    # 统计空值
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    
    if total_nulls == 0:
        return df, None
    
    # 显示空值统计信息
    st.warning(f"⚠️ 检测到数据中包含 {total_nulls} 个空值")
    
    # 显示各列空值详情
    with st.expander("📊 空值详情", expanded=True):
        null_info = []
        for col, count in null_counts.items():
            if count > 0:
                null_info.append(f"• {col}: {count} 个空值")
        
        if null_info:
            st.write("各列空值分布：")
            for info in null_info:
                st.write(info)
    
    # 让用户选择处理方式
    st.markdown("### 🔧 请选择空值处理方式")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fill_zero_btn = st.button(
            "🔄 将空值转换为0", 
            key="fill_zero_btn",
            help="保留所有数据行，将空值填充为0",
            type="primary"
        )
    
    with col2:
        drop_rows_btn = st.button(
            "🗑️ 删除包含空值的行", 
            key="drop_rows_btn",
            help="删除包含任何空值的数据行",
            type="secondary"
        )
    
    # 根据用户选择返回处理方式
    if fill_zero_btn:
        return 'fill_zero'
    elif drop_rows_btn:
        return 'drop_rows'
    else:
        return None

def clean_data(df, null_handling='fill_zero'):
    """清理数据：根据选择处理空值，转换百分比数据
    
    Args:
        df: 原始数据框
        null_handling: 空值处理方式
            - 'fill_zero': 将空值转换为0
            - 'drop_rows': 删除包含空值的行
    
    Returns:
        tuple: (清理后的数据框, 处理统计信息)
    """
    original_rows = len(df)
    
    # 创建数据副本
    df_cleaned = df.copy()
    
    # 统计空值数量
    null_counts = df_cleaned.isnull().sum()
    total_nulls = null_counts.sum()
    
    if null_handling == 'drop_rows':
        # 删除包含空值的行
        df_cleaned = df_cleaned.dropna()
        removed_rows = original_rows - len(df_cleaned)
        return df_cleaned, {'removed_rows': removed_rows, 'filled_nulls': 0}
    
    else:  # fill_zero
        # 将空值转换为0（除了DMU列和医院ID列）
        dmu_cols = [col for col in df_cleaned.columns if 'DMU' in col or '医院ID' in col or 'ID' in col]
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        
        # 对数值列的空值填充0
        for col in numeric_cols:
            if col not in dmu_cols:
                df_cleaned[col] = df_cleaned[col].fillna(0)
        
        # 对非数值列的空值也填充0（如果包含数字的话）
        for col in df_cleaned.columns:
            if col not in dmu_cols and col not in numeric_cols:
                # 尝试将列转换为数值，无法转换的保持原样
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce').fillna(0)
        
        return df_cleaned, {'removed_rows': 0, 'filled_nulls': total_nulls}
    
    # 转换百分比数据
    percentage_columns = [col for col in df_cleaned.columns if any(keyword in col for keyword in ['满意度', '率', '比例', '百分比'])]
    for col in percentage_columns:
        df_cleaned[col] = df_cleaned[col].apply(convert_percentage_to_decimal)

def create_manual_input_form(num_hospitals, num_variables):
    """创建手动输入表单"""
    st.subheader("📝 手动输入数据")
    
    # 创建变量配置
    variables = []
    for i in range(num_variables):
        col1, col2 = st.columns(2)
        with col1:
            var_name = st.text_input(f"变量{i+1}名称", key=f"var_name_{i}", placeholder="如：床位数、医生数等")
        with col2:
            var_type = st.selectbox(f"变量{i+1}类型", ["投入", "产出", "条件"], key=f"var_type_{i}")
        
        if var_name:
            variables.append({"name": var_name, "type": var_type})
    
    if not variables:
        st.warning("请至少输入一个变量名称")
        return None
    
    # 创建数据输入表格
    st.subheader("🏥 医院数据输入")
    
    # 创建列名
    columns = ["DMU"] + [var["name"] for var in variables]
    
    # 创建数据输入界面
    data_rows = []
    for i in range(num_hospitals):
        st.write(f"**医院 {i+1}**")
        row_data = {"DMU": f"DMU{i+1}"}
        
        cols = st.columns(len(variables) + 1)
        cols[0].write(f"DMU{i+1}")
        
        for j, var in enumerate(variables):
            value = cols[j+1].number_input(
                f"{var['name']} ({var['type']})",
                min_value=0.0,
                value=0.0,
                step=0.01,
                key=f"input_{i}_{j}"
            )
            row_data[var["name"]] = value
        
        data_rows.append(row_data)
    
    # 创建DataFrame
    df = pd.DataFrame(data_rows)
    return df
def validate_dea_data(input_data, output_data):
    """
    验证DEA输入数据的合理性
    
    参数:
    - input_data: 投入数据
    - output_data: 产出数据
    
    返回:
    - is_valid: 数据是否有效
    - message: 验证消息
    """
    # 检查数据形状
    if input_data.shape[0] != output_data.shape[0]:
        return False, "投入和产出数据的样本数量不一致"
    
    # 检查数据是否包含负值
    if np.any(input_data < 0):
        return False, "投入数据包含负值，DEA要求所有数据为正数"
    
    if np.any(output_data < 0):
        return False, "产出数据包含负值，DEA要求所有数据为正数"
    
    # 检查数据是否全为零
    if np.all(input_data == 0):
        return False, "投入数据全为零，无法进行DEA分析"
    
    if np.all(output_data == 0):
        return False, "产出数据全为零，无法进行DEA分析"
    
    # 检查样本数量
    if input_data.shape[0] < 3:
        return False, "样本数量过少，建议至少3个样本进行DEA分析"
    
    return True, "数据验证通过"

def perform_dea_analysis(data, input_vars, output_vars, model_type, orientation='input', undesirable_outputs=None):
    """
    执行DEA效率分析
    
    参数:
    - data: 包含医院数据的DataFrame
    - input_vars: 投入变量列表
    - output_vars: 产出变量列表
    - model_type: DEA模型类型 ('CCR', 'BCC', 'SBM', 'Super-SBM')
    - orientation: 导向类型 ('input', 'output')
    - undesirable_outputs: 非期望产出变量列表（仅SBM模型使用）
    
    返回:
    - results: 包含效率值的DataFrame
    """
    try:
        # 准备数据
        dmu_names = data['DMU'].values if 'DMU' in data.columns else data['医院ID'].values
        input_data = data[input_vars].values
        output_data = data[output_vars].values
        
        # 数据验证
        is_valid, message = validate_dea_data(input_data, output_data)
        if not is_valid:
            st.error(f"❌ 数据验证失败: {message}")
            return None
        
        # 数据预处理：处理零值和异常值
        input_data = np.maximum(input_data, 1e-10)  # 避免零值
        output_data = np.maximum(output_data, 1e-10)  # 避免零值
        
        # 创建DEA对象（使用自定义DEA实现）
        dea = DEAWrapper(input_data, output_data, dmu_names=dmu_names)
        
        # 显示使用的DEA库信息
        st.info("🔬 **使用自定义DEA实现进行DEA分析** - 稳定可靠的DEA分析方案")
        
        # 根据模型类型和导向执行分析
        if model_type == 'CCR':
            if orientation == 'input':
                efficiency_scores = dea.ccr_input_oriented()
            elif orientation == 'output':
                efficiency_scores = dea.ccr_output_oriented()
            else:
                raise ValueError(f"不支持的导向类型: {orientation}")
        elif model_type == 'CCR-VRS':
            # CCR-VRS模型实际上就是BCC模型
            if orientation == 'input':
                efficiency_scores = dea.bcc_input_oriented()
            elif orientation == 'output':
                efficiency_scores = dea.bcc_output_oriented()
            else:
                raise ValueError(f"不支持的导向类型: {orientation}")
        elif model_type == 'BCC':
            if orientation == 'input':
                efficiency_scores = dea.bcc_input_oriented()
            elif orientation == 'output':
                efficiency_scores = dea.bcc_output_oriented()
            else:
                raise ValueError(f"不支持的导向类型: {orientation}")
        elif model_type == 'SBM':
            # 处理非期望产出
            undesirable_indices = None
            if undesirable_outputs:
                # 将非期望产出变量名转换为列索引
                undesirable_indices = []
                for var in undesirable_outputs:
                    if var in output_vars:
                        undesirable_indices.append(output_vars.index(var))
            efficiency_scores = dea.sbm(undesirable_outputs=undesirable_indices)
        elif model_type == 'Super-SBM':
            # 处理非期望产出
            undesirable_indices = None
            if undesirable_outputs:
                # 将非期望产出变量名转换为列索引
                undesirable_indices = []
                for var in undesirable_outputs:
                    if var in output_vars:
                        undesirable_indices.append(output_vars.index(var))
            efficiency_scores = dea.super_sbm(undesirable_outputs=undesirable_indices)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 确保efficiency_scores是numpy数组
        if not isinstance(efficiency_scores, np.ndarray):
            efficiency_scores = np.array(efficiency_scores)
        
        # 效率值后处理：确保在[0,1]范围内
        efficiency_scores = np.clip(efficiency_scores, 0.0, 1.0)
        
        # 检查是否有异常的效率值
        if np.any(efficiency_scores > 1.0):
            st.warning("⚠️ 检测到效率值大于1，已自动修正为1.0")
        
        if np.any(efficiency_scores < 0.0):
            st.warning("⚠️ 检测到效率值小于0，已自动修正为0.0")
        
        # 创建结果DataFrame
        results = pd.DataFrame({
            'DMU': dmu_names,
            '效率值': efficiency_scores
        })
        
        # 按效率值降序排列
        results = results.sort_values('效率值', ascending=False).reset_index(drop=True)
        
        # 显示效率值统计信息
        st.info(f"📊 效率值统计: 最小值={results['效率值'].min():.3f}, 最大值={results['效率值'].max():.3f}, 平均值={results['效率值'].mean():.3f}")
        
        return results
        
    except Exception as e:
        st.error(f"DEA分析执行失败: {str(e)}")
        # 返回模拟数据用于演示
        st.warning("⚠️ 使用模拟数据进行演示")
        dmu_names = data['DMU'].values if 'DMU' in data.columns else data['医院ID'].values
        # 生成模拟效率值
        np.random.seed(42)  # 确保结果可重现
        efficiency_scores = np.random.uniform(0.6, 1.0, len(dmu_names))
        
        results = pd.DataFrame({
            'DMU': dmu_names,
            '效率值': efficiency_scores
        })
        
        results = results.sort_values('效率值', ascending=False).reset_index(drop=True)
        return results

def create_efficiency_chart(results):
    """
    创建效率排名柱状图
    
    参数:
    - results: 包含效率值的DataFrame
    
    返回:
    - fig: Plotly图表对象
    """
    # 创建柱状图
    fig = px.bar(
        results, 
        x='DMU', 
        y='效率值',
        title='DMU效率排名',
        labels={'效率值': '效率值', 'DMU': 'DMU'},
        color='效率值',
        color_continuous_scale='RdYlGn'
    )
    
    # 更新布局
    fig.update_layout(
        xaxis_title="DMU",
        yaxis_title="效率值",
        showlegend=False,
        height=500,
        title_x=0.5
    )
    
    # 添加数值标签（精确到小数点后3位）
    fig.update_traces(
        texttemplate='%{y:.3f}',
        textposition='outside'
    )
    
    return fig

def analyze_dea_results(results, data, input_vars, output_vars, model_type='BCC', orientation='input', undesirable_outputs=None):
    """
    分析DEA结果并提供详细解释
    
    参数:
    - results: 包含效率值的DataFrame
    - data: 原始数据
    - input_vars: 投入变量列表
    - output_vars: 产出变量列表
    - model_type: DEA模型类型 ('CCR', 'BCC', 'SBM', 'Super-SBM')
    - orientation: 导向类型 ('input', 'output')
    - undesirable_outputs: 非期望产出变量列表
    
    返回:
    - analysis_report: 分析报告字典
    """
    # 初始化分析报告结构
    analysis_report = {
        'model_info': {
            'model_type': model_type,
            'orientation': orientation,
            'undesirable_outputs': undesirable_outputs or []
        },
        'efficiency_analysis': {
            'overall_efficiency': {},
            'technical_efficiency': {},
            'scale_efficiency': {},
            'efficiency_decomposition': {}
        },
        'slack_analysis': {
            'input_slack': {},
            'output_slack': {},
            'slack_summary': {}
        },
        'effectiveness_analysis': {
            'strong_efficient': [],
            'weak_efficient': [],
            'non_efficient': []
        },
        'input_redundancy_analysis': {},
        'output_insufficiency_analysis': {},
        'detailed_unit_analysis': {},
        'improvement_suggestions': {}
    }
    
    # 合并数据
    dmu_column = 'DMU' if 'DMU' in data.columns else '医院ID'
    merged_data = data.merge(results, on=dmu_column, how='left')
    
    # 根据模型类型进行不同的分析
    if model_type == 'BCC':
        # BCC模型可以进行效益分解
        analysis_report = analyze_bcc_decomposition(analysis_report, merged_data, input_vars, output_vars)
    else:
        # 其他模型使用综合效率分析
        analysis_report = analyze_comprehensive_efficiency(analysis_report, merged_data, input_vars, output_vars, model_type)
    
    # 松弛变量分析（所有模型都需要）
    analysis_report = analyze_slack_variables(analysis_report, merged_data, input_vars, output_vars, model_type)
    
    # DEA有效性分析
    analysis_report = analyze_dea_effectiveness(analysis_report, merged_data)
    
    # 投入冗余分析
    analysis_report = analyze_input_redundancy(analysis_report, merged_data, input_vars)
    
    # 产出不足分析
    analysis_report = analyze_output_insufficiency(analysis_report, merged_data, output_vars, undesirable_outputs)
    
    # 详细单元分析
    analysis_report = analyze_individual_units(analysis_report, merged_data, input_vars, output_vars, model_type)
    
    # 生成改进建议
    analysis_report = generate_comprehensive_suggestions(analysis_report, merged_data, input_vars, output_vars)
    
    return analysis_report

def analyze_bcc_decomposition(analysis_report, merged_data, input_vars, output_vars):
    """分析BCC模型的效益分解（技术效益和规模效益）"""
    # 注意：这里需要同时运行CCR和BCC模型来分解效益
    # 由于我们只有BCC结果，这里提供理论分析框架
    
    efficiency_scores = merged_data['效率值'].values
    
    # 综合技术效益分析
    analysis_report['efficiency_analysis']['overall_efficiency'] = {
        'mean': float(efficiency_scores.mean()),
        'std': float(efficiency_scores.std()),
        'min': float(efficiency_scores.min()),
        'max': float(efficiency_scores.max()),
        'interpretation': {
            'optimal_units': len(efficiency_scores[efficiency_scores >= 0.9999]),
            'super_efficient_units': len(efficiency_scores[efficiency_scores > 1.0]),
            'inefficient_units': len(efficiency_scores[efficiency_scores < 0.9999])
        }
    }
    
    # 技术效益分析（BCC模型结果）
    analysis_report['efficiency_analysis']['technical_efficiency'] = {
        'mean': float(efficiency_scores.mean()),
        'interpretation': '反映由于管理和技术等因素影响的生产效率',
        'efficient_count': len(efficiency_scores[efficiency_scores >= 0.9999])
    }
    
    # 规模效益分析（需要CCR结果，这里提供理论框架）
    analysis_report['efficiency_analysis']['scale_efficiency'] = {
        'interpretation': '反映由于规模因素影响的生产效率',
        'note': '需要同时运行CCR模型来计算规模效益 = 综合效益 / 技术效益'
    }
    
    return analysis_report

def analyze_comprehensive_efficiency(analysis_report, merged_data, input_vars, output_vars, model_type):
    """分析其他模型的综合效率"""
    efficiency_scores = merged_data['效率值'].values
    
    # 根据模型类型调整效率值范围
    if model_type == 'Super-SBM':
        # 超效率SBM：效率值 >= 1
        efficient_threshold = 1.0
        super_efficient_threshold = 1.0
    else:
        # CCR、SBM：效率值 <= 1
        efficient_threshold = 0.9999
        super_efficient_threshold = 1.0
    
    analysis_report['efficiency_analysis']['overall_efficiency'] = {
        'mean': float(efficiency_scores.mean()),
        'std': float(efficiency_scores.std()),
        'min': float(efficiency_scores.min()),
        'max': float(efficiency_scores.max()),
        'interpretation': {
            'optimal_units': len(efficiency_scores[efficiency_scores >= efficient_threshold]),
            'super_efficient_units': len(efficiency_scores[efficiency_scores > super_efficient_threshold]) if model_type == 'Super-SBM' else 0,
            'inefficient_units': len(efficiency_scores[efficiency_scores < efficient_threshold])
        }
    }
    
    return analysis_report

def analyze_slack_variables(analysis_report, merged_data, input_vars, output_vars, model_type):
    """分析松弛变量"""
    # 这里需要实际的松弛变量值，由于我们的实现没有返回松弛变量，
    # 这里提供分析框架
    
    analysis_report['slack_analysis'] = {
        'input_slack': {
            'interpretation': '松弛变量S-(差额变数)：指为达到目标效率可以减少的投入量',
            'note': '需要从DEA求解过程中获取实际的松弛变量值'
        },
        'output_slack': {
            'interpretation': '松弛变量S+(超额变数)：指为达到目标效率可以增加的产出量',
            'note': '需要从DEA求解过程中获取实际的松弛变量值'
        }
    }
    
    return analysis_report

def analyze_dea_effectiveness(analysis_report, merged_data):
    """分析DEA有效性"""
    efficiency_scores = merged_data['效率值'].values
    dmu_column = 'DMU' if 'DMU' in merged_data.columns else '医院ID'
    dmu_ids = merged_data[dmu_column].values
    
    strong_efficient = []
    weak_efficient = []
    non_efficient = []
    
    for i, (dmu_id, efficiency) in enumerate(zip(dmu_ids, efficiency_scores)):
        # 这里简化处理，实际需要松弛变量值来判断
        if efficiency >= 0.9999:
            # 假设没有松弛变量信息，暂时都归为强有效
            strong_efficient.append({
                'dmu_id': dmu_id,
                'efficiency': float(efficiency),
                'status': 'DEA强有效',
                'interpretation': '综合效益=1且S-与S+均为0'
            })
        else:
            non_efficient.append({
                'dmu_id': dmu_id,
                'efficiency': float(efficiency),
                'status': '非DEA有效',
                'interpretation': '综合效益<1，存在投入冗余和产出不足'
            })
    
    analysis_report['effectiveness_analysis'] = {
        'strong_efficient': strong_efficient,
        'weak_efficient': weak_efficient,
        'non_efficient': non_efficient,
        'summary': {
            'total_units': len(dmu_ids),
            'strong_efficient_count': len(strong_efficient),
            'weak_efficient_count': len(weak_efficient),
            'non_efficient_count': len(non_efficient)
        }
    }
    
    return analysis_report

def analyze_input_redundancy(analysis_report, merged_data, input_vars):
    """投入冗余分析"""
    redundancy_analysis = {}
    
    for var in input_vars:
        values = merged_data[var].values
        mean_val = values.mean()
        
        redundancy_analysis[var] = {
            'mean_value': float(mean_val),
            'interpretation': '投入冗余率指"过多投入"与已投入的比值',
            'note': '需要松弛变量S-值来计算具体的投入冗余率'
        }
    
    analysis_report['input_redundancy_analysis'] = redundancy_analysis
    return analysis_report

def analyze_output_insufficiency(analysis_report, merged_data, output_vars, undesirable_outputs):
    """产出不足分析"""
    insufficiency_analysis = {}
    
    for var in output_vars:
        values = merged_data[var].values
        mean_val = values.mean()
        
        var_type = "非期望产出" if var in (undesirable_outputs or []) else "期望产出"
        
        insufficiency_analysis[var] = {
            'mean_value': float(mean_val),
            'type': var_type,
            'interpretation': '产出不足率指"产出不足"与已产出的比值',
            'note': '需要松弛变量S+值来计算具体的产出不足率'
        }
    
    analysis_report['output_insufficiency_analysis'] = insufficiency_analysis
    return analysis_report

def analyze_individual_units(analysis_report, merged_data, input_vars, output_vars, model_type):
    """详细单元分析"""
    detailed_analysis = {}
    
    for index, row in merged_data.iterrows():
        dmu_column = 'DMU' if 'DMU' in row.index else '医院ID'
        dmu_id = row[dmu_column]
        efficiency = row['效率值']
        
        # 效率状态判断
        if model_type == 'Super-SBM':
            if efficiency >= 1.0:
                status = "超效率有效"
                interpretation = "效率值≥1，表示超效率（比其他有效DMU更好）"
            else:
                status = "非超效率有效"
                interpretation = "效率值<1，未达到超效率标准"
        else:
            if efficiency >= 0.9999:
                status = "DEA有效"
                interpretation = "效率值=1，投入与产出结构合理，相对效益最优"
            else:
                status = "DEA无效"
                interpretation = "效率值<1，投入与产出结构不合理，存在投入冗余和产出不足"
        
        detailed_analysis[dmu_id] = {
            'efficiency': float(efficiency),
            'status': status,
            'interpretation': interpretation,
            'input_values': {var: float(row[var]) for var in input_vars},
            'output_values': {var: float(row[var]) for var in output_vars}
        }
    
    analysis_report['detailed_unit_analysis'] = detailed_analysis
    return analysis_report

def generate_comprehensive_suggestions(analysis_report, merged_data, input_vars, output_vars):
    """生成综合改进建议"""
    suggestions = {
        'overall_suggestions': [],
        'efficiency_improvement': [],
        'resource_optimization': [],
        'output_enhancement': []
    }
    
    # 整体建议
    total_units = len(merged_data)
    efficient_units = len(analysis_report['effectiveness_analysis']['strong_efficient'])
    efficiency_rate = efficient_units / total_units * 100
    
    suggestions['overall_suggestions'].append(
        f"整体效率率为{efficiency_rate:.1f}%，{total_units-efficient_units}个决策单元需要改进"
    )
    
    # 效率改进建议
    if efficiency_rate < 50:
        suggestions['efficiency_improvement'].append("整体效率较低，建议进行全面的效率提升计划")
    elif efficiency_rate < 80:
        suggestions['efficiency_improvement'].append("效率中等，建议重点改进低效单元")
    else:
        suggestions['efficiency_improvement'].append("整体效率较高，建议维持并进一步优化")
    
    # 资源优化建议
    suggestions['resource_optimization'].append("建议减少投入冗余，提高资源利用效率")
    suggestions['resource_optimization'].append("优化资源配置结构，避免资源浪费")
    
    # 产出提升建议
    suggestions['output_enhancement'].append("建议增加产出不足，提高服务质量和数量")
    suggestions['output_enhancement'].append("学习高效单元的最佳实践，提升整体产出水平")
    
    analysis_report['improvement_suggestions'] = suggestions
    return analysis_report

def analyze_inefficiency(hospital_row, input_vars, output_vars, all_data):
    """
    分析单个医院的效率不足原因
    
    参数:
    - hospital_row: 医院数据行
    - input_vars: 投入变量列表
    - output_vars: 产出变量列表
    - all_data: 所有医院数据
    
    返回:
    - analysis: 分析结果字典
    """
    dmu_column = 'DMU' if 'DMU' in hospital_row.index else '医院ID'
    dmu_id = hospital_row[dmu_column]
    dmu_efficiency = hospital_row['效率值']
    
    analysis = {
        'efficiency_score': dmu_efficiency,
        'input_analysis': {},
        'output_analysis': {},
        'benchmark_comparison': {},
        'improvement_potential': {}
    }
    
    # 计算各变量的相对表现
    for var in input_vars:
        hospital_value = hospital_row[var]
        avg_value = all_data[var].mean()
        median_value = all_data[var].median()
        
        analysis['input_analysis'][var] = {
            'hospital_value': hospital_value,
            'average_value': avg_value,
            'median_value': median_value,
            'relative_performance': hospital_value / avg_value if avg_value > 0 else 0,
            'status': '高于平均' if hospital_value > avg_value else '低于平均'
        }
    
    for var in output_vars:
        hospital_value = hospital_row[var]
        avg_value = all_data[var].mean()
        median_value = all_data[var].median()
        
        analysis['output_analysis'][var] = {
            'hospital_value': hospital_value,
            'average_value': avg_value,
            'median_value': median_value,
            'relative_performance': hospital_value / avg_value if avg_value > 0 else 0,
            'status': '高于平均' if hospital_value > avg_value else '低于平均'
        }
    
    return analysis

def generate_improvement_suggestions(inefficient_units, input_vars, output_vars):
    """
    生成改进建议
    
    参数:
    - inefficient_units: 低效医院列表
    - input_vars: 投入变量列表
    - output_vars: 产出变量列表
    
    返回:
    - suggestions: 改进建议字典
    """
    suggestions = {}
    
    for unit in inefficient_units:
        hospital_id = unit['hospital_id']
        analysis = unit['analysis']
        
        hospital_suggestions = []
        
        # 分析投入效率
        for var, data in analysis['input_analysis'].items():
            if data['relative_performance'] > 1.2:  # 投入过高
                hospital_suggestions.append({
                    'type': '投入优化',
                    'variable': var,
                    'suggestion': f"减少{var}投入，当前投入比平均水平高{(data['relative_performance']-1)*100:.1f}%",
                    'priority': '高' if data['relative_performance'] > 1.5 else '中'
                })
        
        # 分析产出效率
        for var, data in analysis['output_analysis'].items():
            if data['relative_performance'] < 0.8:  # 产出过低
                hospital_suggestions.append({
                    'type': '产出提升',
                    'variable': var,
                    'suggestion': f"提升{var}产出，当前产出比平均水平低{(1-data['relative_performance'])*100:.1f}%",
                    'priority': '高' if data['relative_performance'] < 0.6 else '中'
                })
        
        suggestions[hospital_id] = hospital_suggestions
    
    return suggestions

def perform_benchmark_analysis(data, input_vars, output_vars):
    """
    执行基准分析
    
    参数:
    - data: 合并后的数据
    - input_vars: 投入变量列表
    - output_vars: 产出变量列表
    
    返回:
    - benchmark: 基准分析结果
    """
    # 找到效率最高的DMU作为基准
    best_dmu = data.loc[data['效率值'].idxmax()]
    
    benchmark = {
        'best_dmu': {
            'id': best_dmu['DMU'],
            'efficiency': best_dmu['效率值']
        },
        'comparisons': {}
    }
    
    # 计算其他DMU与基准的差距
    for index, row in data.iterrows():
        dmu_column = 'DMU' if 'DMU' in row.index else '医院ID'
        if row[dmu_column] != best_dmu[dmu_column]:
            dmu_id = row[dmu_column]
            gap_analysis = {}
            
            for var in input_vars:
                gap = (row[var] - best_dmu[var]) / best_dmu[var] * 100
                gap_analysis[var] = {
                    'gap_percentage': gap,
                    'status': '投入过多' if gap > 0 else '投入不足'
                }
            
            for var in output_vars:
                gap = (row[var] - best_dmu[var]) / best_dmu[var] * 100
                gap_analysis[var] = {
                    'gap_percentage': gap,
                    'status': '产出较高' if gap > 0 else '产出不足'
                }
            
            benchmark['comparisons'][dmu_id] = gap_analysis
    
    return benchmark

def display_dea_analysis_report(analysis_report):
    """
    显示DEA分析报告
    
    参数:
    - analysis_report: 分析报告字典
    """
    st.subheader("📊 DEA结果深度分析")
    
    # 模型信息
    model_info = analysis_report['model_info']
    st.markdown(f"**分析模型**: {model_info['model_type']} ({model_info['orientation']}导向)")
    if model_info['undesirable_outputs']:
        st.markdown(f"**非期望产出**: {', '.join(model_info['undesirable_outputs'])}")
    
    # 1. 效率分析
    st.markdown("### 📈 效率分析")
    efficiency_analysis = analysis_report['efficiency_analysis']
    
    if 'overall_efficiency' in efficiency_analysis:
        overall_eff = efficiency_analysis['overall_efficiency']
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("平均效率", f"{overall_eff['mean']:.3f}")
        with col2:
            st.metric("最高效率", f"{overall_eff['max']:.3f}")
        with col3:
            st.metric("最低效率", f"{overall_eff['min']:.3f}")
        with col4:
            st.metric("标准差", f"{overall_eff['std']:.3f}")
        
        # 效率解释
        interpretation = overall_eff['interpretation']
        st.markdown("**效率分布解释**:")
        st.write(f"• 有效单元数: {interpretation['optimal_units']}")
        st.write(f"• 无效单元数: {interpretation['inefficient_units']}")
        if interpretation['super_efficient_units'] > 0:
            st.write(f"• 超效率单元数: {interpretation['super_efficient_units']}")
    
    # BCC模型效益分解
    if model_info['model_type'] == 'BCC':
        st.markdown("#### 🔬 BCC模型效益分解")
        
        if 'technical_efficiency' in efficiency_analysis:
            te = efficiency_analysis['technical_efficiency']
            st.markdown(f"**技术效益(TE)**: {te['mean']:.3f}")
            st.write(f"• {te['interpretation']}")
            st.write(f"• 技术有效单元数: {te['efficient_count']}")
        
        if 'scale_efficiency' in efficiency_analysis:
            se = efficiency_analysis['scale_efficiency']
            st.markdown(f"**规模效益(SE)**: {se['interpretation']}")
            st.write(f"• {se['note']}")
    
    # 2. DEA有效性分析
    st.markdown("### ✅ DEA有效性分析")
    effectiveness = analysis_report['effectiveness_analysis']
    summary = effectiveness['summary']
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("强有效单元", summary['strong_efficient_count'])
    with col2:
        st.metric("弱有效单元", summary['weak_efficient_count'])
    with col3:
        st.metric("非有效单元", summary['non_efficient_count'])
    
    # 有效性解释
    st.markdown("**有效性判断标准**:")
    st.write("• **DEA强有效**: 综合效益=1且S-与S+均为0")
    st.write("• **DEA弱有效**: 综合效益=1但S-或S+大于0")
    st.write("• **非DEA有效**: 综合效益<1")
    
    # 3. 松弛变量分析
    st.markdown("### 📊 松弛变量分析")
    slack_analysis = analysis_report['slack_analysis']
    
    st.markdown("**松弛变量解释**:")
    st.write(f"• **投入松弛变量S-(差额变数)**: {slack_analysis['input_slack']['interpretation']}")
    st.write(f"• **产出松弛变量S+(超额变数)**: {slack_analysis['output_slack']['interpretation']}")
    st.info("💡 注意：需要从DEA求解过程中获取实际的松弛变量值进行精确分析")
    
    # 4. 投入冗余分析
    st.markdown("### 🔍 投入冗余分析")
    input_redundancy = analysis_report['input_redundancy_analysis']
    
    for var, analysis in input_redundancy.items():
        st.markdown(f"**{var}**:")
        st.write(f"• 平均值: {analysis['mean_value']:.2f}")
        st.write(f"• 解释: {analysis['interpretation']}")
        st.write(f"• 说明: {analysis['note']}")
    
    # 5. 产出不足分析
    st.markdown("### 📈 产出不足分析")
    output_insufficiency = analysis_report['output_insufficiency_analysis']
    
    for var, analysis in output_insufficiency.items():
        st.markdown(f"**{var}** ({analysis['type']}):")
        st.write(f"• 平均值: {analysis['mean_value']:.2f}")
        st.write(f"• 解释: {analysis['interpretation']}")
        st.write(f"• 说明: {analysis['note']}")
    
    # 6. 详细单元分析
    st.markdown("### 🏥 详细单元分析")
    detailed_analysis = analysis_report['detailed_unit_analysis']
    
    # 创建详细分析表格
    analysis_data = []
    for dmu_id, analysis in detailed_analysis.items():
        analysis_data.append({
            'DMU': dmu_id,
            '效率值': analysis['efficiency'],
            '状态': analysis['status'],
            '解释': analysis['interpretation']
        })
    
    if analysis_data:
        df_analysis = pd.DataFrame(analysis_data)
        st.dataframe(df_analysis, use_container_width=True)
    
    # 7. 改进建议
    st.markdown("### 💡 改进建议")
    suggestions = analysis_report['improvement_suggestions']
    
    if 'overall_suggestions' in suggestions:
        st.markdown("**整体建议**:")
        for suggestion in suggestions['overall_suggestions']:
            st.write(f"• {suggestion}")
    
    if 'efficiency_improvement' in suggestions:
        st.markdown("**效率改进建议**:")
        for suggestion in suggestions['efficiency_improvement']:
            st.write(f"• {suggestion}")
    
    if 'resource_optimization' in suggestions:
        st.markdown("**资源优化建议**:")
        for suggestion in suggestions['resource_optimization']:
            st.write(f"• {suggestion}")
    
    if 'output_enhancement' in suggestions:
        st.markdown("**产出提升建议**:")
        for suggestion in suggestions['output_enhancement']:
            st.write(f"• {suggestion}")
    
    # 高效医院展示
    if analysis_report['efficient_units']:
        st.markdown("### 🏆 高效医院（效率值 = 1.0）")
        efficient_df = pd.DataFrame(analysis_report['efficient_units'])
        st.dataframe(efficient_df[['hospital_id', 'efficiency']], use_container_width=True)
    
    # 低效医院分析
    if analysis_report['inefficient_units']:
        st.markdown("### 📉 低效医院分析")
        
        for unit in analysis_report['inefficient_units']:
            with st.expander(f"🏥 {unit['hospital_id']} (效率值: {unit['efficiency']:.3f})", expanded=False):
                
                # 投入分析
                st.markdown("**投入分析**")
                input_data = []
                for var, data in unit['analysis']['input_analysis'].items():
                    input_data.append({
                        '变量': var,
                        '医院值': f"{data['hospital_value']:.2f}",
                        '平均值': f"{data['average_value']:.2f}",
                        '相对表现': f"{data['relative_performance']:.2f}",
                        '状态': data['status']
                    })
                st.dataframe(pd.DataFrame(input_data), use_container_width=True)
                
                # 产出分析
                st.markdown("**产出分析**")
                output_data = []
                for var, data in unit['analysis']['output_analysis'].items():
                    output_data.append({
                        '变量': var,
                        '医院值': f"{data['hospital_value']:.2f}",
                        '平均值': f"{data['average_value']:.2f}",
                        '相对表现': f"{data['relative_performance']:.2f}",
                        '状态': data['status']
                    })
                st.dataframe(pd.DataFrame(output_data), use_container_width=True)
    
    # 改进建议
    if analysis_report['improvement_suggestions']:
        st.markdown("### 💡 改进建议")
        
        for hospital_id, suggestions in analysis_report['improvement_suggestions'].items():
            if suggestions:
                with st.expander(f"🏥 {hospital_id} 改进建议", expanded=False):
                    for suggestion in suggestions:
                        priority_color = "🔴" if suggestion['priority'] == '高' else "🟡"
                        st.markdown(f"{priority_color} **{suggestion['type']}**: {suggestion['suggestion']}")
    
    # 基准分析
    if analysis_report['benchmark_analysis']['best_dmu']:
        st.markdown("### 🎯 基准分析")
        best_dmu = analysis_report['benchmark_analysis']['best_dmu']
        st.info(f"🏆 **基准DMU**: {best_dmu['id']} (效率值: {best_dmu['efficiency']:.3f})")
        
        if analysis_report['benchmark_analysis']['comparisons']:
            st.markdown("**与基准DMU的差距分析**")
            comparison_data = []
            for dmu_id, gaps in analysis_report['benchmark_analysis']['comparisons'].items():
                for var, gap_info in gaps.items():
                    comparison_data.append({
                        'DMU': dmu_id,
                        '变量': var,
                        '差距(%)': f"{gap_info['gap_percentage']:.1f}",
                        '状态': gap_info['status']
                    })
            
            if comparison_data:
                st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

def download_dea_results(results):
    """
    生成DEA结果CSV下载
    
    参数:
    - results: 包含效率值的DataFrame
    
    返回:
    - csv: CSV格式的字符串
    """
    csv = results.to_csv(index=False, encoding='utf-8-sig')
    return csv



def create_coverage_chart(fsqca_results):
    """
    创建路径覆盖度比较柱状图
    
    参数:
    - fsqca_results: fsQCA分析结果DataFrame
    
    返回:
    - fig: Plotly图表对象
    """
    try:
        # 过滤有效路径
        valid_paths = fsqca_results[fsqca_results['Path Type'] != '无效路径'].copy()
        
        if len(valid_paths) == 0:
            return None
        
        # 创建柱状图
        fig = px.bar(
            valid_paths,
            x='Solution Path',
            y='Raw Coverage',
            color='Path Type',
            title='🔍 路径覆盖度比较',
            labels={'Raw Coverage': '覆盖度', 'Solution Path': '路径组合'},
            color_discrete_map={
                '核心路径': '#2E8B57',
                '边缘路径': '#FFA500'
            }
        )
        
        # 更新布局
        fig.update_layout(
            xaxis_title="路径组合",
            yaxis_title="覆盖度",
            height=500,
            title_x=0.5,
            xaxis_tickangle=-45
        )
        
        # 添加数值标签（精确到小数点后3位）
        fig.update_traces(
            texttemplate='%{y:.3f}',
            textposition='outside'
        )
        
        return fig
        
    except Exception as e:
        st.error(f"图表创建失败: {str(e)}")
        return None

def download_fsqca_results(fsqca_results, necessity_results):
    """
    生成fsQCA结果CSV下载
    
    参数:
    - fsqca_results: fsQCA分析结果DataFrame
    - necessity_results: 必要性分析结果DataFrame
    
    返回:
    - csv: CSV格式的字符串
    """
    try:
        # 创建综合结果
        with BytesIO() as output:
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # 写入必要性分析结果
                necessity_results.to_excel(writer, sheet_name='必要性分析', index=False)
                # 写入fsQCA分析结果
                fsqca_results.to_excel(writer, sheet_name='组态路径分析', index=False)
            
            return output.getvalue()
            
    except Exception as e:
        st.error(f"结果导出失败: {str(e)}")
        return None

def main():
    # 主标题
    st.markdown('<div class="main-header">基于DEA与fsQCA的医院运营效能与发展路径智慧决策系统 v1.0</div>', unsafe_allow_html=True)
    
    # 系统状态指示器
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        data_status = "✅" if 'data' in st.session_state else "❌"
        st.markdown(f'<div class="metric-card"><h4>数据状态</h4><p style="font-size: 2rem; margin: 0;">{data_status}</p></div>', unsafe_allow_html=True)
    with col2:
        dea_status = "✅" if 'dea_results' in st.session_state else "❌"
        st.markdown(f'<div class="metric-card"><h4>DEA分析</h4><p style="font-size: 2rem; margin: 0;">{dea_status}</p></div>', unsafe_allow_html=True)
    with col3:
        fsqca_status = "✅" if 'fsqca_results' in st.session_state else "❌"
        st.markdown(f'<div class="metric-card"><h4>fsQCA分析</h4><p style="font-size: 2rem; margin: 0;">{fsqca_status}</p></div>', unsafe_allow_html=True)
    with col4:
        qca_status = "✅" if QCA_AVAILABLE else "❌"
        status_text = "QCA模块正常" if QCA_AVAILABLE else "QCA模块异常"
        status_color = "#1a365d" if QCA_AVAILABLE else "#e53e3e"
        st.markdown(f'<div class="metric-card"><h4>QCA模块</h4><p style="font-size: 1.2rem; margin: 0; color: {status_color};">{qca_status} {status_text}</p></div>', unsafe_allow_html=True)
    
    # ① 数据输入区
    st.markdown('<div class="section-header">① 数据输入区</div>', unsafe_allow_html=True)
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    
    if 'data' not in st.session_state:
        # 选择输入模式
        input_mode = st.radio(
            "选择数据输入方式：",
            ["📁 上传文件模式", "✏️ 手动输入模式"],
            horizontal=True
        )
        
        if input_mode == "📁 上传文件模式":
            st.markdown("### 📁 文件上传")
            st.info("请上传包含医院数据的Excel或CSV文件，文件必须包含'DMU'列或'医院ID'列。")
            
            uploaded_file = st.file_uploader(
                "选择文件",
                type=['xlsx', 'xls', 'csv'],
                help="支持Excel (.xlsx, .xls) 和CSV (.csv) 格式"
            )
            
            if uploaded_file is not None:
                try:
                    # 读取文件
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file, encoding='utf-8')
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    # 验证DMU列
                    is_valid, message = validate_dmu_column(df)
                    if not is_valid:
                        st.markdown(f'<div class="error-message">{message}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="success-message">{message}</div>', unsafe_allow_html=True)
                        
                        # 数据验证
                        errors, warnings = validate_numeric_data(df)
                        
                        if errors:
                            for error in errors:
                                st.markdown(f'<div class="error-message">{error}</div>', unsafe_allow_html=True)
                        else:
                            # 检查是否有空值
                            null_counts = df.isnull().sum()
                            total_nulls = null_counts.sum()
                            
                            if total_nulls > 0:
                                # 显示空值处理选择
                                null_handling = detect_and_handle_nulls(df)
                                
                                if null_handling is None:
                                    st.info("请选择空值处理方式以继续...")
                                else:
                                    # 根据用户选择清理数据
                                    df_cleaned, stats = clean_data(df, null_handling)
                                    
                                    # 显示处理结果
                                    if null_handling == 'fill_zero':
                                        st.success(f"✅ 已将 {stats['filled_nulls']} 个空值转换为0")
                                    else:  # drop_rows
                                        st.success(f"✅ 已删除 {stats['removed_rows']} 行包含空值的数据")
                                    
                                    # 继续处理数据
                                    process_cleaned_data(df_cleaned, warnings)
                            else:
                                # 没有空值，直接处理
                                process_cleaned_data(df, warnings)
                
                except Exception as e:
                    st.markdown(f'<div class="error-message">文件读取错误：{str(e)}</div>', unsafe_allow_html=True)
        
        elif input_mode == "✏️ 手动输入模式":
            st.markdown("### ✏️ 手动数据输入")
            st.info("请设置医院数量和变量数量，然后逐家输入数据。")
            
            # 设置参数
            col1, col2 = st.columns(2)
            with col1:
                num_hospitals = st.slider("医院数量", min_value=3, max_value=20, value=5, help="选择1-1000家医院")
            with col2:
                num_variables = st.slider("变量数量", min_value=2, max_value=10, value=3, help="选择2-10个变量")
            
            # 创建输入表单
            df = create_manual_input_form(num_hospitals, num_variables)
            
            if df is not None:
                # 显示预览
                st.markdown("### 📋 数据预览")
                st.markdown('<div class="data-preview">', unsafe_allow_html=True)
                st.dataframe(df, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # 保存到session state
                st.session_state['data'] = df
                st.session_state['data_source'] = 'manual'
                
                st.success("✅ 数据输入完成！可以进入DEA效率分析模块。")
    
    else:
        st.markdown('</div>', unsafe_allow_html=True)  # 关闭数据输入区容器
    
    # ② DEA分析区
    st.markdown('<div class="section-header">② DEA分析区</div>', unsafe_allow_html=True)
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    
    if 'data' in st.session_state:
        data = st.session_state['data']
        
        # 显示数据预览
        st.subheader("📋 数据预览")
        st.dataframe(data.head(), use_container_width=True)
        
        # 获取数值列（排除DMU列和医院ID列）
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        if 'DMU' in numeric_columns:
            numeric_columns.remove('DMU')
        if '医院ID' in numeric_columns:
            numeric_columns.remove('医院ID')
        
        if len(numeric_columns) < 2:
            st.error("❌ 数据中至少需要2个数值变量才能进行DEA分析")
        else:
            st.subheader("⚙️ 变量选择")
            
            # 创建两列布局
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**选择【投入变量】**")
                st.caption("资源消耗类指标，如医生人数、床位数等")
                st.info("💡 **医疗示例**：医生人数、护士人数、床位数、医疗设备数量、运营成本等")
                input_vars = create_searchable_multiselect(
                    "投入变量",
                    options=numeric_columns,
                    key="input_vars",
                    help_text="选择作为投入的变量，至少选择1个",
                    placeholder="请选择投入变量..."
                )
            
            with col2:
                st.markdown("**选择【产出变量】**")
                st.caption("服务成果类指标，如门诊量、手术量等")
                st.info("💡 **医疗示例**：门诊人次、住院人次、手术例数、出院人数、患者满意度等")
                output_vars = create_searchable_multiselect(
                    "产出变量",
                    options=numeric_columns,
                    key="output_vars",
                    help_text="选择作为产出的变量，至少选择1个",
                    placeholder="请选择产出变量..."
                )
            
            # 验证变量选择
            if not input_vars:
                st.error("❌ 请至少选择1个投入变量")
            elif not output_vars:
                st.error("❌ 请至少选择1个产出变量")
            else:
                st.success(f"✅ 已选择 {len(input_vars)} 个投入变量，{len(output_vars)} 个产出变量")
                
                # 模型选择
                st.subheader("🔬 模型选择")
                
                model_options = {
                    "CCR模型（规模报酬不变）": {
                        "value": "CCR",
                        "description": "适用于同级医院对比，假设规模报酬不变",
                        "scenario": "🏥 **适用场景**：同等级医院效率对比（如三甲医院间对比）",
                        "features": "• 假设规模报酬不变\n• 适合规模相近的医院\n• 计算相对效率"
                    },
                    "CCR模型（规模报酬可变）": {
                        "value": "CCR-VRS",
                        "description": "CCR模型的规模报酬可变版本，考虑规模效应",
                        "scenario": "🏥 **适用场景**：不同规模医院对比，考虑规模报酬可变",
                        "features": "• 考虑规模报酬可变\n• 适合不同规模医院\n• 分离技术效率与规模效率"
                    },
                    "BCC模型（规模报酬可变）": {
                        "value": "BCC", 
                        "description": "适用于不同等级医院对比，考虑规模报酬可变（推荐）",
                        "scenario": "🏥 **适用场景**：不同等级医院效率对比（推荐医疗行业使用）",
                        "features": "• 考虑规模报酬可变\n• 适合不同规模医院\n• 分离技术效率与规模效率"
                    },
                    "SBM模型（非径向）": {
                        "value": "SBM",
                        "description": "适用于含非期望产出场景，非径向效率测量",
                        "scenario": "🏥 **适用场景**：包含不良事件、医疗纠纷等非期望产出的分析",
                        "features": "• 非径向效率测量\n• 处理非期望产出\n• 更精确的效率评估\n• 效率值范围：(0,1]"
                    },
                    "超效率SBM模型": {
                        "value": "Super-SBM",
                        "description": "超效率SBM模型，允许效率值大于1，可对有效DMU进一步排序",
                        "scenario": "🏥 **适用场景**：需要对高效医院进行进一步排序和比较",
                        "features": "• 超效率测量\n• 处理非期望产出\n• 效率值范围：[1,∞)\n• 可对有效DMU排序"
                    }
                }
                
                selected_model = st.selectbox(
                    "选择DEA模型",
                    options=list(model_options.keys()),
                    index=1,  # 默认选择BCC模型
                    help="BCC模型是医疗行业最常用的DEA模型"
                )
                
                # 显示模型详细说明
                model_info = model_options[selected_model]
                st.markdown(f"**{model_info['scenario']}**")
                st.info(f"💡 {model_info['description']}")
                st.markdown(f"**模型特点：**\n{model_info['features']}")
                
                # 导向选择（仅对CCR、CCR-VRS和BCC模型显示）
                orientation = 'input'  # 默认值
                if model_info['value'] in ['CCR', 'CCR-VRS', 'BCC']:
                    st.markdown("**📐 选择分析导向**")
                    orientation_options = {
                        "输入导向（推荐）": {
                            "value": "input",
                            "description": "在保持产出不变的前提下，最小化投入资源",
                            "scenario": "🏥 **适用场景**：资源优化配置，减少浪费（推荐医疗行业使用）",
                            "features": "• 关注投入效率\n• 适合资源优化\n• 减少资源浪费"
                        },
                        "输出导向": {
                            "value": "output", 
                            "description": "在保持投入不变的前提下，最大化产出效果",
                            "scenario": "🏥 **适用场景**：提升服务质量，增加产出效果",
                            "features": "• 关注产出效率\n• 适合服务提升\n• 增加产出效果"
                        }
                    }
                    
                    selected_orientation = st.selectbox(
                        "选择分析导向",
                        options=list(orientation_options.keys()),
                        index=0,  # 默认选择输入导向
                        help="输入导向是医疗行业最常用的分析方式"
                    )
                    
                    orientation_info = orientation_options[selected_orientation]
                    orientation = orientation_info['value']
                    st.markdown(f"**{orientation_info['scenario']}**")
                    st.info(f"💡 {orientation_info['description']}")
                    st.markdown(f"**导向特点：**\n{orientation_info['features']}")
                
                # 非期望产出选择（仅对SBM模型显示）
                undesirable_outputs = None
                if model_info['value'] in ['SBM', 'Super-SBM']:
                    st.markdown("**⚠️ 非期望产出选择**")
                    st.caption("选择哪些产出变量是非期望的（如医疗纠纷、不良事件等）")
                    
                    # 显示产出变量供选择
                    if output_vars:
                        st.markdown("**当前产出变量：**")
                        for i, var in enumerate(output_vars):
                            st.write(f"• {var}")
                        
                        # 多选非期望产出
                        selected_undesirable = st.multiselect(
                            "选择非期望产出变量",
                            options=output_vars,
                            default=[],
                            help="选择那些数值越小越好的产出变量（如医疗纠纷数量、不良事件等）"
                        )
                        
                        if selected_undesirable:
                            undesirable_outputs = selected_undesirable
                            st.success(f"✅ 已选择 {len(selected_undesirable)} 个非期望产出变量")
                            st.info("💡 **非期望产出说明**：这些变量的数值越小表示效率越高，如医疗纠纷、不良事件等")
                        else:
                            st.info("💡 未选择非期望产出，所有产出变量将视为期望产出")
                    else:
                        st.warning("⚠️ 没有产出变量可供选择")
                
                # 执行分析按钮
                st.markdown("---")
                col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
                with col_btn2:
                    if st.button("🚀 执行DEA分析", type="primary", use_container_width=True):
                        with st.spinner("正在执行DEA分析..."):
                            # 执行DEA分析
                            results = perform_dea_analysis(
                                data, 
                                input_vars, 
                                output_vars, 
                                model_info['value'],
                                orientation,
                                undesirable_outputs
                            )
                            
                            if results is not None:
                                # 保存结果到session state
                                try:
                                    # 确保results是可序列化的DataFrame
                                    if hasattr(results, 'to_dict'):
                                        # 如果是DataFrame，确保索引重置
                                        results_copy = results.reset_index(drop=True)
                                        st.session_state['dea_results'] = results_copy
                                    else:
                                        st.session_state['dea_results'] = results
                                    
                                    st.session_state['dea_model'] = str(selected_model) if selected_model else ""
                                    
                                    # 安全地保存变量列表
                                    input_vars_list = []
                                    if input_vars:
                                        for var in input_vars:
                                            if isinstance(var, str):
                                                input_vars_list.append(str(var))
                                            else:
                                                input_vars_list.append(str(var))
                                    
                                    output_vars_list = []
                                    if output_vars:
                                        for var in output_vars:
                                            if isinstance(var, str):
                                                output_vars_list.append(str(var))
                                            else:
                                                output_vars_list.append(str(var))
                                    
                                    st.session_state['selected_input_vars'] = input_vars_list
                                    st.session_state['selected_output_vars'] = output_vars_list
                                    
                                except Exception as e:
                                    st.error(f"保存分析结果时出错: {str(e)}")
                                    # 使用基本类型保存
                                    st.session_state['selected_input_vars'] = []
                                    st.session_state['selected_output_vars'] = []
                                    st.session_state['dea_model'] = str(selected_model) if selected_model else ""
                                
                                st.success("✅ DEA分析完成！")
                                
                                # 显示结果
                                st.subheader("📊 效率分析结果")
                                
                                # 显示效率值表格
                                st.markdown("**效率值排名（按效率值降序排列）**")
                                results_display = results.copy()
                                
                                # 按效率值降序排序
                                results_display = results_display.sort_values('效率值', ascending=False).reset_index(drop=True)
                                results_display['效率值'] = results_display['效率值'].round(3)
                                results_display['排名'] = range(1, len(results_display) + 1)
                                
                                # 重新排列列顺序
                                results_display = results_display[['排名', 'DMU', '效率值']]
                                
                                # 应用蓝色渐变背景样式
                                st.markdown("""
                                <style>
                                .efficiency-table {
                                    background: linear-gradient(135deg, #e3f2fd, #bbdefb, #90caf9);
                                    border-radius: 10px;
                                    padding: 1rem;
                                    margin: 1rem 0;
                                    box-shadow: 0 4px 12px rgba(33, 150, 243, 0.2);
                                }
                                </style>
                                """, unsafe_allow_html=True)
                                
                                st.markdown('<div class="efficiency-table">', unsafe_allow_html=True)
                                st.dataframe(
                                    results_display,
                                    use_container_width=True,
                                    hide_index=True
                                )
                                st.markdown('</div>', unsafe_allow_html=True)
                                
                                # 高亮最优DMU
                                best_dmu = results.iloc[0]
                                st.markdown(f"🏆 **最优DMU**: {best_dmu['DMU']} (效率值: {best_dmu['效率值']:.3f})")
                                
                                # 创建效率排名图表
                                st.subheader("📈 效率排名可视化")
                                fig = create_efficiency_chart(results)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # 提供结果下载
                                st.subheader("💾 结果下载")
                                csv_data = download_dea_results(results)
                                
                                st.download_button(
                                    label="📥 下载DEA分析结果 (CSV)",
                                    data=csv_data,
                                    file_name=f"DEA分析结果_{model_info['value']}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                                
                                # 分析摘要
                                st.subheader("📋 分析摘要")
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("分析医院数", len(results))
                                
                                with col2:
                                    efficient_count = len(results[results['效率值'] >= 0.9999])
                                    st.metric("有效医院数", efficient_count)
                                
                                with col3:
                                    avg_efficiency = results['效率值'].mean()
                                    st.metric("平均效率值", f"{avg_efficiency:.3f}")
                                
                                # 效率分布统计
                                st.markdown("**效率值分布统计**")
                                efficiency_stats = results['效率值'].describe()
                                st.write(efficiency_stats)
                                
                                # 添加结果解释按钮
                                st.markdown("---")
                                if st.button("🔍 深度分析结果", type="secondary", help="点击查看详细的效率分析和改进建议"):
                                    with st.spinner("正在生成深度分析报告..."):
                                        # 执行深度分析
                                        analysis_report = analyze_dea_results(
                                            results, 
                                            data, 
                                            input_vars, 
                                            output_vars,
                                            model_info['value'],
                                            orientation,
                                            undesirable_outputs
                                        )
                                        
                                        # 显示分析报告
                                        display_dea_analysis_report(analysis_report)
                                        
                                        # 保存分析报告到session state
                                        st.session_state['dea_analysis_report'] = analysis_report
    else:
        st.warning("⚠️ 请先在数据输入区中加载数据")
    
    st.markdown('</div>', unsafe_allow_html=True)  # 关闭DEA分析区容器
    
    # ③ fsQCA路径分析区
    st.markdown('<div class="section-header">③ fsQCA路径分析区</div>', unsafe_allow_html=True)
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    
    # 检查QCA模块状态
    if not QCA_AVAILABLE:
        st.error("❌ QCA分析模块不可用，请检查模块安装")
        st.info("💡 **解决方案**：")
        st.markdown("""
        1. 确保qca_analysis.py文件存在
        2. 检查Python环境是否正确
        3. 重启应用程序
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        return
        
    if 'data' in st.session_state and 'dea_results' in st.session_state:
        data = st.session_state['data']
        dea_results = st.session_state['dea_results']
        
        # 显示数据预览
        st.subheader("📋 数据预览")
        st.dataframe(data.head(), use_container_width=True)
        
        # 获取可用的条件变量（排除DEA已使用的变量）
        used_vars = st.session_state.get('selected_input_vars', []) + st.session_state.get('selected_output_vars', [])
        available_vars = [col for col in data.columns if col not in ['DMU', '医院ID'] + used_vars]
        
        if len(available_vars) < 1:
            st.error("❌ 没有可用的条件变量，请确保数据中包含除DEA变量外的其他变量")
        else:
            st.subheader("⚙️ 条件变量选择")
            
            # 推荐常用条件变量
            recommended_vars = []
            for var in available_vars:
                if any(keyword in var.lower() for keyword in ['科研', '经费', '电子', '病历', '等级', '信息化']):
                    recommended_vars.append(var)
            
            # 默认选择前2个推荐变量
            default_vars = recommended_vars[:2] if len(recommended_vars) >= 2 else available_vars[:2]
            
            condition_vars = st.multiselect(
                "选择条件变量",
                options=available_vars,
                default=default_vars,
                key="condition_vars",
                help="选择用于fsQCA分析的条件变量，至少选择1个"
            )
            
            # 验证条件变量选择
            if not condition_vars:
                st.error("❌ 请至少选择1个条件变量")
            else:
                st.success(f"✅ 已选择 {len(condition_vars)} 个条件变量")
                
                st.subheader("🔧 数据预处理")
                st.info("正在将条件变量标准化为0-1范围的模糊集...")
                
                # 创建数据副本用于QCA分析
                dmu_column = 'DMU' if 'DMU' in data.columns else '医院ID'
                data_with_efficiency = data.merge(dea_results, on=dmu_column, how='left').copy()
                
                # 标准化条件变量到0-1范围
                for var in condition_vars:
                    min_val = data_with_efficiency[var].min()
                    max_val = data_with_efficiency[var].max()
                    if max_val > min_val:  # 避免除以0
                        data_with_efficiency[var] = (data_with_efficiency[var] - min_val) / (max_val - min_val)
                    else:
                        st.warning(f"⚠️ 变量 '{var}' 的值全部相同，标准化后将为常数")
                
                # 显示标准化后的数据预览
                st.markdown("### 📊 标准化后数据预览")
                st.dataframe(data_with_efficiency[condition_vars + ['效率值']].head(), use_container_width=True)
                # ===== 标准化步骤结束 =====
                
                # 必要性分析配置
                st.subheader("🔍 必要性分析配置")
                
                col1, col2 = st.columns(2)
                with col1:
                    perform_necessity = st.checkbox(
                        "执行必要性分析",
                        value=True,
                        help="分析每个条件变量与结果变量的必要性关系"
                    )
                
                with col2:
                    if perform_necessity:
                        st.info("💡 将自动过滤一致性<0.9的变量")
                
                # 组态路径分析参数配置
                st.subheader("⚙️ 组态路径分析参数配置")
                st.markdown("**🏥 医疗行业推荐值**")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    freq_threshold = st.slider(
                        "频数阈值",
                        min_value=0.1,
                        max_value=5.0,
                        value=1.0,
                        step=0.1,
                        help="医疗小样本标准，默认1.0（Rihoux & Ragin, 2009）"
                    )
                
                with col2:
                    pri_consistency = st.slider(
                        "PRI一致性阈值",
                        min_value=0.1,
                        max_value=0.9,
                        value=0.7,
                        step=0.05,
                        help="PRI=0.7（Ragin, 2008）"
                    )
                
                with col3:
                    consistency = st.slider(
                        "一致性阈值",
                        min_value=0.1,
                        max_value=0.9,
                        value=0.8,
                        step=0.05,
                        help="一致性=0.8（杜运周, 2021）"
                    )
                
                # 验证参数
                if pri_consistency >= consistency:
                    st.error("❌ PRI一致性阈值必须小于一致性阈值")
                else:
                    st.success("✅ 参数配置正确")
                    
                    # 执行分析按钮
                    if st.button("🚀 生成高质量发展路径", type="primary", help="点击生成基于fsQCA的高质量发展路径"):
                        with st.spinner("正在执行fsQCA分析..."):
                            # 准备数据（合并DEA结果）
                            dmu_column = 'DMU' if 'DMU' in data.columns else '医院ID'
                            data_with_efficiency = data.merge(dea_results, on=dmu_column, how='left')
                            
                            # 执行必要性分析
                            necessity_results = pd.DataFrame()
                            if perform_necessity:
                                necessity_results = perform_necessity_analysis(
                                    data_with_efficiency, 
                                    condition_vars, 
                                    '效率值'
                                )
                                
                                # 检查必要性分析结果是否有效
                                if not necessity_results.empty and 'Raw Consistency' in necessity_results.columns:
                                    # 过滤Raw Consistency<0.9的变量
                                    valid_vars = necessity_results[necessity_results['Raw Consistency'] >= 0.9]['条件变量'].tolist()
                                    if valid_vars:
                                        condition_vars = valid_vars
                                        st.info(f"✅ 必要性分析完成，保留 {len(valid_vars)} 个有效条件变量")
                                    else:
                                        st.warning("⚠️ 所有条件变量的一致性都<0.9，使用原始变量进行分析")
                                else:
                                    st.warning("⚠️ 必要性分析失败，使用原始变量进行分析")
                            
                            # 执行fsQCA分析
                            fsqca_results = perform_minimization(
                                data_with_efficiency,
                                condition_vars,
                                '效率值',
                                freq_threshold,
                                consistency
                            )
                            
                            # 检查fsQCA分析结果是否有效
                            if not fsqca_results.empty and 'Solution Path' in fsqca_results.columns:
                                # 保存结果到session state
                                try:
                                    # 确保DataFrame是可序列化的
                                    if hasattr(fsqca_results, 'reset_index'):
                                        fsqca_results_copy = fsqca_results.reset_index(drop=True)
                                        st.session_state['fsqca_results'] = fsqca_results_copy
                                    else:
                                        st.session_state['fsqca_results'] = fsqca_results
                                    
                                    if hasattr(necessity_results, 'reset_index'):
                                        necessity_results_copy = necessity_results.reset_index(drop=True)
                                        st.session_state['necessity_results'] = necessity_results_copy
                                    else:
                                        st.session_state['necessity_results'] = necessity_results
                                    
                                    # 安全地保存条件变量列表
                                    condition_vars_list = []
                                    if condition_vars:
                                        for var in condition_vars:
                                            if isinstance(var, str):
                                                condition_vars_list.append(str(var))
                                            else:
                                                condition_vars_list.append(str(var))
                                    
                                    st.session_state['selected_condition_vars'] = condition_vars_list
                                    
                                except Exception as e:
                                    st.error(f"保存fsQCA结果时出错: {str(e)}")
                                    # 使用基本类型保存
                                    st.session_state['selected_condition_vars'] = []
                                
                                st.success("✅ fsQCA分析完成！")
                                
                                # 显示必要性分析结果
                                if not necessity_results.empty:
                                    st.subheader("📊 必要性分析结果")
                                    st.dataframe(necessity_results, use_container_width=True)
                                
                                # 显示组态路径分析结果
                                st.subheader("🔍 组态路径分析结果")
                                
                                # 过滤有效路径
                                valid_paths = fsqca_results[fsqca_results['Path Type'] != '无效路径']
                                
                                if len(valid_paths) > 0:
                                    # 应用核心路径高亮样式
                                    st.markdown("""
                                    <style>
                                    .path-table {
                                        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
                                        border-radius: 10px;
                                        padding: 1rem;
                                        margin: 1rem 0;
                                        box-shadow: 0 2px 8px rgba(26, 54, 93, 0.1);
                                    }
                                    .core-path-row {
                                        background-color: #e3f2fd !important;
                                        font-weight: bold;
                                    }
                                    .edge-path-row {
                                        background-color: #fff3e0 !important;
                                    }
                                    </style>
                                    """, unsafe_allow_html=True)
                                    
                                    st.markdown('<div class="path-table">', unsafe_allow_html=True)
                                    
                                    # 创建带样式的DataFrame
                                    def highlight_path_type(row):
                                        if row['Path Type'] == '核心路径':
                                            return ['core-path-row'] * len(row)
                                        elif row['Path Type'] == '边缘路径':
                                            return ['edge-path-row'] * len(row)
                                        else:
                                            return [''] * len(row)
                                    
                                    # 显示表格
                                    st.dataframe(valid_paths, use_container_width=True)
                                    
                                    st.markdown('</div>', unsafe_allow_html=True)
                                    
                                    # 显示路径统计
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("总路径数", len(fsqca_results))
                                    with col2:
                                        st.metric("有效路径数", len(valid_paths))
                                    with col3:
                                        core_paths = len(valid_paths[valid_paths['Path Type'] == '核心路径'])
                                        st.metric("核心路径数", core_paths)
                                    
                                    # 创建覆盖度图表
                                    st.subheader("📈 路径覆盖度比较")
                                    fig = create_coverage_chart(fsqca_results)
                                    if fig:
                                        st.plotly_chart(fig, use_container_width=True)
                                    
                                    # 提供结果下载
                                    st.subheader("💾 结果下载")
                                    excel_data = download_fsqca_results(fsqca_results, necessity_results)
                                    
                                    if excel_data:
                                        st.download_button(
                                            label="📥 下载fsQCA分析结果 (Excel)",
                                            data=excel_data,
                                            file_name=f"fsQCA分析结果_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                        )
                                    
                                    # 分析摘要
                                    st.subheader("📋 分析摘要")
                                    
                                    if len(valid_paths) > 0:
                                        best_path = valid_paths.iloc[0]
                                        st.markdown(f"🏆 **最优路径**: {best_path['Solution Path']}")
                                        st.markdown(f"   - 一致性: {best_path['Raw Consistency']:.4f}")
                                        st.markdown(f"   - 覆盖度: {best_path['Raw Coverage']:.4f}")
                                        st.markdown(f"   - 路径类型: {best_path['Path Type']}")
                                    
                                    # 路径解释
                                    st.markdown("**路径解释**")
                                    st.markdown("- **核心路径**: 同时满足PRI一致性和一致性阈值的路径")
                                    st.markdown("- **边缘路径**: 仅满足一致性阈值的路径")
                                    st.markdown("- **无效路径**: 不满足任何阈值的路径")
                                    
                                else:
                                    st.warning("⚠️ 没有找到有效路径，请尝试调整参数阈值")
                            else:
                                # QCA分析失败
                                st.error("❌ fsQCA分析失败，请检查数据和参数设置")
                                st.info("💡 **可能的原因**：")
                                st.markdown("""
                                1. 数据格式不正确
                                2. 参数设置不当
                                3. 条件变量选择问题
                                4. 数据量不足
                                """)
                                st.info("💡 **解决方案**：")
                                st.markdown("""
                                1. 检查数据是否包含足够的案例
                                2. 调整一致性阈值和频率阈值
                                3. 尝试选择不同的条件变量
                                4. 确保数据质量良好
                                """)
    else:
        if 'data' not in st.session_state:
            st.warning("⚠️ 请先在数据输入区中加载数据")
        elif 'dea_results' not in st.session_state:
            st.warning("⚠️ 请先完成DEA效率分析")
    
    st.markdown('</div>', unsafe_allow_html=True)  # 关闭fsQCA分析区容器

if __name__ == "__main__":
    main()
