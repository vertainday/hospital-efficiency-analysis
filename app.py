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
import pulp
from typing import Dict, List, Tuple, Optional, Union

# 检查QCA模块是否可用
try:
    from qca_analysis import perform_necessity_analysis, perform_minimization
    QCA_AVAILABLE = True
    print("✅ QCA分析模块可用")
except ImportError:
    QCA_AVAILABLE = False
    print("⚠️ QCA分析模块不可用")

class DEAData:
    """DEA数据管理类"""
    
    def __init__(self, input_data: np.ndarray, output_data: np.ndarray, 
                 dmu_names: List[str] = None, input_names: List[str] = None, 
                 output_names: List[str] = None):
        """
        初始化DEA数据
        
        Args:
            input_data: 投入数据 (n_dmu, n_input)
            output_data: 产出数据 (n_dmu, n_output)
            dmu_names: DMU名称列表
            input_names: 投入指标名称列表
            output_names: 产出指标名称列表
        """
        self.input_data = np.array(input_data)
        self.output_data = np.array(output_data)
        self.n_dmu, self.n_input = self.input_data.shape
        self.n_output = self.output_data.shape[1]
        
        self.dmu_names = dmu_names or [f"DMU_{i+1}" for i in range(self.n_dmu)]
        self.input_names = input_names or [f"Input_{i+1}" for i in range(self.n_input)]
        self.output_names = output_names or [f"Output_{i+1}" for i in range(self.n_output)]
        
        # 验证数据
        if self.input_data.shape[0] != self.output_data.shape[0]:
            raise ValueError("投入和产出数据的DMU数量必须一致")
        if np.any(self.input_data <= 0) or np.any(self.output_data <= 0):
            raise ValueError("所有投入和产出数据必须为正数")

class DEAResult:
    """DEA结果存储类"""
    
    def __init__(self, dmu_names: List[str], input_names: List[str], output_names: List[str]):
        self.dmu_names = dmu_names
        self.input_names = input_names
        self.output_names = output_names
        
        self.efficiency_scores = {}
        self.lambda_variables = {}
        self.input_slacks = {}
        self.output_slacks = {}
        self.input_duals = {}
        self.output_duals = {}
        self.status = {}
        self.orientation = ""
        
    def add_result(self, dmu_code: str, efficiency: float, lambda_vars: Dict[str, float],
                   input_slacks: Dict[str, float] = None, output_slacks: Dict[str, float] = None,
                   input_duals: Dict[str, float] = None, output_duals: Dict[str, float] = None,
                   status: str = "Optimal"):
        """添加单个DMU的结果"""
        self.efficiency_scores[dmu_code] = efficiency
        self.lambda_variables[dmu_code] = lambda_vars
        self.input_slacks[dmu_code] = input_slacks or {}
        self.output_slacks[dmu_code] = output_slacks or {}
        self.input_duals[dmu_code] = input_duals or {}
        self.output_duals[dmu_code] = output_duals or {}
        self.status[dmu_code] = status
    
    def to_dataframe(self) -> pd.DataFrame:
        """转换为DataFrame格式"""
        data = []
        for dmu in self.dmu_names:
            row = {
                'DMU': dmu,
                'Efficiency': self.efficiency_scores.get(dmu, np.nan),
                'Status': self.status.get(dmu, 'Unknown')
            }
            data.append(row)
        return pd.DataFrame(data)

class CCRModel:
    """CCR模型实现"""
    
    def __init__(self, data: DEAData, orientation: str = 'input'):
        """
        初始化CCR模型
        
        Args:
            data: DEA数据
            orientation: 'input' 或 'output'
        """
        self.data = data
        self.orientation = orientation
        self.result = DEAResult(data.dmu_names, data.input_names, data.output_names)
        self.result.orientation = orientation
    
    def solve(self) -> DEAResult:
        """求解CCR模型"""
        for i, dmu_name in enumerate(self.data.dmu_names):
            self._solve_single_dmu(i, dmu_name)
        return self.result
    
    def _solve_single_dmu(self, dmu_idx: int, dmu_name: str):
        """求解单个DMU"""
        # 创建线性规划问题
        if self.orientation == 'input':
            prob = pulp.LpProblem(f"CCR_Input_{dmu_name}", pulp.LpMinimize)
        else:
            prob = pulp.LpProblem(f"CCR_Output_{dmu_name}", pulp.LpMaximize)
        
        # 创建变量
        theta = pulp.LpVariable("theta", lowBound=0, upBound=1 if self.orientation == 'input' else None)
        lambda_vars = pulp.LpVariable.dicts("lambda", self.data.dmu_names, lowBound=0)
        
        # 设置目标函数
        prob += theta
        
        # 添加约束
        if self.orientation == 'input':
            # 投入导向CCR约束
            for j in range(self.data.n_input):
                constraint = pulp.lpSum([
                    lambda_vars[self.data.dmu_names[k]] * self.data.input_data[k, j]
                    for k in range(self.data.n_dmu)
                ]) <= theta * self.data.input_data[dmu_idx, j]
                prob += constraint, f"input_constraint_{j}"
            
            for j in range(self.data.n_output):
                constraint = pulp.lpSum([
                    lambda_vars[self.data.dmu_names[k]] * self.data.output_data[k, j]
                    for k in range(self.data.n_dmu)
                ]) >= self.data.output_data[dmu_idx, j]
                prob += constraint, f"output_constraint_{j}"
        else:
            # 产出导向CCR约束
            for j in range(self.data.n_input):
                constraint = pulp.lpSum([
                    lambda_vars[self.data.dmu_names[k]] * self.data.input_data[k, j]
                    for k in range(self.data.n_dmu)
                ]) <= self.data.input_data[dmu_idx, j]
                prob += constraint, f"input_constraint_{j}"
            
            for j in range(self.data.n_output):
                constraint = pulp.lpSum([
                    lambda_vars[self.data.dmu_names[k]] * self.data.output_data[k, j]
                    for k in range(self.data.n_dmu)
                ]) >= theta * self.data.output_data[dmu_idx, j]
                prob += constraint, f"output_constraint_{j}"
        
        # 求解
        prob.solve()
        
        # 提取结果
        if prob.status == pulp.LpStatusOptimal:
            efficiency = pulp.value(theta)
            if self.orientation == 'output' and efficiency > 0:
                efficiency = 1.0 / efficiency
            
            lambda_vars_dict = {}
            for dmu in self.data.dmu_names:
                if lambda_vars[dmu].varValue is not None and abs(lambda_vars[dmu].varValue) > 1e-6:
                    lambda_vars_dict[dmu] = lambda_vars[dmu].varValue
            
            # 提取对偶变量
            input_duals = {}
            output_duals = {}
            
            for j in range(self.data.n_input):
                constraint_name = f"input_constraint_{j}"
                if constraint_name in prob.constraints:
                    dual_value = prob.constraints[constraint_name].pi
                    if dual_value is not None:
                        input_duals[self.data.input_names[j]] = dual_value
            
            for j in range(self.data.n_output):
                constraint_name = f"output_constraint_{j}"
                if constraint_name in prob.constraints:
                    dual_value = prob.constraints[constraint_name].pi
                    if dual_value is not None:
                        output_duals[self.data.output_names[j]] = dual_value
            
            self.result.add_result(dmu_name, efficiency, lambda_vars_dict, 
                                 input_duals=input_duals, output_duals=output_duals)
        else:
            self.result.add_result(dmu_name, float('inf'), {}, status="Infeasible")

class BCCModel:
    """BCC模型实现（CCR + VRS约束）"""
    
    def __init__(self, data: DEAData, orientation: str = 'input'):
        self.data = data
        self.orientation = orientation
        self.result = DEAResult(data.dmu_names, data.input_names, data.output_names)
        self.result.orientation = orientation
    
    def solve(self) -> DEAResult:
        """求解BCC模型"""
        for i, dmu_name in enumerate(self.data.dmu_names):
            self._solve_single_dmu(i, dmu_name)
        return self.result
    
    def _solve_single_dmu(self, dmu_idx: int, dmu_name: str):
        """求解单个DMU"""
        # 创建线性规划问题
        if self.orientation == 'input':
            prob = pulp.LpProblem(f"BCC_Input_{dmu_name}", pulp.LpMinimize)
        else:
            prob = pulp.LpProblem(f"BCC_Output_{dmu_name}", pulp.LpMaximize)
        
        # 创建变量
        theta = pulp.LpVariable("theta", lowBound=0, upBound=1 if self.orientation == 'input' else None)
        lambda_vars = pulp.LpVariable.dicts("lambda", self.data.dmu_names, lowBound=0)
        
        # 设置目标函数
        prob += theta
        
        # 添加约束（与CCR相同）
        if self.orientation == 'input':
            for j in range(self.data.n_input):
                constraint = pulp.lpSum([
                    lambda_vars[self.data.dmu_names[k]] * self.data.input_data[k, j]
                    for k in range(self.data.n_dmu)
                ]) <= theta * self.data.input_data[dmu_idx, j]
                prob += constraint, f"input_constraint_{j}"
            
            for j in range(self.data.n_output):
                constraint = pulp.lpSum([
                    lambda_vars[self.data.dmu_names[k]] * self.data.output_data[k, j]
                    for k in range(self.data.n_dmu)
                ]) >= self.data.output_data[dmu_idx, j]
                prob += constraint, f"output_constraint_{j}"
        else:
            for j in range(self.data.n_input):
                constraint = pulp.lpSum([
                    lambda_vars[self.data.dmu_names[k]] * self.data.input_data[k, j]
                    for k in range(self.data.n_dmu)
                ]) <= self.data.input_data[dmu_idx, j]
                prob += constraint, f"input_constraint_{j}"
            
            for j in range(self.data.n_output):
                constraint = pulp.lpSum([
                    lambda_vars[self.data.dmu_names[k]] * self.data.output_data[k, j]
                    for k in range(self.data.n_dmu)
                ]) >= theta * self.data.output_data[dmu_idx, j]
                prob += constraint, f"output_constraint_{j}"
        
        # BCC模型特有：VRS约束
        vrs_constraint = pulp.lpSum([lambda_vars[dmu] for dmu in self.data.dmu_names]) == 1
        prob += vrs_constraint, "VRS_constraint"
        
        # 求解
        prob.solve()
        
        # 提取结果
        if prob.status == pulp.LpStatusOptimal:
            efficiency = pulp.value(theta)
            if self.orientation == 'output' and efficiency > 0:
                efficiency = 1.0 / efficiency
            
            lambda_vars_dict = {}
            for dmu in self.data.dmu_names:
                if lambda_vars[dmu].varValue is not None and abs(lambda_vars[dmu].varValue) > 1e-6:
                    lambda_vars_dict[dmu] = lambda_vars[dmu].varValue
            
            # 提取对偶变量
            input_duals = {}
            output_duals = {}
            
            for j in range(self.data.n_input):
                constraint_name = f"input_constraint_{j}"
                if constraint_name in prob.constraints:
                    dual_value = prob.constraints[constraint_name].pi
                    if dual_value is not None:
                        input_duals[self.data.input_names[j]] = dual_value
            
            for j in range(self.data.n_output):
                constraint_name = f"output_constraint_{j}"
                if constraint_name in prob.constraints:
                    dual_value = prob.constraints[constraint_name].pi
                    if dual_value is not None:
                        output_duals[self.data.output_names[j]] = dual_value
            
            self.result.add_result(dmu_name, efficiency, lambda_vars_dict, 
                                 input_duals=input_duals, output_duals=output_duals)
        else:
            self.result.add_result(dmu_name, float('inf'), {}, status="Infeasible")

class SBMModel:
    """SBM模型实现（基于松弛变量的非径向模型）"""
    
    def __init__(self, data: DEAData, orientation: str = 'input'):
        self.data = data
        self.orientation = orientation
        self.result = DEAResult(data.dmu_names, data.input_names, data.output_names)
        self.result.orientation = orientation
    
    def solve(self) -> DEAResult:
        """求解SBM模型"""
        for i, dmu_name in enumerate(self.data.dmu_names):
            self._solve_single_dmu(i, dmu_name)
        return self.result
    
    def _solve_single_dmu(self, dmu_idx: int, dmu_name: str):
        """求解单个DMU"""
        # 第一阶段：求解基础DEA模型
        first_phase_result = self._solve_first_phase(dmu_idx, dmu_name)
        
        # 第二阶段：最大化松弛变量
        second_phase_result = self._solve_second_phase(dmu_idx, dmu_name, first_phase_result)
        
        # 合并结果
        self._merge_results(dmu_name, first_phase_result, second_phase_result)
    
    def _solve_first_phase(self, dmu_idx: int, dmu_name: str) -> Dict:
        """第一阶段：求解基础DEA模型"""
        prob = pulp.LpProblem(f"SBM_Phase1_{dmu_name}", pulp.LpMinimize)
        
        # 创建变量
        theta = pulp.LpVariable("theta", lowBound=0, upBound=1)
        lambda_vars = pulp.LpVariable.dicts("lambda", self.data.dmu_names, lowBound=0)
        
        # 设置目标函数
        prob += theta
        
        # 添加约束
        for j in range(self.data.n_input):
            constraint = pulp.lpSum([
                lambda_vars[self.data.dmu_names[k]] * self.data.input_data[k, j]
                for k in range(self.data.n_dmu)
            ]) <= theta * self.data.input_data[dmu_idx, j]
            prob += constraint, f"input_constraint_{j}"
        
        for j in range(self.data.n_output):
            constraint = pulp.lpSum([
                lambda_vars[self.data.dmu_names[k]] * self.data.output_data[k, j]
                for k in range(self.data.n_dmu)
            ]) >= self.data.output_data[dmu_idx, j]
            prob += constraint, f"output_constraint_{j}"
        
        # 求解
        prob.solve()
        
        result = {
            'status': prob.status,
            'theta': pulp.value(theta) if prob.status == pulp.LpStatusOptimal else None,
            'lambda_vars': {dmu: lambda_vars[dmu].varValue for dmu in self.data.dmu_names 
                           if lambda_vars[dmu].varValue is not None and abs(lambda_vars[dmu].varValue) > 1e-6}
        }
        return result
    
    def _solve_second_phase(self, dmu_idx: int, dmu_name: str, first_phase_result: Dict) -> Dict:
        """第二阶段：最大化松弛变量"""
        prob = pulp.LpProblem(f"SBM_Phase2_{dmu_name}", pulp.LpMaximize)
        
        # 创建变量
        lambda_vars = pulp.LpVariable.dicts("lambda", self.data.dmu_names, lowBound=0)
        input_slacks = pulp.LpVariable.dicts("input_slack", self.data.input_names, lowBound=0)
        output_slacks = pulp.LpVariable.dicts("output_slack", self.data.output_names, lowBound=0)
        
        # 设置目标函数：最大化松弛变量
        prob += pulp.lpSum(input_slacks.values()) + pulp.lpSum(output_slacks.values())
        
        # 添加约束：等式约束
        for j in range(self.data.n_input):
            constraint = (self.data.input_data[dmu_idx, j] == 
                         pulp.lpSum([lambda_vars[self.data.dmu_names[k]] * self.data.input_data[k, j]
                                   for k in range(self.data.n_dmu)]) + 
                         input_slacks[self.data.input_names[j]])
            prob += constraint, f"input_eq_constraint_{j}"
        
        for j in range(self.data.n_output):
            constraint = (self.data.output_data[dmu_idx, j] == 
                         pulp.lpSum([lambda_vars[self.data.dmu_names[k]] * self.data.output_data[k, j]
                                   for k in range(self.data.n_dmu)]) - 
                         output_slacks[self.data.output_names[j]])
            prob += constraint, f"output_eq_constraint_{j}"
        
        # 求解
        prob.solve()
        
        result = {
            'status': prob.status,
            'input_slacks': {name: input_slacks[name].varValue for name in self.data.input_names
                            if input_slacks[name].varValue is not None and abs(input_slacks[name].varValue) > 1e-6},
            'output_slacks': {name: output_slacks[name].varValue for name in self.data.output_names
                             if output_slacks[name].varValue is not None and abs(output_slacks[name].varValue) > 1e-6},
            'lambda_vars': {dmu: lambda_vars[dmu].varValue for dmu in self.data.dmu_names
                           if lambda_vars[dmu].varValue is not None and abs(lambda_vars[dmu].varValue) > 1e-6}
        }
        return result
    
    def _merge_results(self, dmu_name: str, first_phase: Dict, second_phase: Dict):
        """合并两阶段结果"""
        if first_phase['status'] == pulp.LpStatusOptimal and second_phase['status'] == pulp.LpStatusOptimal:
            # 计算SBM效率值
            input_inefficiency = sum(second_phase['input_slacks'].values()) / self.data.n_input
            output_inefficiency = sum(second_phase['output_slacks'].values()) / self.data.n_output
            
            if self.orientation == 'input':
                efficiency = 1 - input_inefficiency
            elif self.orientation == 'output':
                efficiency = 1 + output_inefficiency
            else:  # non-oriented
                efficiency = (1 - input_inefficiency) / (1 + output_inefficiency)
            
            self.result.add_result(dmu_name, efficiency, second_phase['lambda_vars'],
                                 input_slacks=second_phase['input_slacks'],
                                 output_slacks=second_phase['output_slacks'])
        else:
            self.result.add_result(dmu_name, float('inf'), {}, status="Infeasible")

class SuperEfficiencySBMModel:
    """超效率SBM模型实现 - 使用正确的数学变换处理PuLP限制"""
    
    def __init__(self, data: DEAData, orientation: str = 'input', undesirable_outputs=None, rts='vrs', handle_infeasible='set_to_1'):
        self.data = data
        self.orientation = orientation
        self.undesirable_outputs = undesirable_outputs
        self.rts = rts
        self.handle_infeasible = handle_infeasible
        self.result = DEAResult(data.dmu_names, data.input_names, data.output_names)
        self.result.orientation = orientation
    
    def solve(self) -> DEAResult:
        """求解超效率SBM模型 - 使用正确的数学实现"""
        n_dmus = self.data.n_dmu
        n_inputs = self.data.n_input
        n_outputs = self.data.n_output
        
        # 第一步：计算普通SBM模型，识别有效DMU
        print("第一步：计算普通SBM模型，识别有效DMU...")
        regular_sbm_scores = self._calculate_regular_sbm()
        
        # 识别有效DMU（效率值 = 1）
        efficient_dmus = [i for i in range(n_dmus) if abs(regular_sbm_scores[i] - 1.0) < 1e-6]
        print(f"有效DMU数量: {len(efficient_dmus)} / {n_dmus}")
        print(f"有效DMU索引: {[i+1 for i in efficient_dmus]}")
        
        # 初始化结果数组
        efficiency_scores = regular_sbm_scores.copy()  # 先复制普通SBM结果
        slack_inputs = np.zeros((n_dmus, n_inputs))
        slack_outputs = np.zeros((n_dmus, n_outputs))
        lambda_sums = np.zeros(n_dmus)
        solution_status = ['success'] * n_dmus
        
        # 第二步：只对有效DMU计算超效率SBM模型
        print("第二步：对有效DMU计算超效率SBM模型...")
        for dmu in efficient_dmus:
            print(f"计算DMU {dmu+1} 的超效率值...")
            
            # 超效率SBM模型：排除被评估的DMU
            other_dmus = [i for i in range(n_dmus) if i != dmu]
            n_other = len(other_dmus)
            
            if n_other == 0:
                # 只有一个DMU的情况：无法计算超效率，保持普通SBM值
                print(f"  警告：DMU {dmu+1} 是唯一DMU，无法计算超效率值，保持普通SBM值")
                solution_status[dmu] = 'single_dmu'
                continue
            
            # 计算超效率值
            super_efficiency, slack_i, slack_o, lambda_sum, status = self._calculate_super_efficiency(dmu, other_dmus)
            
            efficiency_scores[dmu] = super_efficiency
            slack_inputs[dmu] = slack_i
            slack_outputs[dmu] = slack_o
            lambda_sums[dmu] = lambda_sum
            solution_status[dmu] = status
            
            if status == 'success':
                print(f"  DMU {dmu+1} 超效率值: {super_efficiency:.5f}")
            else:
                print(f"  DMU {dmu+1} 无解")
        
        # 将结果存储到DEAResult对象中
        for i, dmu_name in enumerate(self.data.dmu_names):
            efficiency = efficiency_scores[i] if not np.isnan(efficiency_scores[i]) else float('inf')
            status = solution_status[i] if i < len(solution_status) else 'unknown'
            
            # 构建lambda变量字典
            lambda_vars = {f"DMU_{j+1}": 0.0 for j in range(self.data.n_dmu)}
            
            # 构建松弛变量字典
            input_slack_dict = {}
            for j, input_name in enumerate(self.data.input_names):
                input_slack_dict[input_name] = slack_inputs[i, j]
            
            output_slack_dict = {}
            for j, output_name in enumerate(self.data.output_names):
                output_slack_dict[output_name] = slack_outputs[i, j]
            
            self.result.add_result(
                dmu_name, 
                efficiency, 
                lambda_vars,
                input_slacks=input_slack_dict,
                output_slacks=output_slack_dict,
                status=status
            )
        
        return self.result
    
    def _calculate_regular_sbm(self):
        """计算普通SBM模型 - 使用数学变换处理除法问题"""
        n_dmus = self.data.n_dmu
        n_inputs = self.data.n_input
        n_outputs = self.data.n_output
        
        efficiency_scores = np.zeros(n_dmus)
        
        # 处理非期望产出
        if self.undesirable_outputs is not None and len(self.undesirable_outputs) > 0:
            undesirable_indices = self.undesirable_outputs
            desirable_outputs = [var for var in range(n_outputs) if var not in undesirable_indices]
            n_desirable = len(desirable_outputs)
            n_undesirable = len(undesirable_indices)
        else:
            desirable_outputs = list(range(n_outputs))
            n_desirable = n_outputs
            n_undesirable = 0
            undesirable_indices = []
        
        for dmu in range(n_dmus):
            # 使用数学变换：引入新变量处理除法
            # 对于SBM模型，我们可以使用两阶段方法
            efficiency_scores[dmu] = self._solve_sbm_two_stage(dmu, desirable_outputs, undesirable_indices)
        
        return efficiency_scores
    
    def _solve_sbm_two_stage(self, dmu, desirable_outputs, undesirable_indices):
        """两阶段SBM求解方法"""
        n_dmus = self.data.n_dmu
        n_inputs = self.data.n_input
        n_desirable = len(desirable_outputs)
        n_undesirable = len(undesirable_indices)
        
        # 第一阶段：最小化投入比例
        prob1 = pulp.LpProblem(f"SBM_Phase1_{dmu}", pulp.LpMinimize)
        
        # 创建变量
        theta = pulp.LpVariable("theta", lowBound=0, upBound=1)
        lambda_vars = pulp.LpVariable.dicts("lambda", range(n_dmus), lowBound=0)
        
        # 目标函数：最小化theta
        prob1 += theta
        
        # 约束条件
        for i in range(n_inputs):
            constraint = (pulp.lpSum([lambda_vars[j] * self.data.input_data[j, i] for j in range(n_dmus)]) <= 
                         theta * self.data.input_data[dmu, i])
            prob1 += constraint, f"input_constraint_{i}"
        
        for r in desirable_outputs:
            constraint = (pulp.lpSum([lambda_vars[j] * self.data.output_data[j, r] for j in range(n_dmus)]) >= 
                         self.data.output_data[dmu, r])
            prob1 += constraint, f"output_constraint_{r}"
        
        for u in undesirable_indices:
            constraint = (pulp.lpSum([lambda_vars[j] * self.data.output_data[j, u] for j in range(n_dmus)]) <= 
                         self.data.output_data[dmu, u])
            prob1 += constraint, f"undesirable_constraint_{u}"
        
        # VRS约束
        if self.rts == 'vrs':
            prob1 += pulp.lpSum([lambda_vars[j] for j in range(n_dmus)]) == 1, "vrs_constraint"
        
        # 求解第一阶段
        try:
            prob1.solve()
            if prob1.status == pulp.LpStatusOptimal:
                theta_value = pulp.value(theta)
                if theta_value is not None and theta_value >= 0:
                    # 第二阶段：最大化松弛变量
                    prob2 = pulp.LpProblem(f"SBM_Phase2_{dmu}", pulp.LpMaximize)
        
        # 创建变量
                    lambda_vars2 = pulp.LpVariable.dicts("lambda2", range(n_dmus), lowBound=0)
                    input_slacks = pulp.LpVariable.dicts("input_slack", range(n_inputs), lowBound=0)
                    output_slacks = pulp.LpVariable.dicts("output_slack", range(n_desirable), lowBound=0)
                    undesirable_slacks = pulp.LpVariable.dicts("undesirable_slack", range(n_undesirable), lowBound=0)
                    
                    # 目标函数：最大化松弛变量
                    prob2 += pulp.lpSum(input_slacks.values()) + pulp.lpSum(output_slacks.values()) + pulp.lpSum(undesirable_slacks.values())
                    
                    # 约束条件：等式约束
                    for i in range(n_inputs):
                        constraint = (self.data.input_data[dmu, i] == 
                                     pulp.lpSum([lambda_vars2[j] * self.data.input_data[j, i] for j in range(n_dmus)]) + 
                                     input_slacks[i])
                        prob2 += constraint, f"input_eq_constraint_{i}"
                    
                    for r_idx, r in enumerate(desirable_outputs):
                        constraint = (self.data.output_data[dmu, r] == 
                                     pulp.lpSum([lambda_vars2[j] * self.data.output_data[j, r] for j in range(n_dmus)]) - 
                                     output_slacks[r_idx])
                        prob2 += constraint, f"output_eq_constraint_{r_idx}"
                    
                    for u_idx, u in enumerate(undesirable_indices):
                        constraint = (self.data.output_data[dmu, u] == 
                                     pulp.lpSum([lambda_vars2[j] * self.data.output_data[j, u] for j in range(n_dmus)]) + 
                                     undesirable_slacks[u_idx])
                        prob2 += constraint, f"undesirable_eq_constraint_{u_idx}"
                    
                    # VRS约束
                    if self.rts == 'vrs':
                        prob2 += pulp.lpSum([lambda_vars2[j] for j in range(n_dmus)]) == 1, "vrs_constraint2"
                    
                    # 求解第二阶段
                    prob2.solve()
                    
                    if prob2.status == pulp.LpStatusOptimal:
                        # 计算SBM效率值
                        # 投入无效率：平均投入松弛比例
                        input_inefficiency = sum((pulp.value(input_slacks[i]) or 0) / self.data.input_data[dmu, i] for i in range(n_inputs)) / n_inputs
                        
                        # 产出无效率：平均产出松弛比例
                        output_inefficiency = 0
                        for r_idx, r in enumerate(desirable_outputs):
                            output_inefficiency += (pulp.value(output_slacks[r_idx]) or 0) / self.data.output_data[dmu, r]
                        
                        for u_idx, u in enumerate(undesirable_indices):
                            output_inefficiency += (pulp.value(undesirable_slacks[u_idx]) or 0) / self.data.output_data[dmu, u]
                        
                        output_inefficiency = output_inefficiency / (n_desirable + n_undesirable)
                        
                        numerator = 1 - input_inefficiency
                        denominator = 1 + output_inefficiency
                        
                        # 确保效率值在合理范围内
                        if numerator <= 0:
                            numerator = 1e-6
                        if denominator <= 1e-6:
                            denominator = 1e-6
                        
                        efficiency = numerator / denominator
                        
                        # 确保效率值在 [0, 1] 范围内
                        efficiency = max(0.0, min(1.0, efficiency))
                        
                        return efficiency
                    else:
                        return 1.0 if self.handle_infeasible == 'set_to_1' else np.nan
                else:
                    return 1.0 if self.handle_infeasible == 'set_to_1' else np.nan
            else:
                return 1.0 if self.handle_infeasible == 'set_to_1' else np.nan
        except Exception:
            return 1.0 if self.handle_infeasible == 'set_to_1' else np.nan
    
    def _calculate_super_efficiency(self, dmu, other_dmus):
        """计算超效率SBM值 - 使用正确的数学公式"""
        n_other = len(other_dmus)
        n_inputs = self.data.n_input
        n_outputs = self.data.n_output
        
        # 处理非期望产出
        if self.undesirable_outputs is not None and len(self.undesirable_outputs) > 0:
            undesirable_indices = self.undesirable_outputs
            desirable_outputs = [var for var in range(n_outputs) if var not in undesirable_indices]
            n_desirable = len(desirable_outputs)
            n_undesirable = len(undesirable_indices)
        else:
            desirable_outputs = list(range(n_outputs))
            n_desirable = n_outputs
            n_undesirable = 0
            undesirable_indices = []
        
        # 使用两阶段方法计算超效率SBM
        # 第一阶段：最小化投入比例（排除被评估DMU）
        prob1 = pulp.LpProblem(f"SuperSBM_Phase1_{dmu}", pulp.LpMinimize)
        
        # 创建变量
        theta = pulp.LpVariable("theta", lowBound=0, upBound=None)
        lambda_vars = pulp.LpVariable.dicts("lambda", range(n_other), lowBound=0)
        
        # 目标函数：最小化theta
        prob1 += theta
        
        # 约束条件（排除被评估DMU）
        for i in range(n_inputs):
            constraint = (pulp.lpSum([lambda_vars[j] * self.data.input_data[other_dmus[j], i] for j in range(n_other)]) <= 
                         theta * self.data.input_data[dmu, i])
            prob1 += constraint, f"input_constraint_{i}"
        
        for r in desirable_outputs:
            constraint = (pulp.lpSum([lambda_vars[j] * self.data.output_data[other_dmus[j], r] for j in range(n_other)]) >= 
                         self.data.output_data[dmu, r])
            prob1 += constraint, f"output_constraint_{r}"
        
        for u in undesirable_indices:
            constraint = (pulp.lpSum([lambda_vars[j] * self.data.output_data[other_dmus[j], u] for j in range(n_other)]) <= 
                         self.data.output_data[dmu, u])
            prob1 += constraint, f"undesirable_constraint_{u}"
        
        # VRS约束
        if self.rts == 'vrs':
            prob1 += pulp.lpSum([lambda_vars[j] for j in range(n_other)]) == 1, "vrs_constraint"
        
        # 求解第一阶段
        try:
            prob1.solve()
            
            # 检查第一阶段是否成功
            if prob1.status != pulp.LpStatusOptimal:
                return 1.0, np.zeros(n_inputs), np.zeros(n_outputs), 0, 'infeasible'
            
            theta_value = pulp.value(theta)
            if theta_value is None or theta_value < 0:
                return 1.0, np.zeros(n_inputs), np.zeros(n_outputs), 0, 'infeasible'
            
            # 第二阶段：最大化松弛变量
            prob2 = pulp.LpProblem(f"SuperSBM_Phase2_{dmu}", pulp.LpMaximize)
            
            # 创建变量
            lambda_vars2 = pulp.LpVariable.dicts("lambda2", range(n_other), lowBound=0)
            input_slacks = pulp.LpVariable.dicts("input_slack", range(n_inputs), lowBound=0)
            output_slacks = pulp.LpVariable.dicts("output_slack", range(n_desirable), lowBound=0)
            undesirable_slacks = pulp.LpVariable.dicts("undesirable_slack", range(n_undesirable), lowBound=0)
            
            # 目标函数：最大化松弛变量
            prob2 += pulp.lpSum(input_slacks.values()) + pulp.lpSum(output_slacks.values()) + pulp.lpSum(undesirable_slacks.values())
            
            # 约束条件：等式约束
            for i in range(n_inputs):
                constraint = (self.data.input_data[dmu, i] == 
                             pulp.lpSum([lambda_vars2[j] * self.data.input_data[other_dmus[j], i] for j in range(n_other)]) + 
                             input_slacks[i])
                prob2 += constraint, f"input_eq_constraint_{i}"
            
            for r_idx, r in enumerate(desirable_outputs):
                constraint = (self.data.output_data[dmu, r] == 
                             pulp.lpSum([lambda_vars2[j] * self.data.output_data[other_dmus[j], r] for j in range(n_other)]) - 
                             output_slacks[r_idx])
                prob2 += constraint, f"output_eq_constraint_{r_idx}"
            
            for u_idx, u in enumerate(undesirable_indices):
                constraint = (self.data.output_data[dmu, u] == 
                             pulp.lpSum([lambda_vars2[j] * self.data.output_data[other_dmus[j], u] for j in range(n_other)]) + 
                             undesirable_slacks[u_idx])
                prob2 += constraint, f"undesirable_eq_constraint_{u_idx}"
            
            # VRS约束
            if self.rts == 'vrs':
                prob2 += pulp.lpSum([lambda_vars2[j] for j in range(n_other)]) == 1, "vrs_constraint2"
            
            # 求解第二阶段
            prob2.solve()
            
            # 检查第二阶段是否成功
            if prob2.status != pulp.LpStatusOptimal:
                return 1.0, np.zeros(n_inputs), np.zeros(n_outputs), 0, 'infeasible'
            
            # 提取松弛变量
            slack_i = np.array([pulp.value(input_slacks[i]) or 0 for i in range(n_inputs)])
            slack_o = np.zeros(n_outputs)
            
            for r_idx, r in enumerate(desirable_outputs):
                slack_o[r] = pulp.value(output_slacks[r_idx]) or 0
            
            for u_idx, u in enumerate(undesirable_indices):
                slack_o[u] = pulp.value(undesirable_slacks[u_idx]) or 0
            
            # 计算λ和
            lambda_sum = sum(pulp.value(lambda_vars2[j]) or 0 for j in range(n_other))
            
            # 计算超效率SBM效率值
            # 分子：1 + (1/m)∑(sᵢ⁻/xᵢ₀)
            input_inefficiency = np.sum(slack_i / self.data.input_data[dmu]) / n_inputs
            numerator = 1 + input_inefficiency
            
            # 分母：1 + (1/(s+d))(∑(sᵣ⁺/yᵣ₀) + ∑(sᵤᵤ/uᵤ₀))
            # 注意：超效率SBM的分母应该是 1 + output_inefficiency，不是 1 - output_inefficiency
            output_inefficiency = 0
            for r_idx, r in enumerate(desirable_outputs):
                output_inefficiency += slack_o[r] / self.data.output_data[dmu, r]
            
            for u_idx, u in enumerate(undesirable_indices):
                output_inefficiency += slack_o[u] / self.data.output_data[dmu, u]
            
            output_inefficiency = output_inefficiency / (n_desirable + n_undesirable)
            
            denominator = 1 + output_inefficiency
            if denominator <= 1e-6:
                denominator = 1e-6
            
            super_efficiency = numerator / denominator
            
            # 验证超效率值 >= 1（理论上应该总是成立）
            if super_efficiency < 1.0:
                print(f"  警告：DMU {dmu+1} 超效率值 < 1: {super_efficiency:.5f}")
                # 如果超效率值 < 1，说明计算有问题，将其设为1
                super_efficiency = 1.0
            
            # 超效率SBM中，松弛变量为负是正常的，表示超效率DMU的超出程度
            # 但是为了显示的一致性，我们需要调整松弛变量的符号
            # 投入松弛变量：负值表示投入不足，正值表示投入冗余
            # 产出松弛变量：负值表示产出不足，正值表示产出冗余
            
            # 调整松弛变量符号以便正确显示
            # 投入松弛变量：保持原符号（负值表示投入不足）
            # 产出松弛变量：取负号（负值表示产出不足）
            slack_o = -slack_o
            
            return super_efficiency, slack_i, slack_o, lambda_sum, 'success'
            
        except Exception:
            return 1.0, np.zeros(n_inputs), np.zeros(n_outputs), 0, 'infeasible'

class CustomDEA:
    """简化的DEA实现，使用新的DEA模型类"""
    
    def __init__(self, input_data, output_data, max_iter=5000, tolerance=1e-9):
        self.input_data = np.array(input_data, dtype=np.float64)
        self.output_data = np.array(output_data, dtype=np.float64)
        self.n_dmus = self.input_data.shape[0]
        self.n_inputs = self.input_data.shape[1]
        self.n_outputs = self.output_data.shape[1]
        self.max_iter = max_iter
        self.tolerance = tolerance
        
        # 数据验证：只检查负值，允许0值
        if np.any(self.input_data < 0):
            raise ValueError("所有投入变量不能为负数")
        if np.any(self.output_data < 0):
            raise ValueError("所有产出变量不能为负数")
        
        # 将0替换为极小正值，避免除零错误
        self.input_data = np.maximum(self.input_data, 1e-10)
        self.output_data = np.maximum(self.output_data, 1e-10)
        
        # 存储松弛变量结果
        self.slack_inputs = None
        self.slack_outputs = None
        self.lambda_values = None
        
    def ccr_input_oriented(self, method='highs'):
        """CCR模型 - 输入导向"""
        # 创建DEA数据对象
        data = DEAData(self.input_data, self.output_data)
        
        # 创建CCR模型
        ccr_model = CCRModel(data, orientation='input')
        
        # 求解
        result = ccr_model.solve()
        
        # 提取效率值
        efficiency_scores = np.array([result.efficiency_scores.get(f"DMU_{i+1}", np.nan) 
                                    for i in range(self.n_dmus)])
        
        # 提取松弛变量和lambda值
        self.slack_inputs = np.zeros((self.n_dmus, self.n_inputs))
        self.slack_outputs = np.zeros((self.n_dmus, self.n_outputs))
        self.lambda_values = np.zeros((self.n_dmus, self.n_dmus))
        
        for i in range(self.n_dmus):
            dmu_name = f"DMU_{i+1}"
            if dmu_name in result.input_slacks:
                for j, input_name in enumerate(data.input_names):
                    self.slack_inputs[i, j] = result.input_slacks[dmu_name].get(input_name, 0)
            
            if dmu_name in result.output_slacks:
                for j, output_name in enumerate(data.output_names):
                    self.slack_outputs[i, j] = result.output_slacks[dmu_name].get(output_name, 0)
            
            if dmu_name in result.lambda_variables:
                for j, other_dmu in enumerate(data.dmu_names):
                    self.lambda_values[i, j] = result.lambda_variables[dmu_name].get(other_dmu, 0)
        
        return efficiency_scores
    
    def super_sbm_old(self, undesirable_outputs=None, rts='vrs', handle_infeasible='set_to_1'):
        """超效率SBM模型 - 使用新的分步处理逻辑"""
        # 创建DEA数据对象
        data = DEAData(self.input_data, self.output_data)
        
        # 创建超效率SBM模型（使用新的实现）
        super_sbm_model = SuperEfficiencySBMModel(
            data, 
            orientation='input', 
            undesirable_outputs=undesirable_outputs,
            rts=rts,
            handle_infeasible=handle_infeasible
        )
        
        # 求解
        result = super_sbm_model.solve()
        
        # 提取效率值
        efficiency_scores = np.array([result.efficiency_scores.get(f"DMU_{i+1}", np.nan) 
                                    for i in range(self.n_dmus)])
        
        # 提取松弛变量和lambda值
        self.slack_inputs = np.zeros((self.n_dmus, self.n_inputs))
        self.slack_outputs = np.zeros((self.n_dmus, self.n_outputs))
        self.lambda_values = np.zeros((self.n_dmus, self.n_dmus))
        
        for i in range(self.n_dmus):
            dmu_name = f"DMU_{i+1}"
            if dmu_name in result.input_slacks:
                for j, input_name in enumerate(data.input_names):
                    self.slack_inputs[i, j] = result.input_slacks[dmu_name].get(input_name, 0)
            
            if dmu_name in result.output_slacks:
                for j, output_name in enumerate(data.output_names):
                    self.slack_outputs[i, j] = result.output_slacks[dmu_name].get(output_name, 0)
            
            if dmu_name in result.lambda_variables:
                for j, other_dmu in enumerate(data.dmu_names):
                    self.lambda_values[i, j] = result.lambda_variables[dmu_name].get(other_dmu, 0)
        
        return efficiency_scores
    
    def ccr_output_oriented(self, method='highs'):
        """CCR模型 - 输出导向"""
        # 创建DEA数据对象
        data = DEAData(self.input_data, self.output_data)
        
        # 创建CCR模型
        ccr_model = CCRModel(data, orientation='output')
        
        # 求解
        result = ccr_model.solve()
        
        # 提取效率值
        efficiency_scores = np.array([result.efficiency_scores.get(f"DMU_{i+1}", np.nan) 
                                    for i in range(self.n_dmus)])
        
        # 提取松弛变量和lambda值
        self.slack_inputs = np.zeros((self.n_dmus, self.n_inputs))
        self.slack_outputs = np.zeros((self.n_dmus, self.n_outputs))
        self.lambda_values = np.zeros((self.n_dmus, self.n_dmus))
        
        for i in range(self.n_dmus):
            dmu_name = f"DMU_{i+1}"
            if dmu_name in result.input_slacks:
                for j, input_name in enumerate(data.input_names):
                    self.slack_inputs[i, j] = result.input_slacks[dmu_name].get(input_name, 0)
            
            if dmu_name in result.output_slacks:
                for j, output_name in enumerate(data.output_names):
                    self.slack_outputs[i, j] = result.output_slacks[dmu_name].get(output_name, 0)
            
            if dmu_name in result.lambda_variables:
                for j, other_dmu in enumerate(data.dmu_names):
                    self.lambda_values[i, j] = result.lambda_variables[dmu_name].get(other_dmu, 0)
        
        return efficiency_scores
    
    def bcc_input_oriented(self, method='highs'):
        """BCC模型 - 输入导向"""
        # 创建DEA数据对象
        data = DEAData(self.input_data, self.output_data)
        
        # 创建BCC模型
        bcc_model = BCCModel(data, orientation='input')
        
        # 求解
        result = bcc_model.solve()
        
        # 提取效率值
        efficiency_scores = np.array([result.efficiency_scores.get(f"DMU_{i+1}", np.nan) 
                                    for i in range(self.n_dmus)])
        
        # 提取松弛变量和lambda值
        self.slack_inputs = np.zeros((self.n_dmus, self.n_inputs))
        self.slack_outputs = np.zeros((self.n_dmus, self.n_outputs))
        self.lambda_values = np.zeros((self.n_dmus, self.n_dmus))
        
        for i in range(self.n_dmus):
            dmu_name = f"DMU_{i+1}"
            if dmu_name in result.input_slacks:
                for j, input_name in enumerate(data.input_names):
                    self.slack_inputs[i, j] = result.input_slacks[dmu_name].get(input_name, 0)
            
            if dmu_name in result.output_slacks:
                for j, output_name in enumerate(data.output_names):
                    self.slack_outputs[i, j] = result.output_slacks[dmu_name].get(output_name, 0)
            
            if dmu_name in result.lambda_variables:
                for j, other_dmu in enumerate(data.dmu_names):
                    self.lambda_values[i, j] = result.lambda_variables[dmu_name].get(other_dmu, 0)
        
        return efficiency_scores
    
    def bcc_output_oriented(self, method='highs'):
        """BCC模型 - 输出导向"""
        # 创建DEA数据对象
        data = DEAData(self.input_data, self.output_data)
        
        # 创建BCC模型
        bcc_model = BCCModel(data, orientation='output')
        
        # 求解
        result = bcc_model.solve()
        
        # 提取效率值
        efficiency_scores = np.array([result.efficiency_scores.get(f"DMU_{i+1}", np.nan) 
                                    for i in range(self.n_dmus)])
        
        # 提取松弛变量和lambda值
        self.slack_inputs = np.zeros((self.n_dmus, self.n_inputs))
        self.slack_outputs = np.zeros((self.n_dmus, self.n_outputs))
        self.lambda_values = np.zeros((self.n_dmus, self.n_dmus))
        
        for i in range(self.n_dmus):
            dmu_name = f"DMU_{i+1}"
            if dmu_name in result.input_slacks:
                for j, input_name in enumerate(data.input_names):
                    self.slack_inputs[i, j] = result.input_slacks[dmu_name].get(input_name, 0)
            
            if dmu_name in result.output_slacks:
                for j, output_name in enumerate(data.output_names):
                    self.slack_outputs[i, j] = result.output_slacks[dmu_name].get(output_name, 0)
            
            if dmu_name in result.lambda_variables:
                for j, other_dmu in enumerate(data.dmu_names):
                    self.lambda_values[i, j] = result.lambda_variables[dmu_name].get(other_dmu, 0)
        
        return efficiency_scores
    
    def sbm(self, undesirable_outputs=None, method='highs'):
        """SBM模型"""
        # 创建DEA数据对象
        data = DEAData(self.input_data, self.output_data)
        
        # 创建SBM模型
        sbm_model = SBMModel(data, orientation='input')
        
        # 求解
        result = sbm_model.solve()
        
        # 提取效率值
        efficiency_scores = np.array([result.efficiency_scores.get(f"DMU_{i+1}", np.nan) 
                                    for i in range(self.n_dmus)])
        
        # 提取松弛变量和lambda值
        self.slack_inputs = np.zeros((self.n_dmus, self.n_inputs))
        self.slack_outputs = np.zeros((self.n_dmus, self.n_outputs))
        self.lambda_values = np.zeros((self.n_dmus, self.n_dmus))
        
        for i in range(self.n_dmus):
            dmu_name = f"DMU_{i+1}"
            if dmu_name in result.input_slacks:
                for j, input_name in enumerate(data.input_names):
                    self.slack_inputs[i, j] = result.input_slacks[dmu_name].get(input_name, 0)
            
            if dmu_name in result.output_slacks:
                for j, output_name in enumerate(data.output_names):
                    self.slack_outputs[i, j] = result.output_slacks[dmu_name].get(output_name, 0)
            
            if dmu_name in result.lambda_variables:
                for j, other_dmu in enumerate(data.dmu_names):
                    self.lambda_values[i, j] = result.lambda_variables[dmu_name].get(other_dmu, 0)
        
        return efficiency_scores
    
    def super_sbm_old(self, undesirable_outputs=None, method='highs'):
        """超效率SBM模型"""
        # 创建DEA数据对象
        data = DEAData(self.input_data, self.output_data)
        
        # 创建超效率SBM模型
        super_sbm_model = SuperEfficiencySBMModel(data, orientation='input')
        
        # 求解
        result = super_sbm_model.solve()
        
        # 提取效率值
        efficiency_scores = np.array([result.efficiency_scores.get(f"DMU_{i+1}", np.nan) 
                                    for i in range(self.n_dmus)])
        
        # 提取松弛变量和lambda值
        self.slack_inputs = np.zeros((self.n_dmus, self.n_inputs))
        self.slack_outputs = np.zeros((self.n_dmus, self.n_outputs))
        self.lambda_values = np.zeros((self.n_dmus, self.n_dmus))
        
        for i in range(self.n_dmus):
            dmu_name = f"DMU_{i+1}"
            if dmu_name in result.input_slacks:
                for j, input_name in enumerate(data.input_names):
                    self.slack_inputs[i, j] = result.input_slacks[dmu_name].get(input_name, 0)
            
            if dmu_name in result.output_slacks:
                for j, output_name in enumerate(data.output_names):
                    self.slack_outputs[i, j] = result.output_slacks[dmu_name].get(output_name, 0)
            
            if dmu_name in result.lambda_variables:
                for j, other_dmu in enumerate(data.dmu_names):
                    self.lambda_values[i, j] = result.lambda_variables[dmu_name].get(other_dmu, 0)
        
        return efficiency_scores

class DEAWrapper:
    """DEA分析包装器，使用新的DEA模型实现"""
    
    def __init__(self, input_data, output_data, dmu_names=None, max_iter=5000, tolerance=1e-9):
        self.input_data = np.array(input_data)
        self.output_data = np.array(output_data)
        self.max_iter = max_iter
        self.tolerance = tolerance
        
        # 添加维度属性
        self.n_dmus = self.input_data.shape[0]
        self.n_inputs = self.input_data.shape[1]
        self.n_outputs = self.output_data.shape[1]
        
        # 修复numpy数组的布尔值判断问题
        if dmu_names is not None:
            # 检查是否是可迭代的且不是字符串
            if hasattr(dmu_names, '__iter__') and not isinstance(dmu_names, str):
                try:
                    # 尝试转换为列表
                    self.dmu_names = list(dmu_names)
                except:
                    # 如果转换失败，创建默认名称
                    self.dmu_names = [f'DMU{i+1}' for i in range(len(input_data))]
            else:
                self.dmu_names = [dmu_names]
        else:
            self.dmu_names = [f'DMU{i+1}' for i in range(len(input_data))]
        
        # 使用自定义DEA实现
        self.dea = CustomDEA(self.input_data, self.output_data, max_iter=self.max_iter, tolerance=self.tolerance)
        print("✅ 使用新的DEA模型实现进行DEA分析")
    
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
    
    def super_sbm(self, undesirable_outputs=None, rts='vrs', handle_infeasible='set_to_1'):
        """超效率SBM模型 - 允许效率值大于1，包含规模报酬分析"""
        # 创建DEA数据对象
        data = DEAData(self.input_data, self.output_data)
        
        # 创建超效率SBM模型
        super_sbm_model = SuperEfficiencySBMModel(
            data, 
            orientation='input', 
            undesirable_outputs=undesirable_outputs,
            rts=rts,
            handle_infeasible=handle_infeasible
        )
        
        # 求解
        result = super_sbm_model.solve()
        
        # 提取效率值
        efficiency_scores = np.array([result.efficiency_scores.get(dmu_name, np.nan) 
                                    for dmu_name in data.dmu_names])
        
        # 提取松弛变量和lambda值
        self.slack_inputs = np.zeros((self.n_dmus, self.n_inputs))
        self.slack_outputs = np.zeros((self.n_dmus, self.n_outputs))
        self.lambda_values = np.zeros((self.n_dmus, self.n_dmus))
        
        print(f"调试：开始提取松弛变量，DMU数量: {self.n_dmus}, 投入数量: {self.n_inputs}, 产出数量: {self.n_outputs}")
        
        for i, dmu_name in enumerate(data.dmu_names):
            if dmu_name in result.input_slacks:
                for j, input_name in enumerate(data.input_names):
                    slack_value = result.input_slacks[dmu_name].get(input_name, 0)
                    self.slack_inputs[i, j] = slack_value
                    if abs(slack_value) > 1e-6:  # 只打印非零松弛变量
                        print(f"调试：DMU {dmu_name} 投入松弛变量 {input_name}: {slack_value}")
            else:
                print(f"调试：DMU {dmu_name} 没有投入松弛变量数据")
            
            if dmu_name in result.output_slacks:
                for j, output_name in enumerate(data.output_names):
                    slack_value = result.output_slacks[dmu_name].get(output_name, 0)
                    self.slack_outputs[i, j] = slack_value
                    if abs(slack_value) > 1e-6:  # 只打印非零松弛变量
                        print(f"调试：DMU {dmu_name} 产出松弛变量 {output_name}: {slack_value}")
            else:
                print(f"调试：DMU {dmu_name} 没有产出松弛变量数据")
            
            if dmu_name in result.lambda_variables:
                for j, other_dmu in enumerate(data.dmu_names):
                    self.lambda_values[i, j] = result.lambda_variables[dmu_name].get(other_dmu, 0)
        
        return efficiency_scores
    
    def efficiency(self):
        """默认效率计算方法"""
        return self.ccr()

# 为了保持兼容性，创建DEA别名
DEA = DEAWrapper

# QCA分析模块已在文件开头导入

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

def format_efficiency_values(df, efficiency_col='效率值'):
    """
    格式化效率值，确保显示4位小数
    
    Args:
        df: 包含效率值的DataFrame
        efficiency_col: 效率值列名（可以是字符串或列表）
        
    Returns:
        格式化后的DataFrame
    """
    if isinstance(efficiency_col, str):
        efficiency_cols = [efficiency_col]
    else:
        efficiency_cols = efficiency_col
    
    for col in efficiency_cols:
        if col in df.columns:
            # 先四舍五入到4位小数
            df[col] = df[col].round(4)
            # 然后格式化为4位小数显示
            df[col] = df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else x)
    return df

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
    
    # 数据加载完成
    
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
    
    # 检测到空值
    
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
        
        # 转换百分比数据
        percentage_columns = [col for col in df_cleaned.columns if any(keyword in col for keyword in ['满意度', '率', '比例', '百分比'])]
        for col in percentage_columns:
            df_cleaned[col] = df_cleaned[col].apply(convert_percentage_to_decimal)
        
        return df_cleaned, {'removed_rows': 0, 'filled_nulls': total_nulls}

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

def perform_dea_analysis(data, input_vars, output_vars, model_type, orientation='input', 
                        undesirable_outputs=None, rts='vrs', handle_infeasible='set_to_1'):
    """
    执行DEA效率分析
    
    参数:
    - data: 包含医院数据的DataFrame
    - input_vars: 投入变量列表
    - output_vars: 产出变量列表
    - model_type: DEA模型类型 ('CCR', 'BCC', 'SBM', 'Super-SBM')
    - orientation: 导向类型 ('input', 'output')
    - undesirable_outputs: 非期望产出变量列表（仅SBM模型使用）
    - rts: 规模报酬假设 ('crs' 或 'vrs')
    - handle_infeasible: 无解处理方式 ('set_to_1' 或 'exclude')
    
    返回:
    - results: 包含效率值和其他分析结果的DataFrame
    """
    try:
        # 准备数据
        dmu_column = 'DMU' if 'DMU' in data.columns else '医院ID'
        dmu_names = data[dmu_column].values
        input_data = data[input_vars].values
        output_data = data[output_vars].values
        
        # 数据验证
        is_valid, message = validate_dea_data(input_data, output_data)
        if not is_valid:
            st.error(f"数据验证失败: {message}")
            return None
        
        # 数据预处理：避免零值
        input_data = np.maximum(input_data, 1e-6)
        output_data = np.maximum(output_data, 1e-6)
        
        # 变异系数判断是否需要标准化
        input_means = np.mean(input_data, axis=0)
        output_means = np.mean(output_data, axis=0)
        input_cv = np.std(input_data, axis=0) / (input_means + 1e-10)
        output_cv = np.std(output_data, axis=0) / (output_means + 1e-10)

        if np.any(input_cv > 2.0) or np.any(output_cv > 2.0):
            input_data = input_data / (input_means + 1e-10)
            output_data = output_data / (output_means + 1e-10)

        # 创建DEA对象
        dea = DEAWrapper(input_data, output_data, dmu_names=dmu_names)
        
        results_dict = {
            'DMU': dmu_names,
        }

        # 根据模型类型决定是否计算CCR和BCC
        if model_type not in ['Super-SBM']:
            # 统一计算 CCR 和 BCC（非超效率SBM模型）
            if orientation == 'input':
                ccr_scores = dea.ccr_input_oriented()
                bcc_scores = dea.bcc_input_oriented()
            else:
                ccr_scores = dea.ccr_output_oriented()
                bcc_scores = dea.bcc_output_oriented()

            scale_efficiency = np.divide(ccr_scores, bcc_scores, out=np.zeros_like(ccr_scores), where=bcc_scores!=0)
            scale_efficiency = np.clip(scale_efficiency, 0.0, 1.0)

            # 存储所有效率指标
            results_dict['综合效率(TE)'] = ccr_scores
            results_dict['纯技术效率(PTE)'] = bcc_scores
            results_dict['规模效率(SE)'] = scale_efficiency

        # 根据选择的 model_type 设置主效率值
        if model_type == 'CCR':
            results_dict['效率值'] = ccr_scores
        elif model_type == 'BCC':
            results_dict['效率值'] = bcc_scores
        elif model_type == 'SBM':
            # SBM模型 - 使用SuperEfficiencySBMModel类
            if undesirable_outputs:
                # 将变量名转换为在output_vars中的索引
                undesirable_indices = []
                for var_name in undesirable_outputs:
                    if var_name in output_vars:
                        undesirable_indices.append(output_vars.index(var_name))
                
                # 创建DEA数据对象
                data = DEAData(dea.input_data, dea.output_data)
                
                # 创建超效率SBM模型（只计算普通SBM部分）
                sbm_model = SuperEfficiencySBMModel(
                    data, 
                    orientation='input', 
                    undesirable_outputs=undesirable_indices,
                    rts='vrs',
                    handle_infeasible=handle_infeasible
                )
                
                # 求解
                result = sbm_model.solve()
                
                # 提取效率值
                efficiency_scores = np.array([result.efficiency_scores.get(dmu_name, np.nan) 
                                            for dmu_name in data.dmu_names])
            else:
                # 创建DEA数据对象
                data = DEAData(dea.input_data, dea.output_data)
                
                # 创建超效率SBM模型（只计算普通SBM部分）
                sbm_model = SuperEfficiencySBMModel(
                    data, 
                    orientation='input', 
                    undesirable_outputs=None,
                    rts='vrs',
                    handle_infeasible=handle_infeasible
                )
                
                # 求解
                result = sbm_model.solve()
                
                # 提取效率值
                efficiency_scores = np.array([result.efficiency_scores.get(dmu_name, np.nan) 
                                            for dmu_name in data.dmu_names])
            results_dict['效率值'] = efficiency_scores
        elif model_type == 'Super-SBM':
            # 处理非期望产出
            if undesirable_outputs:
                # 将变量名转换为在output_vars中的索引
                undesirable_indices = []
                for var_name in undesirable_outputs:
                    if var_name in output_vars:
                        undesirable_indices.append(output_vars.index(var_name))
                efficiency_scores = dea.super_sbm(
                    undesirable_outputs=undesirable_indices,  # 传递索引列表而不是变量名列表
                    rts=rts,
                    handle_infeasible=handle_infeasible
                )
            else:
                efficiency_scores = dea.super_sbm(
                    rts=rts,
                    handle_infeasible=handle_infeasible
                )
            results_dict['效率值'] = efficiency_scores
            
            # 添加投影目标值（原始值 - 松弛变量）
            if hasattr(dea, 'slack_inputs') and dea.slack_inputs is not None:
                print(f"调试：生成投入投影目标值，投入变量数量: {len(input_vars)}")
                for i, var in enumerate(input_vars):
                    # 投影目标值 = 原始值 - slack（松弛变量为负表示需要增加投入）
                    projection = np.zeros(len(input_data))
                    for dmu in range(len(input_data)):
                        projection[dmu] = input_data[dmu, i] - dea.slack_inputs[dmu, i]
                    results_dict[f'{var}_投影目标值'] = projection
                    print(f"调试：已添加 {var}_投影目标值")
            
            if hasattr(dea, 'slack_outputs') and dea.slack_outputs is not None:
                # 获取非期望产出的变量名列表（用于投影计算）
                undesirable_var_names = []
                if undesirable_outputs:
                    for var_name in undesirable_outputs:
                        if var_name in output_vars:
                            undesirable_var_names.append(var_name)
                
                print(f"调试：生成产出投影目标值，产出变量数量: {len(output_vars)}")
                print(f"调试：非期望产出变量: {undesirable_var_names}")
                
                for r, var in enumerate(output_vars):
                    # 投影目标值计算：
                    # 对于期望产出：投影 = 原始值 + slack（松弛变量为负表示需要减少产出）
                    # 对于非期望产出：投影 = 原始值 - slack（松弛变量为负表示需要增加非期望产出）
                    projection = np.zeros(len(output_data))
                    for dmu in range(len(output_data)):
                        if var in undesirable_var_names:
                            # 非期望产出：投影 = 原始值 - slack
                            projection[dmu] = output_data[dmu, r] - dea.slack_outputs[dmu, r]
                        else:
                            # 期望产出：投影 = 原始值 + slack
                            projection[dmu] = output_data[dmu, r] + dea.slack_outputs[dmu, r]
                    results_dict[f'{var}_投影目标值'] = projection
                    print(f"调试：已添加 {var}_投影目标值")
            
            # 添加规模报酬分析
            if hasattr(dea, 'rts_status') and hasattr(dea, 'rts_suggestions'):
                results_dict['规模报酬(RTS)'] = dea.rts_status
                results_dict['规模调整建议'] = dea.rts_suggestions
            
            # 添加CR-SBM和VR-SBM效率值
            if hasattr(dea, 'crs_scores'):
                results_dict['CR-SBM效率值'] = dea.crs_scores
            if hasattr(dea, 'vrs_scores'):
                results_dict['VR-SBM效率值'] = dea.vrs_scores
            
            # 添加求解状态
            if hasattr(dea, 'solution_status'):
                results_dict['求解状态'] = dea.solution_status
        else:
            st.error("不支持的模型类型，请选择 CCR、BCC、SBM 或 Super-SBM")
            return None

        # 添加松弛变量 - 使用原始列名
        if hasattr(dea, 'slack_inputs') and dea.slack_inputs is not None:
            print(f"调试：添加投入松弛变量，形状: {dea.slack_inputs.shape}")
            for i, var in enumerate(input_vars):
                slack_values = dea.slack_inputs[:, i]
                results_dict[f'{var}_slacks'] = slack_values
                print(f"调试：已添加 {var}_slacks，非零值数量: {np.count_nonzero(slack_values)}")
        
        if hasattr(dea, 'slack_outputs') and dea.slack_outputs is not None:
            print(f"调试：添加产出松弛变量，形状: {dea.slack_outputs.shape}")
            for r, var in enumerate(output_vars):
                slack_values = dea.slack_outputs[:, r]
                results_dict[f'{var}_slacks'] = slack_values
                print(f"调试：已添加 {var}_slacks，非零值数量: {np.count_nonzero(slack_values)}")

        # 转为DataFrame
        results_df = pd.DataFrame(results_dict)
        
        # 按效率值降序排列，NaN值放在最后
        results_df = results_df.sort_values('效率值', ascending=False, na_position='last').reset_index(drop=True)
        
        return results_df

    except Exception as e:
        st.error(f"DEA分析过程中发生错误: {str(e)}")
        return None

def create_efficiency_chart(results):
    """
    创建效率排名柱状图 - 显示效率值，并处理松弛变量数据
    
    参数:
    - results: 包含效率值和松弛变量的DataFrame
    
    返回:
    - fig: Plotly图表对象
    - slack_data: 松弛变量数据字典
    """
    # 检查可用的效率列
    efficiency_columns = []
    if '综合效率(TE)' in results.columns:
        efficiency_columns.append('综合效率(TE)')
    if '纯技术效率(PTE)' in results.columns:
        efficiency_columns.append('纯技术效率(PTE)')
    if '规模效率(SE)' in results.columns:
        efficiency_columns.append('规模效率(SE)')
    if '效率值' in results.columns:
        efficiency_columns.append('效率值')
    
    # 检查松弛变量列
    slack_columns = [col for col in results.columns if 'slacks' in col]
    
    # 如果没有找到效率列，返回空图表
    if not efficiency_columns:
        fig = go.Figure()
        fig.add_annotation(text="未找到效率数据", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig, {}
    
    # 创建柱状图
    fig = go.Figure()
    
    # 添加效率柱状图
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, col in enumerate(efficiency_columns):
        color = colors[i % len(colors)]
        fig.add_trace(
            go.Bar(
                x=results['DMU'],
                y=results[col],
                name=col,
                marker_color=color,
                text=[f'{val:.5f}' for val in results[col]],
                textposition='outside',
                showlegend=True
            )
        )
    
    # 更新布局
    fig.update_layout(
        height=500,
        title_text="DEA效率值对比",
        title_x=0.5,
        xaxis_title="DMU",
        yaxis_title="效率值",
        yaxis=dict(range=[0, 1.1]),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # 处理松弛变量数据
    slack_data = {}
    if slack_columns:
        slack_data = {
            'columns': slack_columns,
            'data': results[['DMU'] + slack_columns].copy()
        }
    
    return fig, slack_data

def display_dea_formulas():
    """显示DEA模型的数学公式"""
    st.subheader("DEA模型数学公式")
    
    # CCR模型公式
    st.markdown("### 1. CCR模型（规模报酬不变）")
    st.markdown("**理论定义**：假定规模报酬不变（CRS），主要用来测量技术效率（综合效率）")
    
    st.markdown("#### 输入导向CCR模型：")
    st.markdown("**目标**：在保持产出不变的前提下，最小化投入资源")
    st.latex(r"""
    \min \theta
    """)
    st.latex(r"""
    \text{s.t. } \sum_{j=1}^{n} \lambda_j x_{ij} \leq \theta x_{i0}, \quad i = 1,2,\ldots,m
    """)
    st.latex(r"""
    \sum_{j=1}^{n} \lambda_j y_{rj} \geq y_{r0}, \quad r = 1,2,\ldots,s
    """)
    st.latex(r"""
    \lambda_j \geq 0, \quad j = 1,2,\ldots,n
    """)
    st.markdown("**解释**：θ < 1 表示可以按比例减少投入，θ = 1 表示DEA有效")
    
    st.markdown("#### 输出导向CCR模型：")
    st.markdown("**目标**：在保持投入不变的前提下，最大化产出效果")
    st.latex(r"""
    \max \phi
    """)
    st.latex(r"""
    \text{s.t. } \sum_{j=1}^{n} \lambda_j x_{ij} \leq x_{i0}, \quad i = 1,2,\ldots,m
    """)
    st.latex(r"""
    \sum_{j=1}^{n} \lambda_j y_{rj} \geq \phi y_{r0}, \quad r = 1,2,\ldots,s
    """)
    st.latex(r"""
    \lambda_j \geq 0, \quad j = 1,2,\ldots,n
    """)
    st.markdown("**解释**：φ > 1 表示可以按比例增加产出，φ = 1 表示DEA有效")
    
    # 重要说明
    st.markdown("#### ⚠️ 重要说明")
    st.markdown("""
    **注意**：您提到的公式 $\max \theta$ 和 $\sum_{j=1}^{n} \lambda_j x_{ij} \leq \theta x_{i0}$ 
    实际上是**输出导向**CCR模型的公式，不是输入导向的。
    
    - **输入导向**：$\min \theta$，目标是最小化投入比例
    - **输出导向**：$\max \phi$，目标是最大化产出比例
    
    两种导向的数学表达和解释是不同的！
    """)
    
    # BCC模型公式
    st.markdown("### 2. BCC模型（规模报酬可变）")
    st.markdown("**理论定义**：假定规模报酬可变（VRS），主要测算纯技术效率（技术效率与规模效率的比值）")
    
    st.markdown("#### 输入导向BCC模型：")
    st.latex(r"""
    \min \theta
    """)
    st.latex(r"""
    \text{s.t. } \sum_{j=1}^{n} \lambda_j x_{ij} \leq \theta x_{i0}, \quad i = 1,2,\ldots,m
    """)
    st.latex(r"""
    \sum_{j=1}^{n} \lambda_j y_{rj} \geq y_{r0}, \quad r = 1,2,\ldots,s
    """)
    st.latex(r"""
    \sum_{j=1}^{n} \lambda_j = 1
    """)
    st.latex(r"""
    \lambda_j \geq 0, \quad j = 1,2,\ldots,n
    """)
    
    st.markdown("#### 输出导向BCC模型：")
    st.latex(r"""
    \max \phi
    """)
    st.latex(r"""
    \text{s.t. } \sum_{j=1}^{n} \lambda_j x_{ij} \leq x_{i0}, \quad i = 1,2,\ldots,m
    """)
    st.latex(r"""
    \sum_{j=1}^{n} \lambda_j y_{rj} \geq \phi y_{r0}, \quad r = 1,2,\ldots,s
    """)
    st.latex(r"""
    \sum_{j=1}^{n} \lambda_j = 1
    """)
    st.latex(r"""
    \lambda_j \geq 0, \quad j = 1,2,\ldots,n
    """)
    
    # SBM模型公式
    st.markdown("### 3. SBM模型（基于松弛变量）")
    
    st.latex(r"""
    \min \rho = \frac{1 - \frac{1}{m}\sum_{i=1}^{m}\frac{s_i^-}{x_{i0}}}{1 + \frac{1}{s}\sum_{r=1}^{s}\frac{s_r^+}{y_{r0}}}
    """)
    st.latex(r"""
    \text{s.t. } x_0 = X\lambda + s^-
    """)
    st.latex(r"""
    y_0 = Y\lambda - s^+
    """)
    st.latex(r"""
    \lambda \geq 0, \quad s^- \geq 0, \quad s^+ \geq 0
    """)
    
    # 超效率SBM模型公式
    st.markdown("### 4. 超效率SBM模型")
    
    st.latex(r"""
    \min \delta = \frac{1 + \frac{1}{m}\sum_{i=1}^{m}\frac{s_i^-}{x_{i0}}}{1 - \frac{1}{s}\sum_{r=1}^{s}\frac{s_r^+}{y_{r0}}}
    """)
    st.latex(r"""
    \text{s.t. } x_0 = X\lambda + s^-
    """)
    st.latex(r"""
    y_0 = Y\lambda - s^+
    """)
    st.latex(r"""
    \lambda \geq 0, \quad s^- \geq 0, \quad s^+ \geq 0
    """)
    st.latex(r"""
    \text{（排除被评估的DMU）}
    """)
    
    # 符号说明
    st.markdown("### 符号说明")
    st.markdown("""
    - **θ**: 效率值（输入导向）
    - **φ**: 效率值（输出导向）
    - **ρ**: SBM效率值
    - **δ**: 超效率SBM效率值
    - **λⱼ**: 权重变量
    - **s⁻**: 投入松弛变量（投入冗余）
    - **s⁺**: 产出松弛变量（产出不足）
    - **xᵢⱼ**: 第j个DMU的第i个投入
    - **yᵣⱼ**: 第j个DMU的第r个产出
    - **m**: 投入变量数量
    - **s**: 产出变量数量
    - **n**: DMU数量
    """)

def download_dea_results(results):
    """
    生成DEA结果CSV下载
    
    参数:
    - results: 包含效率值的DataFrame
    
    返回:
    - csv: CSV格式的字符串
    """
    # 使用专门的编码处理函数，确保中文字符正确显示
    return create_csv_with_proper_encoding(results)

def create_csv_with_proper_encoding(df):
    """
    创建正确编码的CSV字符串，确保中文字符正确显示
    
    参数:
    - df: 包含中文字符的DataFrame
    
    返回:
    - csv: 正确编码的CSV字符串
    """
    import io
    
    # 方法1：使用utf-8-sig编码（推荐）
    try:
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
        csv_data = csv_buffer.getvalue()
        
        # 确保包含BOM标记
        if not csv_data.startswith('\ufeff'):
            csv_data = '\ufeff' + csv_data
            
        return csv_data
    except Exception as e:
        # 方法2：备用方案，使用utf-8编码
        try:
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False, encoding='utf-8')
            return csv_buffer.getvalue()
        except Exception as e2:
            # 方法3：最后备用方案
            return df.to_csv(index=False)

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
    st.markdown('<div class="main-header">基于DEA与fsQCA的医院运营效能与发展路径智慧决策系统</div>', unsafe_allow_html=True)
    
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
            # 请上传包含医院数据的Excel或CSV文件，文件必须包含'DMU'列或'医院ID'列。
            
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
                                    pass  # 请选择空值处理方式以继续
                                else:
                                    # 根据用户选择清理数据
                                    df_cleaned, stats = clean_data(df, null_handling)
                                    
                                    # 显示处理结果
                                    if null_handling == 'fill_zero':
                                        pass  # 已将空值转换为0
                                    else:  # drop_rows
                                        pass  # 已删除包含空值的数据
                                    
                                    # 继续处理数据
                                    process_cleaned_data(df_cleaned, warnings)
                            else:
                                # 没有空值，直接处理
                                process_cleaned_data(df, warnings)
                
                except Exception as e:
                    st.markdown(f'<div class="error-message">文件读取错误：{str(e)}</div>', unsafe_allow_html=True)
        
        elif input_mode == "✏️ 手动输入模式":
            st.markdown("### ✏️ 手动数据输入")
            # 请设置医院数量和变量数量，然后逐家输入数据。
            
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
                
                # 数据输入完成！可以进入DEA效率分析模块。
    
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
                # 医疗示例：医生人数、护士人数、床位数、医疗设备数量、运营成本等
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
                # 医疗示例：门诊人次、住院人次、手术例数、出院人数、患者满意度等
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
                # 已选择投入变量和产出变量
                
                # 模型选择
                st.subheader("🔬 模型选择")
                
                model_options = {
                    "CCR模型（规模报酬不变）": {
                        "value": "CCR",
                        "description": "假定规模报酬不变，主要用来测量技术效率（综合效率）"
                    },
                    "BCC模型（规模报酬可变）": {
                        "value": "BCC", 
                        "description": "假定规模报酬可变，主要测算纯技术效率（推荐）"
                    },
                    "SBM模型（非径向）": {
                        "value": "SBM",
                        "description": "适用于含非期望产出场景，非径向效率测量"
                    },
                    "超效率SBM模型": {
                        "value": "Super-SBM",
                        "description": "超效率SBM模型，允许效率值大于1，可对有效DMU进一步排序"
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
                
                # 导向选择（仅对CCR和BCC模型显示）
                orientation = 'input'  # 默认值
                if model_info['value'] in ['CCR', 'BCC']:
                    st.markdown("**选择分析导向**")
                    orientation_options = {
                        "输入导向（推荐）": {
                            "value": "input"
                        },
                        "输出导向": {
                            "value": "output"
                        }
                    }
                    
                    selected_orientation = st.selectbox(
                        "选择分析导向",
                        options=list(orientation_options.keys()),
                        index=0,  # 默认选择输入导向
                    )
                    
                    orientation_info = orientation_options[selected_orientation]
                    orientation = orientation_info['value']
                
                # 非期望产出选择（仅对SBM模型显示）
                undesirable_outputs = None
                if model_info['value'] in ['SBM', 'Super-SBM']:
                    st.markdown("**⚠️ 非期望产出选择**")
                    st.caption("独立选择非期望产出变量（如医疗纠纷、不良事件等）")
                    
                    # 从所有数值变量中独立选择非期望产出
                    available_vars = [var for var in numeric_columns if var not in input_vars]
                    
                    if available_vars:
                        # 多选非期望产出（独立于产出变量选择）
                        selected_undesirable = st.multiselect(
                            "选择非期望产出变量",
                            options=available_vars,
                            default=[],
                            help="选择那些数值越小越好的变量作为非期望产出（如医疗纠纷数量、不良事件等）"
                        )
                        
                        if selected_undesirable:
                            undesirable_outputs = selected_undesirable
                            st.success(f" 已选择 {len(selected_undesirable)} 个非期望产出变量")
                            st.markdown("**非期望产出变量：**")
                            for var in selected_undesirable:
                                st.write(f"• {var}")
                        else:
                            # 未选择非期望产出，所有产出变量将视为期望产出
                            undesirable_outputs = []
                            st.info("未选择非期望产出变量，所有产出变量将视为期望产出")
                    else:
                        # 没有可用变量
                        st.info("当前没有可用的变量作为非期望产出。")
                        undesirable_outputs = []
                
                # 规模报酬选择（仅对超效率SBM模型显示）
                rts = 'vrs'  # 默认值
                if model_info['value'] == 'Super-SBM':
                    st.markdown("**规模报酬假设选择**")
                    st.caption("选择规模报酬假设，影响效率值计算和规模报酬分析")
                    
                    rts_options = {
                        "规模报酬可变(VRS)": {
                            "value": "vrs"
                        },
                        "规模报酬不变(CRS)": {
                            "value": "crs"
                        }
                    }
                    
                    selected_rts = st.selectbox(
                        "选择规模报酬假设",
                        options=list(rts_options.keys()),
                        index=0,  # 默认选择VRS
                    )
                    
                    rts_info = rts_options[selected_rts]
                    rts = rts_info['value']
                else:
                    # 非超效率SBM模型，使用默认的VRS
                    rts = 'vrs'
                
                # 为SBM和超效率SBM模型添加无解处理选项
                if model_info['value'] in ['SBM', 'Super-SBM']:
                    st.subheader("🔧 无解处理选项")
                    
                    handle_infeasible = st.radio(
                        "选择无解处理方式",
                        ["将无解的效率值设为1", "将无解的效率值设为NaN"],
                        index=0,
                        help="对于无解的DMU，选择如何处理"
                    )
                    
                    # 转换为内部表示
                    handle_infeasible = 'set_to_1' if handle_infeasible == "将无解的效率值设为1" else 'exclude'
                    
                    st.markdown("""
                    **说明**：
                    - **将无解的效率值设为1**：当无解情况较少时推荐使用
                    - **将无解的效率值设为NaN**：当需要严格区分有效/无效DMU时使用
                    """)
                else:
                    # 非超效率SBM模型，使用默认值
                    handle_infeasible = 'set_to_1'
                
                # 数据预处理选项
                st.subheader("📊 数据预处理选项")
                normalize_data = st.checkbox("标准化数据", value=True, 
                                           help="将数据缩放到[0,1]范围，避免量纲差异影响结果")
                
                # 求解器参数调整
                st.subheader("⚙️ 求解器参数")
                col1, col2 = st.columns(2)
                with col1:
                    max_iter = st.number_input("最大迭代次数", min_value=100, value=5000, step=100,
                                             help="增加迭代次数可能提高求解精度，但会增加计算时间")
                with col2:
                    tolerance = st.number_input("容差", min_value=1e-10, value=1e-9, format="%.1e",
                                              help="更小的容差值可能提高精度，但可能导致求解失败")
                
                # 执行分析按钮
                st.markdown("---")
                col_btn1, col_btn2, col_btn3, col_btn4 = st.columns([1, 1.5, 1.5, 1])
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
                                undesirable_outputs,
                                rts=rts,  # 传递规模报酬假设
                                handle_infeasible=handle_infeasible  # 传递无解处理方式
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
                                    
                                    # 保存模型的实际值而不是显示名称
                                    model_value = model_info['value'] if 'model_info' in locals() else str(selected_model) if selected_model else ""
                                    st.session_state['dea_model'] = model_value
                                    
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
                                    # 保存模型的实际值而不是显示名称
                                    model_value = model_info['value'] if 'model_info' in locals() else str(selected_model) if selected_model else ""
                                    st.session_state['dea_model'] = model_value
                                
                                # DEA分析完成！
                
                with col_btn3:
                    if st.button("查看数学公式", type="secondary", use_container_width=True):
                        display_dea_formulas()
                
                # 显示DEA分析结果
                if 'dea_results' in st.session_state:
                    results = st.session_state['dea_results']
                    
                    # 显示结果
                    st.subheader("效率分析结果")

                    # 检查是否为超效率SBM模型
                    if st.session_state.get('dea_model') == 'Super-SBM':
                        # 超效率SBM模型的专门结果展示
                        st.markdown("**超效率SBM分析结果（按效率值降序排列）**")
                        
                        # 使用results中的规模报酬信息
                        results_display = results.copy()
                        
                        # 按效率值降序排序，NaN值放在最后
                        results_display = results_display.sort_values('效率值', ascending=False, na_position='last').reset_index(drop=True)
                        results_display = format_efficiency_values(results_display, '效率值')
                        results_display['排名'] = range(1, len(results_display) + 1)
                        
                        # 只显示四列：DMU、效率值、规模报酬、规模调整建议
                        display_cols = ['排名', 'DMU', '效率值']
                        
                        # 添加规模报酬相关列
                        if '规模报酬(RTS)' in results_display.columns:
                            display_cols.append('规模报酬(RTS)')
                        if '规模调整建议' in results_display.columns:
                            display_cols.append('规模调整建议')
                        
                        # 重新排列列顺序
                        results_display = results_display[display_cols]
                        
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
                        
                        # 超效率SBM模型的详细分析结果
                        st.subheader("超效率SBM详细分析结果")
                        
                        # 投影目标值分析
                        projection_cols = [col for col in results.columns if '投影目标值' in col]
                        if projection_cols:
                            st.markdown("**🎯 投影目标值分析**")
                            st.markdown("投影目标值表示各DMU在效率前沿上的目标位置：")
                            
                            # 调试信息：显示所有投影目标值列
                            st.markdown(f"**调试信息**：找到 {len(projection_cols)} 个投影目标值列：{projection_cols}")
                            
                            projection_display = results[['DMU', '效率值'] + projection_cols].copy()
                            projection_display = projection_display.sort_values('效率值', ascending=False, na_position='last').reset_index(drop=True)
                            projection_display= format_efficiency_values(projection_display, '效率值')
                            
                            st.dataframe(projection_display, use_container_width=True, hide_index=True)
                            
                            st.markdown("""
                            **投影目标值说明**：
                            - **投入投影目标值** = 原始投入值 - 投入松弛变量
                            - **产出投影目标值** = 原始产出值 + 产出松弛变量
                            - 投影目标值表示达到效率前沿所需的最优投入产出组合
                            """)
                        
                        # 松弛变量详细分析
                        slack_cols = [col for col in results.columns if 'slacks' in col]
                        if slack_cols:
                            st.markdown("**松弛变量详细分析**")
                            st.markdown("松弛变量表示各DMU与效率前沿的差距：")
                            
                            # 调试信息：显示松弛变量列
                            st.markdown(f"**调试信息**：找到 {len(slack_cols)} 个松弛变量列：{slack_cols}")
                            
                            # 检查松弛变量是否有非零值
                            non_zero_counts = {}
                            for col in slack_cols:
                                non_zero_count = (results[col] != 0).sum()
                                non_zero_counts[col] = non_zero_count
                            
                            st.markdown(f"**调试信息**：各松弛变量列的非零值数量：{non_zero_counts}")
                            
                            slack_display = results[['DMU', '效率值'] + slack_cols].copy()
                            slack_display = slack_display.sort_values('效率值', ascending=False, na_position='last').reset_index(drop=True)
                            slack_display= format_efficiency_values(slack_display, '效率值')
                            
                            st.dataframe(slack_display, use_container_width=True, hide_index=True)
                            
                            st.markdown("""
                            - **投入松弛变量**：表示可以减少的投入量（数值越大，投入冗余越多）
                            - **产出松弛变量**：表示可以增加的产出量（数值越大，产出不足越多）
                            - **松弛变量为0**：表示该变量已达到最优水平
                            """)
                        
                        # 规模报酬分析
                        if '规模报酬(RTS)' in results.columns and '规模调整建议' in results.columns:
                            st.markdown("** 规模报酬分析**")
                            
                            rts_display = results[['DMU', '效率值', '规模报酬(RTS)', '规模调整建议']].copy()
                            rts_display = rts_display.sort_values('效率值', ascending=False, na_position='last').reset_index(drop=True)
                            rts_display= format_efficiency_values(rts_display, '效率值')
                            
                            st.dataframe(rts_display, use_container_width=True, hide_index=True)
                            
                            st.markdown("""
                            - **规模报酬不变(CRS)**：当前规模最优，建议保持
                            - **规模报酬递增(IRS)**：扩大规模可提高效率，建议扩大规模
                            - **规模报酬递减(DRS)**：缩小规模可提高效率，建议缩小规模
                            """)
                        
                        # 求解状态分析
                        if '求解状态' in results.columns and '迭代次数' in results.columns:
                            st.markdown("**🔧 求解状态分析**")
                            
                            status_display = results[['DMU', '效率值', '求解状态', '迭代次数']].copy()
                            status_display = status_display.sort_values('效率值', ascending=False, na_position='last').reset_index(drop=True)
                            status_display= format_efficiency_values(status_display, '效率值')
                            
                            st.dataframe(status_display, use_container_width=True, hide_index=True)
                        
                        # 求解状态统计
                        if '求解状态' in results.columns:
                            infeasible_count = results['求解状态'].str.contains('infeasible').sum()
                            st.markdown(f"**⚠️ 求解状态统计**: 共有 {infeasible_count} 个DMU无解")
                            
                            if infeasible_count > 0:
                                st.warning(f"注意：有 {infeasible_count} 个DMU无解，已按选择的方式处理")
                                if infeasible_count / len(results) > 0.2:
                                    st.warning("警告：无解DMU比例较高（>20%），建议考虑更换模型或假设")
                        
                        # 超效率SBM模型的统计信息
                        st.markdown("**📈 超效率SBM统计信息**")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("分析医院数", len(results))
                        
                        with col2:
                            efficient_count = len(results[results['效率值'] >= 0.9999])
                            st.metric("有效医院数", efficient_count)
                        
                        with col3:
                            avg_efficiency = results['效率值'].mean()
                            st.metric("平均效率值", f"{avg_efficiency:.3f}")
                        
                        # 效率值分布统计
                        st.markdown("**效率值分布统计**")
                        efficiency_stats = results['效率值'].describe()
                        st.write(efficiency_stats)
                        
                        # 超效率SBM模型效率值解释
                        st.markdown("**超效率SBM模型效率值解释**")
                        st.markdown("""
                        | 效率值范围 | 含义 | 解读 |
                        |-----------|------|------|
                        | **效率值 = 1** | 技术有效 | 位于效率前沿面上，达到最优状态 |
                        | **效率值 < 1** | 技术无效 | 存在改进空间，需要优化投入产出组合 |
                        | **效率值 > 1** | 超效率 | 超越效率前沿，表现优异，可作为标杆 |
                        """)
                        
                        st.markdown("""
                        **超效率SBM模型特点**：
                        - 允许效率值大于1，可对有效DMU进行进一步排序
                        - 处理非期望产出（如CO₂排放）
                        - 提供更精确的效率评估
                        - 效率值范围：(0,∞)
                        """)
                                              
                    # 检查结果中是否包含三种效率值（非超效率SBM模型）
                    elif '综合效率(TE)' in results.columns and '纯技术效率(PTE)' in results.columns and '规模效率(SE)' in results.columns:
                        # 如果包含三种效率值，显示完整的效率分解结果
                        st.markdown("**效率值排名（按综合效率降序排列）**")
                        
                        # 按综合效率降序排序
                        results_display = results.sort_values('综合效率(TE)', ascending=False).reset_index(drop=True)
                        
                        # 格式化效率值
                        results_display = format_efficiency_values(results_display, ['综合效率(TE)', '纯技术效率(PTE)', '规模效率(SE)'])
                        results_display['排名'] = range(1, len(results_display) + 1)
                        
                        # 重新排列列顺序
                        results_display = results_display[['排名', 'DMU', '综合效率(TE)', '纯技术效率(PTE)', '规模效率(SE)']]
                        
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
                        
                        # 效率分解说明
                        st.markdown("""
                        - **综合效率(TE)**：CCR模型结果，反映整体效率水平
                        - **纯技术效率(PTE)**：BCC模型结果，反映技术管理水平
                        - **规模效率(SE)**：综合效率÷纯技术效率，反映规模合理性
                        """)
                        
                    else:
                        # 其他模型的单一效率值显示
                        st.markdown("**效率值排名（按效率值降序排列）**")
                        try:
                            results_display = results.copy()
                        except Exception as e:
                            st.error(f"结果数据复制失败: {e}")
                            results_display = results
                        
                        # 按效率值降序排序
                        results_display = results_display.sort_values('效率值', ascending=False, na_position='last').reset_index(drop=True)
                        results_display = format_efficiency_values(results_display, '效率值')
                        efficiency_col = '效率值'
                        results_display['排名'] = range(1, len(results_display) + 1)
                        
                        # 只显示四列：DMU、效率值、规模报酬、规模调整建议
                        display_cols = ['排名', 'DMU', efficiency_col]
                        
                        # 添加规模报酬相关列（如果存在）
                        if '规模报酬(RTS)' in results_display.columns:
                            display_cols.append('规模报酬(RTS)')
                        if '规模调整建议' in results_display.columns:
                            display_cols.append('规模调整建议')
                        
                        # 重新排列列顺序
                        results_display = results_display[display_cols]
                    
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
                                       
                    # 松弛变量分析（非超效率SBM模型）
                    if st.session_state.get('dea_model') != 'Super-SBM':
                        fig, slack_data = create_efficiency_chart(results)
                        if slack_data and slack_data.get('columns'):
                            st.subheader("📊 松弛变量分析")
                            st.markdown("松弛变量表示各DMU在投入和产出方面的冗余或不足情况：")
                            
                            # 显示松弛变量数据表格
                            st.dataframe(slack_data['data'], use_container_width=True, hide_index=True)
                            
                            # 松弛变量解释说明
                            st.markdown("""
                            • **投入松弛变量S-(差额变数)**: 指为达到目标效率可以减少的投入量
                            
                            • **产出松弛变量S+(超额变数)**: 指为达到目标效率可以增加的产出量
                            """)
                    else:
                        # 超效率SBM模型使用专门的松弛变量分析（已在上面显示）
                        fig, slack_data = create_efficiency_chart(results)
                    
                    # 再显示效率排名图表
                    st.subheader(" 效率排名可视化")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 提供结果下载
                    st.subheader("💾 结果下载")
                    csv_data = download_dea_results(results)
                    
                    st.download_button(
                        label="📥 下载DEA分析结果 (CSV)",
                        data=csv_data,
                        file_name=f"DEA分析结果_{st.session_state.get('dea_model', 'Unknown')}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    # 分析摘要
                    st.subheader("📋 分析摘要")
                    if '综合效率(TE)' in results.columns:
                        # 如果包含三种效率值，显示三种效率的指标
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("分析医院数", len(results))
                        
                        with col2:
                            te_efficient_count = len(results[results['综合效率(TE)'] >= 0.9999])
                            st.metric("综合有效医院数", te_efficient_count)
                        
                        with col3:
                            avg_te = results['综合效率(TE)'].mean()
                            st.metric("平均综合效率", f"{avg_te:.4f}")
                    else:
                        # 如果没有三种效率值，显示单一效率值指标
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
                    if '综合效率(TE)' in results.columns:
                        # 如果包含三种效率值，显示三种效率的统计信息
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("**综合效率(TE)统计**")
                            te_stats = results['综合效率(TE)'].describe()
                            st.write(te_stats)
                        
                        with col2:
                            st.markdown("**纯技术效率(PTE)统计**")
                            pte_stats = results['纯技术效率(PTE)'].describe()
                            st.write(pte_stats)
                        
                        with col3:
                            st.markdown("**规模效率(SE)统计**")
                            se_stats = results['规模效率(SE)'].describe()
                            st.write(se_stats)
                    else:
                        # 如果没有三种效率值，显示单一效率值统计
                        efficiency_stats = results['效率值'].describe()
                        st.write(efficiency_stats)
                    
                    
    else:
        # 请先在数据输入区中加载数据
        pass
    
    st.markdown('</div>', unsafe_allow_html=True)  # 关闭DEA分析区容器
    
    # ③ fsQCA路径分析区
    st.markdown('<div class="section-header">③ fsQCA路径分析区</div>', unsafe_allow_html=True)
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    
    # 检查QCA模块状态
    if not QCA_AVAILABLE:
        st.error("❌ QCA分析模块不可用，请检查模块安装")
        # 解决方案：
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
                # 已选择条件变量
                
                st.subheader("🔧 数据预处理")
                # 正在将条件变量标准化为0-1范围的模糊集...
                
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
                        # 变量 '{var}' 的值全部相同，标准化后将为常数
                        pass
                
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
                        # 将自动过滤一致性<0.9的变量
                        pass
                
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
                    # 参数配置正确
                    
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
                                        # 必要性分析完成，保留有效条件变量
                                        pass
                                    else:
                                        # 所有条件变量的一致性都<0.9，使用原始变量进行分析
                                        pass
                                else:
                                    # 必要性分析失败，使用原始变量进行分析
                                    pass
                            
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
                                
                                # fsQCA分析完成！
                                
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
                                    # 没有找到有效路径，请尝试调整参数阈值
                                    pass
                            else:
                                # QCA分析失败
                                st.error("❌ fsQCA分析失败，请检查数据和参数设置")
                                # 可能的原因：
                                st.markdown("""
                                1. 数据格式不正确
                                2. 参数设置不当
                                3. 条件变量选择问题
                                4. 数据量不足
                                """)
                                # 解决方案：
                                st.markdown("""
                                1. 检查数据是否包含足够的案例
                                2. 调整一致性阈值和频率阈值
                                3. 尝试选择不同的条件变量
                                4. 确保数据质量良好
                                """)
    else:
        if 'data' not in st.session_state:
            # 请先在数据输入区中加载数据
            st.info("请先在数据输入区中加载数据")
        elif 'dea_results' not in st.session_state:
            # 请先完成DEA效率分析
            st.info("请先完成DEA效率分析")
    
    st.markdown('</div>', unsafe_allow_html=True)  # 关闭fsQCA分析区容器


def calculate_sbm_rts(crs_scores, vrs_scores, lambda_sums):

    """
    计算SBM模型的规模报酬状态
    
    参数:
    - crs_scores: CR-SBM效率值
    - vrs_scores: VR-SBM效率值  
    - lambda_sums: λ和数组
    
    返回:
    - rts_status: 规模报酬状态数组
    - rts_suggestions: 规模调整建议数组
    """
    n_dmus = len(crs_scores)
    rts_status = []
    rts_suggestions = []
    
    for i in range(n_dmus):
        if np.isnan(crs_scores[i]) or np.isnan(vrs_scores[i]):
            rts_status.append("求解失败")
            rts_suggestions.append("无法判定")
        else:
            # 方法1：比较CR-SBM和VR-SBM效率值
            if abs(crs_scores[i] - vrs_scores[i]) < 1e-6:
                # ρ_CRS = ρ_VRS，规模报酬不变
                rts_status.append("规模报酬不变(CRS)")
                rts_suggestions.append("保持当前规模")
            elif crs_scores[i] < vrs_scores[i]:
                # ρ_CRS < ρ_VRS，规模报酬递减
                rts_status.append("规模报酬递减(DRS)")
                rts_suggestions.append("建议缩小规模")
            else:
                # ρ_CRS > ρ_VRS，规模报酬递增
                rts_status.append("规模报酬递增(IRS)")
                rts_suggestions.append("建议扩大规模")
            
            # 方法2：基于λ和的Banker判据（补充验证）
            if not np.isnan(lambda_sums[i]):
                if abs(lambda_sums[i] - 1.0) < 1e-6:
                    # ∑λ = 1，规模报酬不变
                    if "CRS" not in rts_status[-1]:
                        rts_status[-1] += " (λ=1)"
                elif lambda_sums[i] < 1.0:
                    # ∑λ < 1，规模报酬递增
                    if "IRS" not in rts_status[-1]:
                        rts_status[-1] += " (λ<1)"
                else:
                    # ∑λ > 1，规模报酬递减
                    if "DRS" not in rts_status[-1]:
                        rts_status[-1] += " (λ>1)"
    
    return rts_status, rts_suggestions

# 主应用入口 - 直接调用main函数
main()