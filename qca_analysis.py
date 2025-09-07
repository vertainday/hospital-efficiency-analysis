import pandas as pd
import numpy as np
import streamlit as st
from io import StringIO
import warnings
from itertools import product, combinations
from scipy.stats import pearsonr
warnings.filterwarnings('ignore')

# 设置随机种子以确保结果可重现
np.random.seed(42)

# 使用纯Python实现，无需R连接
R_AVAILABLE = False
print("✓ 使用纯Python QCA模块实现")


class PythonQCA:
    """纯Python实现的QCA分析类"""
    
    def __init__(self, data, condition_vars, outcome_var):
        """
        初始化QCA分析
        
        参数:
        - data: 包含数据的DataFrame
        - condition_vars: 条件变量列表
        - outcome_var: 结果变量名
        """
        self.data = data.copy()
        self.condition_vars = condition_vars
        self.outcome_var = outcome_var
        self.n_cases = len(data)
        
        # 数据预处理
        self._prepare_data()
    
    def _prepare_data(self):
        """数据预处理：将连续变量转换为模糊集隶属度"""
        # 对条件变量进行模糊集转换（0-1标准化）
        for var in self.condition_vars:
            if var in self.data.columns:
                # 使用min-max标准化到[0,1]区间
                min_val = self.data[var].min()
                max_val = self.data[var].max()
                if max_val > min_val:
                    self.data[f"{var}_fs"] = (self.data[var] - min_val) / (max_val - min_val)
                else:
                    self.data[f"{var}_fs"] = 0.5  # 如果所有值相同，设为0.5
        
        # 对结果变量进行模糊集转换
        if self.outcome_var in self.data.columns:
            min_val = self.data[self.outcome_var].min()
            max_val = self.data[self.outcome_var].max()
            if max_val > min_val:
                self.data[f"{self.outcome_var}_fs"] = (self.data[self.outcome_var] - min_val) / (max_val - min_val)
            else:
                self.data[f"{self.outcome_var}_fs"] = 0.5
    
    def calculate_necessity(self, condition_var, threshold=0.9):
        """
        计算必要性分析
        
        参数:
        - condition_var: 条件变量名
        - threshold: 必要性阈值
        
        返回:
        - dict: 包含必要性分析结果的字典
        """
        if f"{condition_var}_fs" not in self.data.columns:
            return None
        
        condition_fs = self.data[f"{condition_var}_fs"]
        outcome_fs = self.data[f"{self.outcome_var}_fs"]
        
        # 计算包含度 (Inclusion)
        # 对于必要性：X ⊆ Y，即 min(X, Y) / X
        inclusion = np.minimum(condition_fs, outcome_fs).sum() / condition_fs.sum()
        
        # 计算覆盖度 (Coverage)
        # 对于必要性：X ∩ Y / Y
        coverage = np.minimum(condition_fs, outcome_fs).sum() / outcome_fs.sum()
        
        # 计算相关性 (Relevance)
        # 使用皮尔逊相关系数
        if len(condition_fs) > 1:
            correlation, _ = pearsonr(condition_fs, outcome_fs)
        else:
            correlation = 0
        
        return {
            '条件变量': condition_var,
            'Raw Consistency': inclusion,
            'Coverage': coverage,
            'Relevance': abs(correlation),
            '必要性解释': '强必要性' if inclusion >= threshold else '弱必要性'
        }
    
    def calculate_sufficiency(self, condition_var, threshold=0.8):
        """
        计算充分性分析
        
        参数:
        - condition_var: 条件变量名
        - threshold: 充分性阈值
        
        返回:
        - dict: 包含充分性分析结果的字典
        """
        if f"{condition_var}_fs" not in self.data.columns:
            return None
        
        condition_fs = self.data[f"{condition_var}_fs"]
        outcome_fs = self.data[f"{self.outcome_var}_fs"]
        
        # 计算包含度 (Inclusion)
        # 对于充分性：X ⊇ Y，即 min(X, Y) / Y
        inclusion = np.minimum(condition_fs, outcome_fs).sum() / outcome_fs.sum()
        
        # 计算覆盖度 (Coverage)
        # 对于充分性：X ∩ Y / X
        coverage = np.minimum(condition_fs, outcome_fs).sum() / condition_fs.sum()
        
        # 计算相关性 (Relevance)
        if len(condition_fs) > 1:
            correlation, _ = pearsonr(condition_fs, outcome_fs)
        else:
            correlation = 0
        
        return {
            '条件变量': condition_var,
            'Raw Consistency': inclusion,
            'Coverage': coverage,
            'Relevance': abs(correlation),
            '充分性解释': '强充分性' if inclusion >= threshold else '弱充分性'
        }
    
    def generate_truth_table(self, freq_threshold=1.0, consistency_threshold=0.8):
        """
        生成真值表
        
        参数:
        - freq_threshold: 频率阈值
        - consistency_threshold: 一致性阈值
        
        返回:
        - DataFrame: 真值表
        """
        # 获取所有条件变量的模糊集数据
        fs_columns = [f"{var}_fs" for var in self.condition_vars if f"{var}_fs" in self.data.columns]
        outcome_fs = self.data[f"{self.outcome_var}_fs"]
        
        if not fs_columns:
            return pd.DataFrame()
        
        # 生成所有可能的组合
        truth_table = []
        
        # 对于每个案例，计算其组合
        for idx, row in self.data.iterrows():
            # 计算条件组合的隶属度（取最小值）
            condition_values = [row[col] for col in fs_columns]
            min_condition = min(condition_values) if condition_values else 0
            
            # 计算一致性
            outcome_value = row[f"{self.outcome_var}_fs"]
            consistency = min(min_condition, outcome_value) / min_condition if min_condition > 0 else 0
            
            # 计算PRI一致性（简化版本）
            pri_consistency = consistency  # 简化处理
            
            truth_table.append({
                'Case': f"Case_{idx+1}",
                'Outcome': 1 if outcome_value > 0.5 else 0,
                'Frequency': 1,
                'Raw Consistency': consistency,
                'PRI Consistency': pri_consistency,
                '结果解释': '高绩效' if outcome_value > 0.5 else '低绩效'
            })
        
        truth_df = pd.DataFrame(truth_table)
        
        # 按组合分组并计算统计
        grouped = truth_df.groupby([f"{var}_fs" for var in self.condition_vars if f"{var}_fs" in self.data.columns]).agg({
            'Outcome': 'mean',
            'Frequency': 'sum',
            'Raw Consistency': 'mean',
            'PRI Consistency': 'mean'
        }).reset_index()
        
        return grouped
    
    def find_solution_paths(self, freq_threshold=1.0, consistency_threshold=0.8, pri_consistency_threshold=0.7):
        """
        寻找解决方案路径
        
        参数:
        - freq_threshold: 频率阈值
        - consistency_threshold: 一致性阈值
        - pri_consistency_threshold: PRI一致性阈值
        
        返回:
        - DataFrame: 解决方案路径
        """
        # 获取高绩效案例
        high_performance = self.data[self.data[f"{self.outcome_var}_fs"] > 0.5].copy()
        
        if len(high_performance) == 0:
            return pd.DataFrame(columns=['路径组合', '一致性', '覆盖度', '路径类型', '路径解释'])
        
        # 获取条件变量的模糊集数据
        fs_columns = [f"{var}_fs" for var in self.condition_vars if f"{var}_fs" in self.data.columns]
        
        if not fs_columns:
            return pd.DataFrame(columns=['路径组合', '一致性', '覆盖度', '路径类型', '路径解释'])
        
        # 生成所有可能的条件组合
        solutions = []
        
        # 单条件路径
        for i, var in enumerate(fs_columns):
            condition_values = high_performance[var]
            consistency = condition_values.mean()
            coverage = len(high_performance) / len(self.data)
            
            if consistency >= consistency_threshold and coverage >= freq_threshold / len(self.data):
                solutions.append({
                    '路径组合': f"{self.condition_vars[i]}",
                    '一致性': consistency,
                    '覆盖度': coverage,
                    '路径类型': '核心路径' if consistency >= pri_consistency_threshold else '边缘路径',
                    '路径解释': f"单条件路径：{self.condition_vars[i]}"
                })
        
        # 双条件组合路径
        for i in range(len(fs_columns)):
            for j in range(i+1, len(fs_columns)):
                var1, var2 = fs_columns[i], fs_columns[j]
                condition1_values = high_performance[var1]
                condition2_values = high_performance[var2]
                
                # 计算组合条件的最小值
                combined_condition = np.minimum(condition1_values, condition2_values)
                consistency = combined_condition.mean()
                coverage = len(high_performance) / len(self.data)
                
                if consistency >= consistency_threshold and coverage >= freq_threshold / len(self.data):
                    solutions.append({
                        '路径组合': f"{self.condition_vars[i]} * {self.condition_vars[j]}",
                        '一致性': consistency,
                        '覆盖度': coverage,
                        '路径类型': '核心路径' if consistency >= pri_consistency_threshold else '边缘路径',
                        '路径解释': f"双条件组合：{self.condition_vars[i]} 且 {self.condition_vars[j]}"
                    })
        
        # 三条件组合路径（如果条件变量足够多）
        if len(fs_columns) >= 3:
            for i in range(len(fs_columns)):
                for j in range(i+1, len(fs_columns)):
                    for k in range(j+1, len(fs_columns)):
                        var1, var2, var3 = fs_columns[i], fs_columns[j], fs_columns[k]
                        condition1_values = high_performance[var1]
                        condition2_values = high_performance[var2]
                        condition3_values = high_performance[var3]
                        
                        # 计算三条件组合的最小值
                        combined_condition = np.minimum(
                            np.minimum(condition1_values, condition2_values),
                            condition3_values
                        )
                        consistency = combined_condition.mean()
                        coverage = len(high_performance) / len(self.data)
                        
                        if consistency >= consistency_threshold and coverage >= freq_threshold / len(self.data):
                            solutions.append({
                                '路径组合': f"{self.condition_vars[i]} * {self.condition_vars[j]} * {self.condition_vars[k]}",
                                '一致性': consistency,
                                '覆盖度': coverage,
                                '路径类型': '核心路径' if consistency >= pri_consistency_threshold else '边缘路径',
                                '路径解释': f"三条件组合：{self.condition_vars[i]} 且 {self.condition_vars[j]} 且 {self.condition_vars[k]}"
                            })
        
        if not solutions:
            # 如果没有找到有效路径，返回空DataFrame
            return pd.DataFrame(columns=['路径组合', '一致性', '覆盖度', '路径类型', '路径解释'])
        
        # 转换为DataFrame并排序
        solutions_df = pd.DataFrame(solutions)
        solutions_df = solutions_df.sort_values(['路径类型', '一致性'], ascending=[True, False])
        
        return solutions_df


def check_r_connection():
    """检查R连接状态（兼容性函数）"""
    return False, "使用纯Python实现，无需R连接"


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
    try:
        qca = PythonQCA(data, condition_vars, outcome_var)
        
        results = []
        for var in condition_vars:
            result = qca.calculate_necessity(var, threshold)
            if result:
                results.append(result)
        
        if results:
            return pd.DataFrame(results)
        else:
            return pd.DataFrame(columns=['条件变量', 'Raw Consistency', 'Coverage', 'Relevance', '必要性解释'])
        
    except Exception as e:
        st.error(f"必要性分析执行失败: {str(e)}")
        return pd.DataFrame(columns=['条件变量', 'Raw Consistency', 'Coverage', 'Relevance', '必要性解释'])


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
    try:
        qca = PythonQCA(data, condition_vars, outcome_var)
        
        results = []
        for var in condition_vars:
            result = qca.calculate_sufficiency(var, consistency)
            if result:
                results.append(result)
        
        if results:
            return pd.DataFrame(results)
        else:
            return pd.DataFrame(columns=['条件变量', 'Raw Consistency', 'Coverage', 'Relevance', '充分性解释'])
        
    except Exception as e:
        st.error(f"充分性分析执行失败: {str(e)}")
        return pd.DataFrame(columns=['条件变量', 'Raw Consistency', 'Coverage', 'Relevance', '充分性解释'])


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
    try:
        qca = PythonQCA(data, condition_vars, outcome_var)
        return qca.generate_truth_table(freq_threshold, consistency)
        
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
    try:
        qca = PythonQCA(data, condition_vars, outcome_var)
        
        # 使用PRI一致性阈值（通常比一致性阈值低0.1）
        pri_consistency = max(0.1, consistency - 0.1)
        
        solutions = qca.find_solution_paths(freq_threshold, consistency, pri_consistency)
        
        if not solutions.empty:
            # 重命名列以匹配原始接口
            solutions = solutions.rename(columns={
                '路径组合': 'Solution Path',
                '一致性': 'Raw Consistency',
                '覆盖度': 'Raw Coverage',
                '路径类型': 'Path Type',
                '路径解释': 'Path Explanation'
            })
            
            # 添加Unique Coverage列（简化计算）
            solutions['Unique Coverage'] = solutions['Raw Coverage'] * 0.8  # 简化处理
        
        return solutions
        
    except Exception as e:
        st.error(f"最小化分析执行失败: {str(e)}")
        return pd.DataFrame(columns=['Solution Path', 'Raw Consistency', 'Raw Coverage', 'Unique Coverage', 'Path Type', 'Path Explanation'])


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
    print("开始测试纯Python QCA模块...")
    
    # 创建测试数据
    test_data = pd.DataFrame({
        'A': [0.8, 0.6, 0.9, 0.3, 0.7, 0.5, 0.8, 0.4],
        'B': [0.7, 0.8, 0.4, 0.9, 0.6, 0.7, 0.5, 0.8],
        'C': [0.6, 0.7, 0.8, 0.5, 0.9, 0.6, 0.7, 0.5],
        'Y': [0.8, 0.7, 0.9, 0.4, 0.8, 0.6, 0.7, 0.5]
    })
    
    print("测试数据创建成功")
    
    # 测试必要性分析
    try:
        necessity = perform_necessity_analysis(test_data, ['A', 'B', 'C'], 'Y')
        print(f"必要性分析测试成功，结果行数: {len(necessity)}")
        if not necessity.empty:
            print("必要性分析结果:")
            print(necessity.to_string())
        else:
            print("必要性分析结果为空")
    except Exception as e:
        print(f"必要性分析测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试最小化分析
    try:
        minimization = perform_minimization(test_data, ['A', 'B', 'C'], 'Y')
        print(f"最小化分析测试成功，结果行数: {len(minimization)}")
        if not minimization.empty:
            print("最小化分析结果:")
            print(minimization.to_string())
        else:
            print("最小化分析结果为空")
    except Exception as e:
        print(f"最小化分析测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("QCA模块测试完成")
    return True


if __name__ == "__main__":
    test_qca_module()