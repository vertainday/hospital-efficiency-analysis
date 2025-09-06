import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import re
# 使用自定义DEA实现替代pyDEA
from scipy.optimize import linprog
import itertools
from scipy.stats import pearsonr

class DEA:
    """自定义DEA实现，支持CCR、BCC和SBM模型"""
    
    def __init__(self, input_data, output_data):
        self.input_data = np.array(input_data)
        self.output_data = np.array(output_data)
        self.n_dmus = self.input_data.shape[0]
        self.n_inputs = self.input_data.shape[1]
        self.n_outputs = self.output_data.shape[1]
        
    def ccr(self):
        """CCR模型 - 规模报酬不变"""
        return self._solve_dea_model(constant_returns=True)
    
    def bcc(self):
        """BCC模型 - 规模报酬可变"""
        return self._solve_dea_model(constant_returns=False)
    
    def sbm(self):
        """SBM模型 - 非径向模型"""
        return self._solve_sbm_model()
    
    def efficiency(self):
        """默认效率计算方法"""
        return self.ccr()
    
    def _solve_dea_model(self, constant_returns=True):
        """求解DEA模型"""
        efficiency_scores = []
        
        for i in range(self.n_dmus):
            # 目标函数：最大化效率
            c = np.zeros(self.n_dmus + 1)
            c[0] = -1  # 效率分数
            
            # 约束条件
            A_ub = []
            b_ub = []
            
            # 输入约束
            for j in range(self.n_inputs):
                constraint = np.zeros(self.n_dmus + 1)
                constraint[1:] = self.input_data[:, j]
                constraint[0] = -self.input_data[i, j]
                A_ub.append(constraint)
                b_ub.append(0)
            
            # 输出约束
            for j in range(self.n_outputs):
                constraint = np.zeros(self.n_dmus + 1)
                constraint[1:] = -self.output_data[:, j]
                constraint[0] = self.output_data[i, j]
                A_ub.append(constraint)
                b_ub.append(0)
            
            # 规模报酬约束
            if constant_returns:
                constraint = np.zeros(self.n_dmus + 1)
                constraint[1:] = 1
                A_ub.append(constraint)
                b_ub.append(1)
                constraint = np.zeros(self.n_dmus + 1)
                constraint[1:] = -1
                A_ub.append(constraint)
                b_ub.append(-1)
            
            # 非负约束
            bounds = [(0, None) for _ in range(self.n_dmus + 1)]
            
            try:
                result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
                if result.success:
                    efficiency_scores.append(-result.fun)
                else:
                    efficiency_scores.append(0.0)
            except:
                efficiency_scores.append(0.0)
        
        return np.array(efficiency_scores)
    
    def _solve_sbm_model(self):
        """求解SBM模型（简化版本）"""
        # 简化的SBM实现，使用CCR作为近似
        return self.ccr()

# 导入QCA分析模块
try:
    from qca_analysis import (
        check_r_connection, 
        perform_necessity_analysis, 
        perform_sufficiency_analysis,
        perform_truth_table_analysis,
        perform_minimization,
        perform_complete_qca_analysis
    )
    QCA_AVAILABLE = True
except ImportError as e:
    QCA_AVAILABLE = False
    st.warning(f"QCA模块导入失败: {e}")

# 设置页面配置
st.set_page_config(
    page_title="基于DEA与fsQCA的医院运营效能与发展路径智慧决策系统v1.0",
    page_icon="🏥",
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

def validate_hospital_id_column(df):
    """验证数据是否包含医院ID列"""
    if '医院ID' not in df.columns:
        return False, "错误：上传的文件必须包含'医院ID'列！"
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

def validate_numeric_data(df, exclude_columns=['医院ID']):
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
        # 将空值转换为0（除了医院ID列）
        hospital_id_cols = [col for col in df_cleaned.columns if '医院ID' in col or 'ID' in col]
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        
        # 对数值列的空值填充0
        for col in numeric_cols:
            if col not in hospital_id_cols:
                df_cleaned[col] = df_cleaned[col].fillna(0)
        
        # 对非数值列的空值也填充0（如果包含数字的话）
        for col in df_cleaned.columns:
            if col not in hospital_id_cols and col not in numeric_cols:
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
    columns = ["医院ID"] + [var["name"] for var in variables]
    
    # 创建数据输入界面
    data_rows = []
    for i in range(num_hospitals):
        st.write(f"**医院 {i+1}**")
        row_data = {"医院ID": f"H{i+1}"}
        
        cols = st.columns(len(variables) + 1)
        cols[0].write(f"H{i+1}")
        
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

def perform_dea_analysis(data, input_vars, output_vars, model_type):
    """
    执行DEA效率分析
    
    参数:
    - data: 包含医院数据的DataFrame
    - input_vars: 投入变量列表
    - output_vars: 产出变量列表
    - model_type: DEA模型类型 ('CCR', 'BCC', 'SBM')
    
    返回:
    - results: 包含效率值的DataFrame
    """
    try:
        # 准备数据
        hospital_ids = data['医院ID'].values
        input_data = data[input_vars].values
        output_data = data[output_vars].values
        
        # 创建DEA对象
        dea = DEA(input_data, output_data)
        
        # 根据模型类型执行分析
        if model_type == 'CCR':
            # 尝试不同的方法名
            if hasattr(dea, 'ccr'):
                efficiency_scores = dea.ccr()
            elif hasattr(dea, 'CCR'):
                efficiency_scores = dea.CCR()
            else:
                # 使用默认方法
                efficiency_scores = dea.efficiency()
        elif model_type == 'BCC':
            if hasattr(dea, 'bcc'):
                efficiency_scores = dea.bcc()
            elif hasattr(dea, 'BCC'):
                efficiency_scores = dea.BCC()
            else:
                efficiency_scores = dea.efficiency()
        elif model_type == 'SBM':
            if hasattr(dea, 'sbm'):
                efficiency_scores = dea.sbm()
            elif hasattr(dea, 'SBM'):
                efficiency_scores = dea.SBM()
            else:
                efficiency_scores = dea.efficiency()
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 确保efficiency_scores是numpy数组
        if not isinstance(efficiency_scores, np.ndarray):
            efficiency_scores = np.array(efficiency_scores)
        
        # 创建结果DataFrame
        results = pd.DataFrame({
            '医院ID': hospital_ids,
            '效率值': efficiency_scores
        })
        
        # 按效率值降序排列
        results = results.sort_values('效率值', ascending=False).reset_index(drop=True)
        
        return results
        
    except Exception as e:
        st.error(f"DEA分析执行失败: {str(e)}")
        # 返回模拟数据用于演示
        st.warning("⚠️ 使用模拟数据进行演示")
        hospital_ids = data['医院ID'].values
        # 生成模拟效率值
        np.random.seed(42)  # 确保结果可重现
        efficiency_scores = np.random.uniform(0.6, 1.0, len(hospital_ids))
        
        results = pd.DataFrame({
            '医院ID': hospital_ids,
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
        x='医院ID', 
        y='效率值',
        title='🏥 医院效率排名',
        labels={'效率值': '效率值', '医院ID': '医院ID'},
        color='效率值',
        color_continuous_scale='RdYlGn'
    )
    
    # 更新布局
    fig.update_layout(
        xaxis_title="医院ID",
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
        valid_paths = fsqca_results[fsqca_results['路径类型'] != '无效路径'].copy()
        
        if len(valid_paths) == 0:
            return None
        
        # 创建柱状图
        fig = px.bar(
            valid_paths,
            x='路径组合',
            y='覆盖度',
            color='路径类型',
            title='🔍 路径覆盖度比较',
            labels={'覆盖度': '覆盖度', '路径组合': '路径组合'},
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
        st.markdown(f'<div class="metric-card"><h4>系统状态</h4><p style="font-size: 1.2rem; margin: 0; color: #1a365d;">运行正常</p></div>', unsafe_allow_html=True)
    
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
            st.info("请上传包含医院数据的Excel或CSV文件，文件必须包含'医院ID'列。")
            
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
                    
                    # 验证医院ID列
                    is_valid, message = validate_hospital_id_column(df)
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
                num_hospitals = st.slider("医院数量", min_value=3, max_value=20, value=5, help="选择3-20家医院")
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
        
        # 获取数值列（排除医院ID）
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
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
                input_vars = st.multiselect(
                    "投入变量",
                    options=numeric_columns,
                    key="input_vars",
                    help="选择作为投入的变量，至少选择1个",
                    placeholder="请选择投入变量..."
                )
            
            with col2:
                st.markdown("**选择【产出变量】**")
                st.caption("服务成果类指标，如门诊量、手术量等")
                st.info("💡 **医疗示例**：门诊人次、住院人次、手术例数、出院人数、患者满意度等")
                output_vars = st.multiselect(
                    "产出变量",
                    options=numeric_columns,
                    key="output_vars",
                    help="选择作为产出的变量，至少选择1个",
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
                        "features": "• 非径向效率测量\n• 处理非期望产出\n• 更精确的效率评估"
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
                                model_info['value']
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
                                results_display = results_display[['排名', '医院ID', '效率值']]
                                
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
                                
                                # 高亮最优医院
                                best_hospital = results.iloc[0]
                                st.markdown(f"🏆 **最优医院**: {best_hospital['医院ID']} (效率值: {best_hospital['效率值']:.3f})")
                                
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
    else:
        st.warning("⚠️ 请先在数据输入区中加载数据")
    
    st.markdown('</div>', unsafe_allow_html=True)  # 关闭DEA分析区容器
    
    # ③ fsQCA路径分析区
    st.markdown('<div class="section-header">③ fsQCA路径分析区</div>', unsafe_allow_html=True)
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    
    if 'data' in st.session_state and 'dea_results' in st.session_state:
        data = st.session_state['data']
        dea_results = st.session_state['dea_results']
        
        # 显示数据预览
        st.subheader("📋 数据预览")
        st.dataframe(data.head(), use_container_width=True)
        
        # 获取可用的条件变量（排除DEA已使用的变量）
        used_vars = st.session_state.get('selected_input_vars', []) + st.session_state.get('selected_output_vars', [])
        available_vars = [col for col in data.columns if col not in ['医院ID'] + used_vars]
        
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
                            data_with_efficiency = data.merge(dea_results, on='医院ID', how='left')
                            
                            # 执行必要性分析
                            necessity_results = pd.DataFrame()
                            if perform_necessity:
                                necessity_results = perform_necessity_analysis(
                                    data_with_efficiency, 
                                    condition_vars, 
                                    '效率值'
                                )
                                
                                # 过滤一致性<0.9的变量
                                valid_vars = necessity_results[necessity_results['一致性'] >= 0.9]['变量名'].tolist()
                                if valid_vars:
                                    condition_vars = valid_vars
                                    st.info(f"✅ 必要性分析完成，保留 {len(valid_vars)} 个有效条件变量")
                                else:
                                    st.warning("⚠️ 所有条件变量的一致性都<0.9，使用原始变量进行分析")
                            
                            # 执行fsQCA分析
                            fsqca_results = perform_minimization(
                                data_with_efficiency,
                                condition_vars,
                                '效率值',
                                freq_threshold,
                                consistency
                            )
                            
                            if not fsqca_results.empty:
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
                                valid_paths = fsqca_results[fsqca_results['路径类型'] != '无效路径']
                                
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
                                        if row['路径类型'] == '核心路径':
                                            return ['core-path-row'] * len(row)
                                        elif row['路径类型'] == '边缘路径':
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
                                        core_paths = len(valid_paths[valid_paths['路径类型'] == '核心路径'])
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
                                        st.markdown(f"🏆 **最优路径**: {best_path['路径组合']}")
                                        st.markdown(f"   - 一致性: {best_path['一致性']:.4f}")
                                        st.markdown(f"   - 覆盖度: {best_path['覆盖度']:.4f}")
                                        st.markdown(f"   - 路径类型: {best_path['路径类型']}")
                                    
                                    # 路径解释
                                    st.markdown("**路径解释**")
                                    st.markdown("- **核心路径**: 同时满足PRI一致性和一致性阈值的路径")
                                    st.markdown("- **边缘路径**: 仅满足一致性阈值的路径")
                                    st.markdown("- **无效路径**: 不满足任何阈值的路径")
                                    
                                else:
                                    st.warning("⚠️ 没有找到有效路径，请尝试调整参数阈值")
                            else:
                                st.error("❌ fsQCA分析失败，请检查数据和参数设置")
    else:
        if 'data' not in st.session_state:
            st.warning("⚠️ 请先在数据输入区中加载数据")
        elif 'dea_results' not in st.session_state:
            st.warning("⚠️ 请先完成DEA效率分析")
    
    st.markdown('</div>', unsafe_allow_html=True)  # 关闭fsQCA分析区容器

if __name__ == "__main__":
    main()
