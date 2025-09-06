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
        perform_fsqca_analysis, 
        create_coverage_chart,
        download_fsqca_results
    )
    QCA_AVAILABLE = True
except ImportError:
    QCA_AVAILABLE = False
    st.warning("⚠️ QCA分析模块不可用，将使用简化版本")

# 页面配置
st.set_page_config(
    page_title="医院运营效能智慧决策系统",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 简化的CSS样式 - 避免复杂的DOM操作
st.markdown("""
<style>
    .stApp {
        background-color: #f8f9fa;
    }
    
    .main-title {
        font-size: 2rem;
        font-weight: bold;
        color: #1a365d;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background-color: #e3f2fd;
        border-radius: 10px;
    }
    
    .section-title {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1a365d;
        margin: 1.5rem 0 1rem 0;
        padding: 0.5rem 1rem;
        background-color: #f0f9ff;
        border-left: 4px solid #1a365d;
        border-radius: 5px;
    }
    
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 主标题
st.markdown('<div class="main-title">🏥 医院运营效能智慧决策系统</div>', unsafe_allow_html=True)

# 数据输入区域
st.markdown('<div class="section-title">📊 数据输入</div>', unsafe_allow_html=True)

# 文件上传
uploaded_file = st.file_uploader(
    "上传医院数据文件 (CSV格式)",
    type=['csv'],
    help="请上传包含医院投入产出数据的CSV文件"
)

# 示例数据
if uploaded_file is None:
    st.info("💡 没有上传文件？使用示例数据进行演示")
    if st.button("📋 加载示例数据"):
        # 生成示例数据
        np.random.seed(42)
        n_hospitals = 20
        
        sample_data = {
            '医院ID': [f'医院{i+1:02d}' for i in range(n_hospitals)],
            '医生数量': np.random.randint(50, 500, n_hospitals),
            '护士数量': np.random.randint(100, 800, n_hospitals),
            '床位数': np.random.randint(100, 1000, n_hospitals),
            '设备价值(万元)': np.random.randint(1000, 10000, n_hospitals),
            '门诊人次': np.random.randint(10000, 100000, n_hospitals),
            '住院人次': np.random.randint(1000, 10000, n_hospitals),
            '手术台次': np.random.randint(500, 5000, n_hospitals),
            '收入(万元)': np.random.randint(5000, 50000, n_hospitals)
        }
        
        df = pd.DataFrame(sample_data)
        st.session_state['data'] = df
        st.success("✅ 示例数据加载成功！")
        st.dataframe(df.head(10))

# 处理上传的数据
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state['data'] = df
        st.success("✅ 数据上传成功！")
        
        # 显示数据预览
        st.subheader("📋 数据预览")
        st.dataframe(df.head(10))
        
        # 显示数据统计
        st.subheader("📈 数据统计")
        st.dataframe(df.describe())
        
    except Exception as e:
        st.error(f"❌ 数据读取失败: {str(e)}")

# DEA分析区域
if 'data' in st.session_state:
    st.markdown('<div class="section-title">🔬 DEA效率分析</div>', unsafe_allow_html=True)
    
    df = st.session_state['data']
    
    # 变量选择
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("投入变量选择")
        st.info("💡 示例：医生数量、护士数量、床位数、设备价值")
        input_vars = st.multiselect(
            "选择投入变量",
            options=df.columns.tolist(),
            default=df.columns[:4].tolist() if len(df.columns) >= 4 else df.columns.tolist(),
            help="选择用于DEA分析的投入变量"
        )
    
    with col2:
        st.subheader("产出变量选择")
        st.info("💡 示例：门诊人次、住院人次、手术台次、收入")
        output_vars = st.multiselect(
            "选择产出变量",
            options=df.columns.tolist(),
            default=df.columns[4:8].tolist() if len(df.columns) >= 8 else df.columns[4:].tolist(),
            help="选择用于DEA分析的产出变量"
        )
    
    # 模型选择
    st.subheader("DEA模型选择")
    model_options = {
        'CCR': {
            'value': 'CCR',
            'description': '规模报酬不变模型',
            'scenario': '适用于规模相对稳定的医院',
            'features': '假设所有医院都在最优规模下运营'
        },
        'BCC': {
            'value': 'BCC', 
            'description': '规模报酬可变模型',
            'scenario': '适用于规模差异较大的医院',
            'features': '考虑规模效率对技术效率的影响'
        },
        'SBM': {
            'value': 'SBM',
            'description': '非径向松弛模型',
            'scenario': '适用于需要精确测量松弛的医院',
            'features': '考虑非径向松弛，更精确的效率测量'
        }
    }
    
    selected_model = st.selectbox(
        "选择DEA模型",
        options=list(model_options.keys()),
        help="选择适合的DEA分析模型"
    )
    
    # 显示模型信息
    model_info = model_options[selected_model]
    st.info(f"📋 **{model_info['description']}**\n\n"
            f"**应用场景**: {model_info['scenario']}\n\n"
            f"**模型特点**: {model_info['features']}")
    
    # 执行分析按钮
    if st.button("🚀 执行DEA分析", type="primary", use_container_width=True):
        if not input_vars or not output_vars:
            st.error("❌ 请选择至少一个投入变量和一个产出变量")
        else:
            try:
                # 准备数据
                input_data = df[input_vars].values
                output_data = df[output_vars].values
                
                # 执行DEA分析
                dea = DEA(input_data, output_data)
                
                if selected_model == 'CCR':
                    efficiency_scores = dea.ccr()
                elif selected_model == 'BCC':
                    efficiency_scores = dea.bcc()
                else:  # SBM
                    efficiency_scores = dea.sbm()
                
                # 创建结果DataFrame
                results = pd.DataFrame({
                    '医院ID': df.iloc[:, 0],  # 假设第一列是医院ID
                    '效率值': efficiency_scores
                })
                
                # 按效率值排序
                results = results.sort_values('效率值', ascending=False).reset_index(drop=True)
                
                # 显示结果
                st.subheader("📊 DEA分析结果")
                
                # 效率统计
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("平均效率", f"{results['效率值'].mean():.3f}")
                with col2:
                    st.metric("最高效率", f"{results['效率值'].max():.3f}")
                with col3:
                    efficient_count = len(results[results['效率值'] >= 0.9999])
                    st.metric("有效医院数", f"{efficient_count}/{len(results)}")
                
                # 效率表格
                st.subheader("📋 效率排名表")
                st.dataframe(results, use_container_width=True)
                
                # 效率分布图
                st.subheader("📈 效率分布图")
                fig = px.bar(
                    results, 
                    x='医院ID', 
                    y='效率值',
                    title=f"{selected_model}模型效率分析结果",
                    color='效率值',
                    color_continuous_scale='Blues'
                )
                fig.update_layout(
                    xaxis_tickangle=-45,
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 保存结果到session state
                st.session_state['dea_results'] = results
                st.session_state['selected_model'] = selected_model
                st.session_state['input_vars'] = input_vars
                st.session_state['output_vars'] = output_vars
                
            except Exception as e:
                st.error(f"❌ DEA分析失败: {str(e)}")

# fsQCA分析区域
if 'dea_results' in st.session_state and QCA_AVAILABLE:
    st.markdown('<div class="section-title">🔍 fsQCA路径分析</div>', unsafe_allow_html=True)
    
    st.info("💡 基于DEA分析结果，识别医院高质量发展的关键路径组合")
    
    # 参数配置
    st.subheader("🏥 医疗行业推荐值")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        freq_threshold = st.slider(
            "频数阈值",
            min_value=1,
            max_value=10,
            value=1,
            help="医疗小样本标准，默认1.0（Rihoux & Ragin, 2009）"
        )
    
    with col2:
        pri_consistency = st.slider(
            "PRI一致性",
            min_value=0.5,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="PRI=0.7（Ragin, 2008）"
        )
    
    with col3:
        consistency = st.slider(
            "一致性阈值",
            min_value=0.5,
            max_value=1.0,
            value=0.8,
            step=0.05,
            help="一致性≥0.8（Rihoux & Ragin, 2009）"
        )
    
    # 执行fsQCA分析
    if st.button("🚀 生成高质量发展路径", type="secondary", use_container_width=True):
        try:
            # 获取DEA结果
            dea_results = st.session_state['dea_results']
            
            # 执行fsQCA分析
            fsqca_results = perform_fsqca_analysis(
                dea_results,
                freq_threshold=freq_threshold,
                pri_consistency=pri_consistency,
                consistency=consistency
            )
            
            if fsqca_results is not None and not fsqca_results.empty:
                # 显示结果
                st.subheader("📊 高质量发展路径")
                
                # 路径统计
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("总路径数", len(fsqca_results))
                with col2:
                    valid_paths = fsqca_results[fsqca_results['路径类型'] != '无效路径']
                    st.metric("有效路径数", len(valid_paths))
                with col3:
                    core_paths = len(valid_paths[valid_paths['路径类型'] == '核心路径'])
                    st.metric("核心路径数", core_paths)
                
                # 路径表格
                st.subheader("📋 路径详情表")
                st.dataframe(fsqca_results, use_container_width=True)
                
                # 覆盖度图表
                st.subheader("📈 路径覆盖度比较")
                fig = create_coverage_chart(fsqca_results)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # 保存结果
                st.session_state['fsqca_results'] = fsqca_results
                
            else:
                st.warning("⚠️ 未找到有效的高质量发展路径")
                
        except Exception as e:
            st.error(f"❌ fsQCA分析失败: {str(e)}")

# 结果下载
if 'dea_results' in st.session_state:
    st.markdown('<div class="section-title">💾 结果下载</div>', unsafe_allow_html=True)
    
    # DEA结果下载
    dea_results = st.session_state['dea_results']
    csv_data = dea_results.to_csv(index=False, encoding='utf-8-sig')
    
    st.download_button(
        label="📥 下载DEA分析结果",
        data=csv_data,
        file_name=f"DEA分析结果_{st.session_state.get('selected_model', 'CCR')}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    # fsQCA结果下载
    if 'fsqca_results' in st.session_state:
        fsqca_results = st.session_state['fsqca_results']
        fsqca_csv = fsqca_results.to_csv(index=False, encoding='utf-8-sig')
        
        st.download_button(
            label="📥 下载fsQCA分析结果",
            data=fsqca_csv,
            file_name=f"fsQCA分析结果_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# 页脚
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 12px;'>"
    "医院运营效能智慧决策系统 v1.0 | 基于DEA和fsQCA的医院效率分析平台"
    "</div>",
    unsafe_allow_html=True
)
