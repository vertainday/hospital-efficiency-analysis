import numpy as np
import pandas as pd
from scipy.optimize import linprog
import os
import sys
import matplotlib.pyplot as plt

# ================================
# 1. CCR / BCC 输入导向模型
# ================================
def dea_input_oriented(X, Y, returns='CRS'):
    """
    输入导向 DEA 模型（CCR 或 BCC）
    X: (m, n) 输入矩阵（每列一个 DMU）
    Y: (s, n) 输出矩阵
    returns: 'CRS'（规模报酬不变，CCR） or 'VRS'（BCC）
    """
    m, n = X.shape
    s = Y.shape[0]
    efficiency = np.zeros(n)
    lambdas = []

    for k in range(n):
        c = np.zeros(1 + n)
        c[0] = 1.0  # min θ

        # 约束 A_ub @ [θ, λ] <= b_ub
        # 输入：sum λ x_ij <= θ x_ik
        A_inputs = np.zeros((m, 1 + n))
        A_inputs[:, 0] = -X[:, k]        # -θ x_ik
        A_inputs[:, 1:] = X              # sum λ x_ij

        # 输出：sum λ y_rj >= y_rk  => -sum λ y_rj <= -y_rk
        A_outputs = np.zeros((s, 1 + n))
        A_outputs[:, 1:] = -Y

        A_ub = np.vstack([A_inputs, A_outputs])
        b_ub = np.hstack([-X[:, k], -Y[:, k]])

        # BCC 添加 sum λ_j = 1
        A_eq = None
        b_eq = None
        if returns == 'VRS':
            A_eq = np.zeros((1, 1 + n))
            A_eq[0, 1:] = 1.0
            b_eq = [1.0]

        bounds = [(None, None)] + [(0, None) for _ in range(n)]

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method='highs', options={'presolve': True})

        if res.success:
            efficiency[k] = res.x[0]
            lambdas.append(res.x[1:])
        else:
            efficiency[k] = np.nan
            lambdas.append(np.nan * np.ones(n))

    return efficiency, lambdas


# ================================
# 2. CCR / BCC 输出导向模型
# ================================
def dea_output_oriented(X, Y, returns='CRS'):
    """
    输出导向 DEA 模型
    最小化 θ，其中 φ = 1/θ 是输出放大倍数
    """
    m, n = X.shape
    s = Y.shape[0]
    efficiency = np.zeros(n)

    for k in range(n):
        c = np.zeros(1 + n)
        c[0] = 1.0  # min θ = 1/φ

        # 约束 1: sum λ x_ij <= x_ik（输入不可增加）
        A_inputs = np.zeros((m, 1 + n))
        A_inputs[:, 1:] = X
        b_inputs = X[:, k]

        # 约束 2: sum λ y_rj >= θ * y_rk
        # 转为：-sum λ y_rj + θ y_rk <= 0
        A_outputs = np.zeros((s, 1 + n))
        A_outputs[:, 0] = Y[:, k]  # θ y_rk 的系数
        A_outputs[:, 1:] = -Y     # -sum λ y_rj 的系数
        b_outputs = np.zeros(s)

        A_ub = np.vstack([A_inputs, A_outputs])
        b_ub = np.hstack([b_inputs, b_outputs])

        # BCC: sum λ_j = 1
        A_eq = None
        b_eq = None
        if returns == 'VRS':
            A_eq = np.zeros((1, 1 + n))
            A_eq[0, 1:] = 1.0
            b_eq = [1.0]

        bounds = [(0, None)] + [(0, None)] * n  # θ >= 0

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method='highs')

        if res.success:
            efficiency[k] = res.x[0]
        else:
            efficiency[k] = np.nan

    return efficiency


# ================================
# 3. 超效率 CCR/BCC 模型
# ================================
def dea_super_efficiency(X, Y, returns='CRS'):
    """超效率：排除自身"""
    m, n = X.shape
    s = Y.shape[0]
    efficiency = np.zeros(n)

    for k in range(n):
        # 移除第 k 个 DMU
        X_other = np.delete(X, k, axis=1)
        Y_other = np.delete(Y, k, axis=1)
        nn = n - 1

        c = np.zeros(1 + nn)
        c[0] = 1.0

        # 输入约束: sum λ x_ij <= θ x_ik
        A_inputs = np.zeros((m, 1 + nn))
        A_inputs[:, 0] = -X[:, k]
        A_inputs[:, 1:] = X_other

        # 输出约束: sum λ y_rj >= y_rk
        A_outputs = np.zeros((s, 1 + nn))
        A_outputs[:, 1:] = -Y_other

        A_ub = np.vstack([A_inputs, A_outputs])
        b_ub = np.hstack([-X[:, k], -Y[:, k]])

        A_eq = None
        b_eq = None
        if returns == 'VRS':
            A_eq = np.zeros((1, 1 + nn))
            A_eq[0, 1:] = 1.0
            b_eq = [1.0]

        bounds = [(None, None)] + [(0, None)] * nn

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method='highs')

        if res.success:
            efficiency[k] = res.x[0]
        else:
            efficiency[k] = np.nan

    return efficiency


# ================================
# 4. 超效率 SBM 模型（Input-oriented, VRS）
# ================================
def dea_super_sbm(X, Y):
    """
    超效率 SBM 模型（Tone, 2014）Input-oriented, VRS
    使用 Charnes-Cooper 变换线性化
    """
    m, n = X.shape
    s = Y.shape[0]
    efficiency = np.zeros(n)

    for k in range(n):
        X_other = np.delete(X, k, axis=1)  # (m, n-1)
        Y_other = np.delete(Y, k, axis=1)
        nn = n - 1

        num_vars = 1 + nn + m + s  # t, μ, s⁻, s⁺
        c = np.zeros(num_vars)

        # 目标: min ρ = t - (1/m) Σ(s_i⁻ / x_ik)
        c[0] = 1.0
        c[1 + nn:1 + nn + m] = -1.0 / m / X[:, k]  # s⁻ 系数，注意这里是负号

        A_eq = []
        b_eq = []

        # (1) 输入约束: t * x_ik = Σ μ_j x_ij - s_i⁻
        for i in range(m):
            row = np.zeros(num_vars)
            row[0] = X[i, k]
            row[1:1 + nn] = -X_other[i, :]
            row[1 + nn + i] = 1
            A_eq.append(row)
            b_eq.append(0.0)

        # (2) 输出约束: t * y_rk = Σ μ_j y_rj + s_r⁺
        for r in range(s):
            row = np.zeros(num_vars)
            row[0] = -Y[r, k]
            row[1:1 + nn] = Y_other[r, :]
            row[1 + nn + m + r] = 1
            A_eq.append(row)
            b_eq.append(0.0)

        # (3) Σ μ_j = t
        row = np.zeros(num_vars)
        row[0] = -1
        row[1:1 + nn] = 1
        A_eq.append(row)
        b_eq.append(0.0)

        A_eq = np.array(A_eq)
        b_eq = np.array(b_eq)

        bounds = [(1e-7, None)]          # t
        bounds += [(0, None)] * nn      # μ_j
        bounds += [(0, None)] * m       # s⁻
        bounds += [(0, None)] * s       # s⁺

        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                      method='highs', options={'presolve': True})

        if res.success:
            efficiency[k] = res.fun  # ρ = objective value
        else:
            efficiency[k] = np.nan

    return efficiency


# ================================
# 5. 数据读取函数
# ================================
def read_data(file_path):
    """读取数据文件，支持CSV和Excel格式"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    ext = os.path.splitext(file_path)[-1].lower()
    try:
        if ext == '.csv':
            df = pd.read_csv(file_path, encoding='utf-8')
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError("仅支持 .csv 和 .xlsx 格式的文件")
    except Exception as e:
        raise ValueError(f"文件读取失败: {e}")
    
    if df.empty:
        raise ValueError("文件为空或无法读取数据")

    print("📊 数据预览：")
    print(df.head())

    try:
        num_inputs = int(input(f"🔢 请输入输入变量个数（共 {len(df.columns)} 列）: "))
        num_outputs = len(df.columns) - num_inputs
        if num_outputs <= 0:
            raise ValueError("输出列数必须 > 0")
    except:
        print("⚠️ 默认使用前2列为输入")
        num_inputs = 2
        num_outputs = len(df.columns) - 2

    # 检查第一列是否为字符串类型（DMU名称）
    first_col_str = df.iloc[:, 0].dtype == 'object'
    
    # 进一步验证：检查第一列是否包含非数值数据
    if first_col_str:
        try:
            # 尝试将第一列转换为数值，如果失败则认为是DMU名称
            pd.to_numeric(df.iloc[:, 0], errors='raise')
            # 如果转换成功，说明第一列是数值，不是DMU名称
            first_col_str = False
        except (ValueError, TypeError):
            # 转换失败，确认第一列是DMU名称
            first_col_str = True
    
    if first_col_str:
        DMUs = df.iloc[:, 0].values
        X = df.iloc[:, 1:num_inputs + 1].values.T
        Y = df.iloc[:, num_inputs + 1:].values.T
    else:
        DMUs = [f"DMU{i + 1}" for i in range(len(df))]
        X = df.iloc[:, :num_inputs].values.T
        Y = df.iloc[:, num_inputs:].values.T

    # 数据验证
    if X.shape[1] == 0:
        raise ValueError("没有有效的输入数据")
    if Y.shape[1] == 0:
        raise ValueError("没有有效的输出数据")
    
    # 检查是否有负值或NaN
    if np.any(X < 0) or np.any(np.isnan(X)):
        raise ValueError("输入数据不能包含负值或缺失值")
    if np.any(Y < 0) or np.any(np.isnan(Y)):
        raise ValueError("输出数据不能包含负值或缺失值")

    return X, Y, DMUs, df.columns.tolist()


# ================================
# 6. 主程序 + 可视化
# ================================
def main(file_path):
    try:
        X, Y, dmus, col_names = read_data(file_path)
    except Exception as e:
        print("❌ 数据读取失败:", e)
        return

    print(f"\n✅ 共 {X.shape[1]} 个 DMU，{X.shape[0]} 个输入，{Y.shape[0]} 个输出")

    # 存储结果
    results = pd.DataFrame({"DMU": dmus})

    # === 1. CCR Input ===
    print("⏳ 计算 CCR 输入导向...")
    eff, _ = dea_input_oriented(X, Y, 'CRS')
    results['CCR_Input'] = np.round(eff, 4)

    # === 2. BCC Input ===
    print("⏳ 计算 BCC 输入导向...")
    eff, _ = dea_input_oriented(X, Y, 'VRS')
    results['BCC_Input'] = np.round(eff, 4)

    # === 3. CCR Output ===
    print("⏳ 计算 CCR 输出导向...")
    eff = dea_output_oriented(X, Y, 'CRS')
    results['CCR_Output'] = np.round(eff, 4)

    # === 4. BCC Output ===
    print("⏳ 计算 BCC 输出导向...")
    eff = dea_output_oriented(X, Y, 'VRS')
    results['BCC_Output'] = np.round(eff, 4)

    # === 5. Super CCR ===
    print("⏳ 计算 超效率 CCR...")
    eff = dea_super_efficiency(X, Y, 'CRS')
    results['Super_CCR'] = np.round(eff, 4)

    # === 6. Super BCC ===
    print("⏳ 计算 超效率 BCC...")
    eff = dea_super_efficiency(X, Y, 'VRS')
    results['Super_BCC'] = np.round(eff, 4)

    # === 7. Super SBM ===
    print("⏳ 计算 超效率 SBM (Input-VRS)...")
    eff = dea_super_sbm(X, Y)
    results['Super_SBM'] = np.round(eff, 4)

    # === 输出结果 ===
    print("\n" + "="*60)
    print("🎯 完整 DEA 分析结果")
    print("="*60)
    print(results.to_string(index=False))

    # === 保存 Excel ===
    output_file = "DEA_All_Models_Results.xlsx"
    results.to_excel(output_file, index=False)
    print(f"\n✅ 结果已保存至: {output_file}")

    # ================================
    # 可视化
    # ================================
    plt.figure(figsize=(14, 8))

    x = np.arange(len(dmus))
    width = 0.12

    cols_to_plot = ['CCR_Input', 'BCC_Input', 'CCR_Output', 'BCC_Output', 'Super_CCR', 'Super_BCC', 'Super_SBM']
    colors = ['skyblue', 'lightcoral', 'gold', 'lightgreen', 'plum', 'lightsalmon', 'purple']
    labels = ['CCR-In', 'BCC-In', 'CCR-Out', 'BCC-Out', 'Sup-CCR', 'Sup-BCC', 'Sup-SBM']

    for i, (col, color, label) in enumerate(zip(cols_to_plot, colors, labels)):
        plt.bar(x + i * width, results[col], width, label=label, color=color, alpha=0.8)

    plt.xlabel('DMU')
    plt.ylabel('Efficiency Score')
    plt.title('DEA 模型效率对比（所有模型）')
    plt.xticks(x + 3 * width, dmus, rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(0, max(results.max(numeric_only=True)) * 1.1)
    plt.tight_layout()
    plt.savefig("DEA_All_Models_Comparison.png", dpi=150)
    plt.show()

    print("📈 图表已保存: DEA_All_Models_Comparison.png")


# ================================
# 7. 启动入口
# ================================
if __name__ == "__main__":
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = input("📄 请输入数据文件路径（.csv 或 .xlsx）: ").strip()

    if not os.path.exists(filepath):
        print("❌ 文件不存在！")
    else:
        main(filepath)