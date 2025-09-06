# 医院运营效能分析系统

一个基于 Streamlit 的医院运营效能分析系统，支持数据包络分析（DEA）和模糊集定性比较分析（fsQCA）。

## 🚀 功能特性

- **数据上传与处理**：支持 Excel 文件上传和数据预处理
- **DEA 效率分析**：支持 CCR、BCC 和 SBM 模型
- **fsQCA 分析**：模糊集定性比较分析
- **结果可视化**：交互式图表和表格展示
- **结果导出**：支持 Excel 格式导出

## 📋 系统要求

- Python 3.8+
- 依赖包见 requirements.txt

## 🛠️ 安装与运行

### 本地运行
```bash
# 安装依赖
pip install -r requirements.txt

# 运行应用
streamlit run app.py
```

### 在线部署
1. 将代码推送到 GitHub 仓库
2. 在 [Streamlit Community Cloud](https://share.streamlit.io) 部署
3. 访问部署的 URL

## 📊 使用方法

1. **上传数据**：选择包含医院运营数据的 Excel 文件
2. **选择变量**：选择输入变量（如人员、设备）和输出变量（如收入、患者数）
3. **选择模型**：选择 DEA 模型类型（CCR/BCC/SBM）
4. **运行分析**：点击分析按钮开始计算
5. **查看结果**：查看效率分数、排名和可视化图表
6. **导出结果**：下载分析结果为 Excel 文件

## 🔧 技术架构

- **前端**：Streamlit
- **数据处理**：Pandas, NumPy
- **可视化**：Plotly
- **DEA 分析**：自定义实现（基于 scipy.optimize）
- **QCA 分析**：R 语言集成

## 📁 文件结构

```
├── app.py              # 主应用文件
├── qca_analysis.py     # QCA 分析模块
├── requirements.txt    # 依赖包列表
├── README.md          # 项目说明
└── 部署指南.md        # 部署指南
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License