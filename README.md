# RWKV Dataset GUI

一个用于RWKV模型数据集处理的图形化界面工具，支持多种数据格式处理、小说增强处理和数据格式转换。

## 功能特性

### 1. 通用数据处理
- **多种数据格式支持**：
  - 单轮问答
  - 多轮对话
  - 指令问答
  - 长文本处理
  - 带标题文章
  - 小说续写
  - 章节扩展

- **灵活的输入方式**：
  - 单文件上传
  - 目录批量处理

### 2. 小说类增强处理
- **智能文本处理**：
  - 自定义段落长度控制（最小/最大长度）
  - 智能段落分割
  - 文本清理和格式化

- **AI增强功能**：
  - 支持自定义OpenAI类型API接口（如vLLM服务）
  - AI关键词提取
  - 自定义指令模板
  - 可配置API URL和模型名称

- **段落编辑功能**：
  - JSONL文件导入和编辑
  - 段落统计信息显示
  - 批量选择/取消选择段落
  - 删除选中段落
  - AI重新生成关键词
  - 段落内容预览和编辑

### 3. 数据格式转换
- **JSONL到BinIdx转换**：
  - 支持RWKV训练格式转换
  - 自定义输出路径
  - 多种分词器支持

## 安装和使用

### 环境要求
- Python 3.8+
- 所需依赖包（见requirements.txt）

### 安装步骤

1. 克隆项目：
```bash
git clone <repository-url>
cd RWKV-PEFT-GUI
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 确保json2binidx_tool目录存在且包含必要文件：
   - `rwkv_vocab_v20230424.txt`
   - `preprocess_data.py`

### 启动应用

```bash
python rwkv_dataset_gui.py
```

应用将在 `http://127.0.0.1:7863` 启动。

## 使用说明

### 通用数据处理
1. 选择输入方式（文件上传或目录处理）
2. 选择数据格式
3. 可选：设置系统提示词
4. 点击"开始处理"按钮
5. 查看处理结果和预览
6. 下载生成的JSONL文件

### 小说增强处理

#### 文件处理
1. 上传小说文件（支持txt格式）
2. 设置处理参数：
   - 最小段落长度
   - 最大段落长度
   - 是否启用智能分割
3. 配置AI设置（可选）：
   - API密钥
   - API URL（默认支持OpenAI格式）
   - 模型名称
   - 自定义指令
4. 点击"开始处理"
5. 保存生成的JSONL文件

#### 段落编辑
1. 上传已处理的JSONL文件
2. 查看段落统计信息
3. 选择需要编辑的段落
4. 执行操作：
   - 删除选中段落
   - AI重新生成关键词
   - 编辑段落内容
5. 保存修改后的文件

### 数据格式转换
1. 确保已有JSONL数据
2. 设置输出文件名和路径
3. 选择分词器类型
4. 点击"转换为BinIdx"
5. 下载生成的.bin和.idx文件

## API配置

### 支持的API类型
- OpenAI API
- DeepSeek API
- vLLM服务
- 其他OpenAI兼容的API服务

### 配置示例

**OpenAI API：**
- API URL: `https://api.openai.com/v1/chat/completions`
- 模型: `gpt-3.5-turbo` 或 `gpt-4`

**vLLM服务：**
- API URL: `http://localhost:8000/v1/chat/completions`
- 模型: 根据vLLM服务配置的模型名称

**DeepSeek API：**
- API URL: `https://api.deepseek.com/v1/chat/completions`
- 模型: `deepseek-chat`

## 注意事项

1. **文件格式**：确保输入文件为UTF-8编码
2. **API密钥**：使用AI功能时需要有效的API密钥
3. **内存使用**：处理大文件时注意内存使用情况
4. **输出路径**：确保输出路径有写入权限

## 故障排除

### 常见问题

1. **ModuleNotFoundError**：
   - 确保已安装所有依赖包
   - 运行 `pip install -r requirements.txt`

2. **文件路径错误**：
   - 确保json2binidx_tool目录存在
   - 检查词汇文件路径

3. **API调用失败**：
   - 检查API密钥是否正确
   - 确认API URL格式正确
   - 检查网络连接

4. **转换失败**：
   - 确保JSONL格式正确
   - 检查输出路径权限

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。

## 许可证

本项目采用MIT许可证。
