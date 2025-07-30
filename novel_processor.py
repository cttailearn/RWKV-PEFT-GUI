import gradio as gr
import json
import re
import jieba
from collections import Counter
import os
import requests
import subprocess
import sys
from pathlib import Path

# DeepSeek API配置
DEEPSEEK_API_KEY = ""  # 用户需要填入自己的API密钥
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

def extract_keywords_with_ai(text, api_key):
    """使用DeepSeek API提取关键词"""
    if not api_key:
        return extract_keywords(text)  # 回退到本地方法
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        prompt = f"请从以下文本中提取3-5个最重要的关键词，用顿号分隔，只返回关键词，不要其他内容：\n\n{text[:500]}"  # 限制文本长度
        
        data = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 50,
            "temperature": 0.3
        }
        
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            keywords = result['choices'][0]['message']['content'].strip()
            return keywords if keywords else extract_keywords(text)
        else:
            print(f"API请求失败: {response.status_code}")
            return extract_keywords(text)
            
    except Exception as e:
        print(f"API调用出错: {str(e)}")
        return extract_keywords(text)

def extract_keywords(text, num_keywords=5):
    """从文本中提取关键词"""
    # 使用jieba分词
    words = jieba.cut(text)
    # 过滤停用词和标点符号
    stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
    filtered_words = [word for word in words if len(word) > 1 and word not in stop_words and word.isalpha()]
    
    # 统计词频
    word_freq = Counter(filtered_words)
    # 返回最常见的关键词
    keywords = [word for word, freq in word_freq.most_common(num_keywords)]
    return '、'.join(keywords) if keywords else '续写小说'

def clean_text(text):
    """清理文本，去除多余的空行和格式"""
    # 去除多余的空白字符
    text = re.sub(r'\s+', ' ', text)
    # 去除可能的标题标记
    text = re.sub(r'^第[一二三四五六七八九十\d]+章.*?\n', '', text, flags=re.MULTILINE)
    text = re.sub(r'^章节.*?\n', '', text, flags=re.MULTILINE)
    # 去除空行
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return '\n'.join(lines)

def smart_split_paragraphs(text, min_length=100, max_length=500, use_smart=True):
    """智能分割段落"""
    if use_smart:
        # 智能分段：考虑对话、场景转换等
        paragraphs = []
        
        # 先按自然段分割
        natural_paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        for para in natural_paragraphs:
            if len(para) <= max_length and len(para) >= min_length:
                paragraphs.append(para)
            elif len(para) > max_length:
                # 对过长段落进行二次分割
                sub_paragraphs = split_long_paragraph(para, min_length, max_length)
                paragraphs.extend(sub_paragraphs)
            else:
                # 对过短段落尝试合并
                if paragraphs and len(paragraphs[-1] + para) <= max_length:
                    paragraphs[-1] += '\n' + para
                else:
                    paragraphs.append(para)
        
        return paragraphs
    else:
        return split_into_paragraphs(text, min_length, max_length)

def split_long_paragraph(text, min_length=100, max_length=500):
    """分割过长的段落"""
    # 优先按对话分割
    if '"' in text or '"' in text or '"' in text:
        # 使用简单的方法按对话标记分割
        parts = re.split(r'(["""]+[^"""]*["""]+)', text)
        result = []
        current = ""
        
        for part in parts:
            if len(current + part) <= max_length:
                current += part
            else:
                if current and len(current) >= min_length:
                    result.append(current)
                current = part
        
        if current and len(current) >= min_length:
            result.append(current)
        
        return result if result else split_into_paragraphs(text, min_length, max_length)
    else:
        return split_into_paragraphs(text, min_length, max_length)

def split_into_paragraphs(text, min_length=100, max_length=500):
    """将文本分割成合适长度的段落"""
    # 按句号、问号、感叹号分割
    sentences = re.split(r'[。！？]', text)
    paragraphs = []
    current_paragraph = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # 如果当前段落加上新句子超过最大长度，则保存当前段落
        if len(current_paragraph + sentence) > max_length and len(current_paragraph) >= min_length:
            if current_paragraph:
                paragraphs.append(current_paragraph)
            current_paragraph = sentence
        else:
            if current_paragraph:
                current_paragraph += sentence + "。"
            else:
                current_paragraph = sentence + "。"
    
    # 添加最后一个段落
    if current_paragraph and len(current_paragraph) >= min_length:
        paragraphs.append(current_paragraph)
    
    return paragraphs

def process_novel(file_obj, min_paragraph_length, max_paragraph_length, use_smart_split, use_ai_keywords, api_key, custom_instruction):
    """处理小说文件并生成训练数据"""
    if file_obj is None:
        return "请上传小说文件", "", []
    
    try:
        # 修复文件读取错误
        if hasattr(file_obj, 'read'):
            content = file_obj.read()
        else:
            # 如果file_obj是文件路径字符串
            with open(file_obj, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        
        if isinstance(content, bytes):
            content = content.decode('utf-8', errors='ignore')
        
        # 清理文本
        cleaned_text = clean_text(content)
        
        # 分割段落（使用智能分段或传统分段）
        if use_smart_split:
            paragraphs = smart_split_paragraphs(cleaned_text, min_paragraph_length, max_paragraph_length, True)
        else:
            paragraphs = split_into_paragraphs(cleaned_text, min_paragraph_length, max_paragraph_length)
        
        if len(paragraphs) < 2:
            return "文本太短，无法生成训练数据", "", []
        
        # 生成训练数据
        training_data = []
        paragraph_data = []  # 用于界面显示和编辑
        
        for i in range(len(paragraphs) - 1):
            input_text = paragraphs[i]
            response_text = paragraphs[i + 1]
            
            # 提取关键词（使用AI或本地方法）
            if use_ai_keywords and api_key:
                keywords = extract_keywords_with_ai(response_text, api_key)
            else:
                keywords = extract_keywords(response_text)
            
            # 使用自定义指令模板或默认模板
            if custom_instruction:
                instruction = custom_instruction.replace('{keywords}', keywords)
            else:
                instruction = f"请根据关键词'{keywords}'续写小说段落"
            
            data_item = {
                "text": f"Instruction: {instruction}\n\nInput: {input_text}\n\nResponse: {response_text}"
            }
            training_data.append(data_item)
            
            # 保存段落数据用于界面编辑
            paragraph_data.append({
                "index": i,
                "input": input_text,
                "response": response_text,
                "keywords": keywords,
                "instruction": instruction
            })
        
        # 转换为JSONL格式
        jsonl_output = "\n".join([json.dumps(item, ensure_ascii=False) for item in training_data])
        
        status_message = f"处理完成！生成了 {len(training_data)} 条训练数据"
        
        return status_message, jsonl_output, paragraph_data
        
    except Exception as e:
        return f"处理文件时出错: {str(e)}", "", []

def save_jsonl(jsonl_content, filename):
    """保存JSONL内容到文件"""
    if not jsonl_content:
        return "没有内容可保存"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(jsonl_content)
        return f"文件已保存为: {filename}"
    except Exception as e:
        return f"保存文件时出错: {str(e)}"

def convert_jsonl_to_binidx(jsonl_file, output_prefix, tokenizer_type="RWKVTokenizer"):
    """将JSONL文件转换为binidx格式"""
    try:
        # 检查json2binidx工具是否存在，使用绝对路径
        current_dir = Path(__file__).parent
        tool_dir = current_dir / "RWKV-PEFT" / "json2binidx_tool"
        if not tool_dir.exists():
            return "错误：找不到json2binidx_tool目录"
        
        preprocess_script = tool_dir / "tools" / "preprocess_data.py"
        if not preprocess_script.exists():
            return "错误：找不到preprocess_data.py脚本"
        
        # 选择tokenizer文件
        if tokenizer_type == "RWKVTokenizer":
            vocab_file = tool_dir / "rwkv_vocab_v20230424.txt"
        else:
            vocab_file = tool_dir / "20B_tokenizer.json"
        
        if not vocab_file.exists():
            return f"错误：找不到tokenizer文件: {vocab_file}"
        
        # 构建转换命令
        cmd = [
            sys.executable,
            str(preprocess_script),
            "--input", jsonl_file,
            "--output-prefix", output_prefix,
            "--vocab", str(vocab_file),
            "--dataset-impl", "mmap",
            "--tokenizer-type", tokenizer_type,
            "--append-eod"
        ]
        
        # 执行转换
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(tool_dir))
        
        if result.returncode == 0:
            return f"转换成功！生成文件: {output_prefix}.bin 和 {output_prefix}.idx"
        else:
            return f"转换失败: {result.stderr}"
            
    except Exception as e:
        return f"转换过程中出错: {str(e)}"

def update_paragraph_data(paragraph_data, index, new_keywords, new_instruction):
    """更新段落数据"""
    if 0 <= index < len(paragraph_data):
        paragraph_data[index]['keywords'] = new_keywords
        paragraph_data[index]['instruction'] = new_instruction
        
        # 重新生成训练数据项
        input_text = paragraph_data[index]['input']
        response_text = paragraph_data[index]['response']
        
        data_item = {
            "text": f"Instruction: {new_instruction}\n\nInput: {input_text}\n\nResponse: {response_text}"
        }
        
        return data_item
    return None

def regenerate_jsonl_from_paragraph_data(paragraph_data):
    """从段落数据重新生成JSONL"""
    training_data = []
    
    for item in paragraph_data:
        data_item = {
            "text": f"Instruction: {item['instruction']}\n\nInput: {item['input']}\n\nResponse: {item['response']}"
        }
        training_data.append(data_item)
    
    return "\n".join([json.dumps(item, ensure_ascii=False) for item in training_data])

def parse_manual_selection(selection_text, max_count):
    """解析手动选择的段落编号"""
    if not selection_text.strip():
        return []
    
    selected_indices = []
    try:
        parts = selection_text.split(',')
        for part in parts:
            part = part.strip()
            if '-' in part:
                # 处理范围选择，如 "5-8"
                start, end = part.split('-')
                start_idx = int(start.strip()) - 1  # 转换为0基索引
                end_idx = int(end.strip()) - 1
                for i in range(start_idx, end_idx + 1):
                    if 0 <= i < max_count:
                        selected_indices.append(i)
            else:
                # 处理单个选择
                idx = int(part) - 1  # 转换为0基索引
                if 0 <= idx < max_count:
                    selected_indices.append(idx)
        
        # 去重并排序
        selected_indices = sorted(list(set(selected_indices)))
        return selected_indices
    
    except Exception as e:
        return []

def generate_paragraph_list_html(paragraph_data, selected_indices=None):
    """生成段落列表的HTML"""
    if not paragraph_data:
        return "<div style='padding: 10px; border: 1px solid #ddd; border-radius: 5px; max-height: 400px; overflow-y: auto;'>请先加载JSONL文件</div>"
    
    if selected_indices is None:
        selected_indices = []
    
    html_parts = []
    html_parts.append("<div style='padding: 10px; border: 1px solid #ddd; border-radius: 5px; max-height: 400px; overflow-y: auto;'>")
    
    for i, item in enumerate(paragraph_data):
        checked = "checked" if i in selected_indices else ""
        input_preview = item['input'][:80] + "..." if len(item['input']) > 80 else item['input']
        keywords_preview = item['keywords'][:30] + "..." if len(item['keywords']) > 30 else item['keywords']
        
        html_parts.append(f"""
        <div style='margin-bottom: 10px; padding: 8px; border: 1px solid #eee; border-radius: 3px; background-color: {'#f0f8ff' if checked else '#fafafa'};'>
            <label style='display: flex; align-items: flex-start; cursor: pointer;'>
                <input type='checkbox' {checked} onchange='toggleParagraph({i})' style='margin-right: 8px; margin-top: 2px;'>
                <div style='flex: 1;'>
                    <div style='font-weight: bold; color: #333; margin-bottom: 4px;'>段落 {i+1}</div>
                    <div style='font-size: 12px; color: #666; margin-bottom: 2px;'>关键词: {keywords_preview}</div>
                    <div style='font-size: 11px; color: #888; line-height: 1.3;'>{input_preview}</div>
                </div>
            </label>
        </div>
        """)
    
    html_parts.append("</div>")
    
    # 添加JavaScript
    html_parts.append("""
    <script>
    function toggleParagraph(index) {
        // 这里需要通过Gradio的接口来更新选择状态
        console.log('Toggle paragraph:', index);
    }
    </script>
    """)
    
    return "".join(html_parts)

def load_jsonl_file(file_path):
    """加载JSONL文件并解析为段落数据"""
    if not file_path:
        return [], "请选择JSONL文件"
    
    try:
        paragraph_data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                data = json.loads(line)
                text = data.get('text', '')
                
                # 解析instruction, input, response
                parts = text.split('\n\n')
                if len(parts) >= 3:
                    instruction_part = parts[0].replace('Instruction: ', '')
                    input_part = parts[1].replace('Input: ', '')
                    response_part = parts[2].replace('Response: ', '')
                    
                    # 从instruction中提取关键词
                    keywords = extract_keywords_from_instruction(instruction_part)
                    
                    paragraph_data.append({
                        'index': i,
                        'input': input_part,
                        'response': response_part,
                        'keywords': keywords,
                        'instruction': instruction_part
                    })
        
        return paragraph_data, f"成功加载 {len(paragraph_data)} 个段落"
    
    except Exception as e:
        return [], f"加载文件失败: {str(e)}"

def extract_keywords_from_instruction(instruction):
    """从指令中提取关键词"""
    # 尝试从指令中提取关键词
    import re
    match = re.search(r"关键词['\"](.*?)['\"]", instruction)
    if match:
        return match.group(1)
    return "续写小说"

def delete_selected_paragraphs(paragraph_data, selected_indices):
    """删除选中的段落"""
    if not selected_indices:
        return paragraph_data, "没有选择要删除的段落"
    
    try:
        # 将选择的索引转换为实际索引
        indices_to_delete = []
        for selected in selected_indices:
            idx = int(selected.split(":")[0].replace("段落 ", "")) - 1
            indices_to_delete.append(idx)
        
        # 按降序排序，从后往前删除
        indices_to_delete.sort(reverse=True)
        
        new_paragraph_data = paragraph_data.copy()
        for idx in indices_to_delete:
            if 0 <= idx < len(new_paragraph_data):
                new_paragraph_data.pop(idx)
        
        # 重新编号
        for i, item in enumerate(new_paragraph_data):
            item['index'] = i
        
        return new_paragraph_data, f"成功删除 {len(indices_to_delete)} 个段落"
    
    except Exception as e:
        return paragraph_data, f"删除失败: {str(e)}"

def ai_regenerate_keywords_batch(paragraph_data, selected_indices, api_key, prompt_template):
    """批量使用AI重新生成关键词"""
    if not selected_indices:
        return paragraph_data, "没有选择要处理的段落"
    
    if not api_key:
        return paragraph_data, "请输入API密钥"
    
    try:
        # 将选择的索引转换为实际索引
        indices_to_process = []
        for selected in selected_indices:
            idx = int(selected.split(":")[0].replace("段落 ", "")) - 1
            indices_to_process.append(idx)
        
        new_paragraph_data = paragraph_data.copy()
        success_count = 0
        
        for idx in indices_to_process:
            if 0 <= idx < len(new_paragraph_data):
                response_text = new_paragraph_data[idx]['response']
                
                # 使用自定义提示词模板
                prompt = prompt_template.replace('{text}', response_text[:500])
                
                # 调用AI API
                new_keywords = extract_keywords_with_ai_custom(response_text, api_key, prompt)
                
                if new_keywords and new_keywords != extract_keywords(response_text):
                    new_paragraph_data[idx]['keywords'] = new_keywords
                    new_paragraph_data[idx]['instruction'] = f"请根据关键词'{new_keywords}'续写小说段落"
                    success_count += 1
        
        return new_paragraph_data, f"成功为 {success_count} 个段落重新生成关键词"
    
    except Exception as e:
        return paragraph_data, f"AI处理失败: {str(e)}"

def extract_keywords_with_ai_custom(text, api_key, custom_prompt):
    """使用自定义提示词的AI关键词提取"""
    if not api_key:
        return extract_keywords(text)
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "user", "content": custom_prompt}
            ],
            "max_tokens": 50,
            "temperature": 0.3
        }
        
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            keywords = result['choices'][0]['message']['content'].strip()
            return keywords if keywords else extract_keywords(text)
        else:
            return extract_keywords(text)
            
    except Exception as e:
        return extract_keywords(text)

# 创建Gradio界面
def create_interface():
    
    with gr.Blocks(title="小说训练数据生成器", theme=gr.themes.Ocean()) as demo:
        # 状态变量存储段落数据
        paragraph_data_state = gr.State([])
        
        gr.Markdown("# 🚀 小说训练数据生成器")
        gr.Markdown("上传小说文件，自动生成指令微调格式的训练数据，支持AI关键词提取和智能分段")
        
        with gr.Tabs():
            # 主要处理标签页
            with gr.TabItem("📚 文件处理"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📁 文件上传")
                        file_input = gr.File(
                            label="上传小说文件",
                            file_types=[".txt", ".md"],
                            type="filepath"
                        )
                        
                        gr.Markdown("### ⚙️ 处理设置")
                        with gr.Row():
                            min_length = gr.Slider(
                                minimum=50,
                                maximum=300,
                                value=100,
                                step=10,
                                label="最小段落长度"
                            )
                            max_length = gr.Slider(
                                minimum=200,
                                maximum=800,
                                value=500,
                                step=50,
                                label="最大段落长度"
                            )
                        
                        use_smart_split = gr.Checkbox(
                            label="🧠 使用智能分段",
                            value=True,
                            info="考虑对话、场景转换等进行智能分段"
                        )
                        
                        gr.Markdown("### 🤖 AI设置")
                        use_ai_keywords = gr.Checkbox(
                            label="使用AI提取关键词",
                            value=False,
                            info="使用DeepSeek API提取更准确的关键词"
                        )
                        
                        api_key_input = gr.Textbox(
                            label="DeepSeek API Key",
                            type="password",
                            placeholder="sk-...",
                            visible=False
                        )
                        
                        custom_instruction = gr.Textbox(
                            label="自定义指令模板",
                            placeholder="请根据关键词'{keywords}'续写小说段落",
                            info="使用{keywords}作为关键词占位符"
                        )
                        
                        process_btn = gr.Button("🚀 处理文件", variant="primary", size="lg")
                        
                    with gr.Column(scale=2):
                        status_output = gr.Textbox(
                            label="📊 处理状态",
                            interactive=False
                        )
                        
                        jsonl_output = gr.Textbox(
                            label="📄 生成的JSONL数据",
                            lines=15,
                            max_lines=25,
                            interactive=False
                        )
                        
                        with gr.Row():
                            filename_input = gr.Textbox(
                                value="novel_training_data.jsonl",
                                label="💾 保存文件名",
                                scale=3
                            )
                            save_btn = gr.Button("💾 保存JSONL", scale=1)
                        
                        save_status = gr.Textbox(
                            label="💾 保存状态",
                            interactive=False
                        )
            
            # 段落编辑标签页
            with gr.TabItem("✏️ 段落编辑"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 📁 JSONL文件上传")
                        jsonl_upload = gr.File(
                            label="上传JSONL文件",
                            file_types=[".jsonl"],
                            type="filepath"
                        )
                        load_jsonl_btn = gr.Button("📂 加载JSONL文件", variant="secondary")
                        
                        gr.Markdown("### 📊 段落统计")
                        paragraph_stats = gr.Textbox(
                            label="统计信息",
                            interactive=False
                        )

                        gr.Markdown("### 📝 段落管理")
                        with gr.Row():
                            select_all_btn = gr.Button("✅ 全选", scale=1)
                            deselect_all_btn = gr.Button("❌ 取消全选", scale=1)
                        
                        # 段落选择状态
                        selected_paragraphs_state = gr.State([])
                        
                        # 手动选择段落输入框
                        manual_selection = gr.Textbox(
                            label="手动选择段落 (输入段落编号，用逗号分隔，如: 1,3,5-8)",
                            placeholder="例如: 1,3,5-8 表示选择段落1,3,5,6,7,8",
                            value=""
                        )
                        
                        update_selection_btn = gr.Button("🔄 更新选择", variant="secondary", size="sm")
                        
                        # 段落列表显示区域
                        paragraph_list_html = gr.HTML(
                            value="<div style='padding: 10px; border: 1px solid #ddd; border-radius: 5px; max-height: 400px; overflow-y: auto;'>请先加载JSONL文件</div>",
                            label="段落列表"
                        )
                        
                        with gr.Row():
                            delete_selected_btn = gr.Button("🗑️ 删除选中", variant="stop", scale=1)
                            ai_regenerate_btn = gr.Button("🤖 AI重新生成关键词", variant="primary", scale=1)
                        
                        gr.Markdown("### 🤖 AI设置")
                        ai_prompt_template = gr.Textbox(
                            label="AI提示词模板",
                            value="请从以下文本中提取3-5个最重要的关键词，用顿号分隔，只返回关键词：\n\n{text}",
                            lines=3,
                            info="使用{text}作为文本占位符"
                        )
                        
                        ai_api_key_edit = gr.Textbox(
                            label="DeepSeek API Key",
                            type="password",
                            placeholder="sk-..."
                        )

                    with gr.Column():    
                        gr.Markdown("### 🏷️ 单个段落编辑")
                        single_paragraph_selector = gr.Dropdown(
                            label="选择要编辑的段落",
                            choices=[],
                            interactive=True
                        )
                        
                        edit_keywords = gr.Textbox(
                            label="关键词",
                            placeholder="关键词1、关键词2、关键词3"
                        )
                        
                        edit_instruction = gr.Textbox(
                            label="指令",
                            placeholder="请根据关键词'关键词'续写小说段落"
                        )
                        
                        update_single_btn = gr.Button("🔄 更新单个段落", variant="secondary")
                        regenerate_btn = gr.Button("🔄 重新生成JSONL", variant="primary")
                        
                    with gr.Column(scale=2):
                        
                        
                        gr.Markdown("### 📖 段落内容预览")
                        input_preview = gr.Textbox(
                            label="Input (前一段)",
                            lines=5,
                            interactive=False
                        )
                        
                        response_preview = gr.Textbox(
                            label="Response (后一段)",
                            lines=5,
                            interactive=False
                        )
                        
                        current_instruction_preview = gr.Textbox(
                            label="当前指令",
                            lines=2,
                            interactive=False
                        )
                        
                        operation_status = gr.Textbox(
                            label="操作状态",
                            interactive=False
                        )
            
            # 格式转换标签页
            with gr.TabItem("🔄 格式转换"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 📊 JSONL转BinIdx")
                        gr.Markdown("将JSONL格式转换为RWKV训练所需的BinIdx格式")
                        
                        convert_jsonl_file = gr.File(
                            label="选择JSONL文件",
                            file_types=[".jsonl"]
                        )
                        
                        output_prefix = gr.Textbox(
                            label="输出文件前缀",
                            value="./data/novel_data",
                            placeholder="./data/novel_data"
                        )
                        
                        tokenizer_type = gr.Radio(
                            choices=["RWKVTokenizer", "HFTokenizer"],
                            value="RWKVTokenizer",
                            label="Tokenizer类型",
                            info="RWKVTokenizer用于多语言模型，HFTokenizer用于GPT-NeoX模型"
                        )
                        
                        convert_btn = gr.Button("🔄 开始转换", variant="primary")
                        
                    with gr.Column():
                        convert_status = gr.Textbox(
                            label="🔄 转换状态",
                            lines=10,
                            interactive=False
                        )
        
        # 事件绑定
        def toggle_api_key_visibility(use_ai):
            return gr.update(visible=use_ai)
        
        def process_and_update(file_obj, min_len, max_len, smart_split, ai_keywords, api_key, custom_instr):
            status, jsonl, paragraphs = process_novel(file_obj, min_len, max_len, smart_split, ai_keywords, api_key, custom_instr)
            
            # 更新段落选择器
            choices = [f"段落 {i+1}: {p['input'][:50]}..." for i, p in enumerate(paragraphs)]
            
            # 确保返回的是正确的类型
            return status, jsonl, paragraphs, gr.update(choices=choices, value=None)
        
        def load_jsonl_and_update(file_path):
            """加载JSONL文件并更新界面"""
            if not file_path:
                return [], [], "", generate_paragraph_list_html([]), "请选择JSONL文件"
            
            paragraph_data, status = load_jsonl_file(file_path)
            
            # 更新下拉选择器选项
            dropdown_choices = [f"段落 {i+1}: {p['input'][:50]}..." for i, p in enumerate(paragraph_data)]
            
            # 生成段落列表HTML
            paragraph_html = generate_paragraph_list_html(paragraph_data, [])
            
            # 统计信息
            stats = f"总段落数: {len(paragraph_data)}"
            
            return paragraph_data, dropdown_choices, stats, paragraph_html, status
        
        def select_all_paragraphs(paragraph_data):
            """全选段落"""
            if not paragraph_data:
                return [], generate_paragraph_list_html([])
            
            selected_indices = list(range(len(paragraph_data)))
            paragraph_html = generate_paragraph_list_html(paragraph_data, selected_indices)
            return selected_indices, paragraph_html
        
        def deselect_all_paragraphs(paragraph_data):
            """取消全选"""
            if not paragraph_data:
                return [], generate_paragraph_list_html([])
            
            paragraph_html = generate_paragraph_list_html(paragraph_data, [])
            return [], paragraph_html
        
        def delete_selected_and_update(paragraph_data, selected_indices):
            """删除选中段落并更新界面"""
            if not selected_indices:
                return paragraph_data, [], generate_paragraph_list_html(paragraph_data, []), f"总段落数: {len(paragraph_data)}", "没有选择要删除的段落"
            
            # 删除选中的段落
            indices_to_delete = sorted(selected_indices, reverse=True)
            new_paragraph_data = paragraph_data.copy()
            
            for idx in indices_to_delete:
                if 0 <= idx < len(new_paragraph_data):
                    new_paragraph_data.pop(idx)
            
            # 重新编号
            for i, item in enumerate(new_paragraph_data):
                item['index'] = i
            
            # 更新下拉选择器选项
            dropdown_choices = [f"段落 {i+1}: {p['input'][:50]}..." for i, p in enumerate(new_paragraph_data)]
            
            # 生成新的段落列表HTML
            paragraph_html = generate_paragraph_list_html(new_paragraph_data, [])
            
            # 统计信息
            stats = f"总段落数: {len(new_paragraph_data)}"
            
            status = f"成功删除 {len(indices_to_delete)} 个段落"
            
            return new_paragraph_data, dropdown_choices, [], paragraph_html, stats, status
        
        def ai_regenerate_and_update(paragraph_data, selected_indices, api_key, prompt_template):
            """AI批量重新生成关键词并更新界面"""
            if not selected_indices:
                return paragraph_data, generate_paragraph_list_html(paragraph_data, []), f"总段落数: {len(paragraph_data)}", "没有选择要处理的段落"
            
            if not api_key:
                return paragraph_data, generate_paragraph_list_html(paragraph_data, selected_indices), f"总段落数: {len(paragraph_data)}", "请输入API密钥"
            
            try:
                new_paragraph_data = paragraph_data.copy()
                success_count = 0
                
                for idx in selected_indices:
                    if 0 <= idx < len(new_paragraph_data):
                        response_text = new_paragraph_data[idx]['response']
                        
                        # 使用自定义提示词模板
                        prompt = prompt_template.replace('{text}', response_text[:500])
                        
                        # 调用AI API
                        new_keywords = extract_keywords_with_ai_custom(response_text, api_key, prompt)
                        
                        if new_keywords and new_keywords != extract_keywords(response_text):
                            new_paragraph_data[idx]['keywords'] = new_keywords
                            new_paragraph_data[idx]['instruction'] = f"请根据关键词'{new_keywords}'续写小说段落"
                            success_count += 1
                
                # 生成更新后的段落列表HTML
                paragraph_html = generate_paragraph_list_html(new_paragraph_data, selected_indices)
                
                # 统计信息
                stats = f"总段落数: {len(new_paragraph_data)}"
                status = f"成功为 {success_count} 个段落重新生成关键词"
                
                return new_paragraph_data, paragraph_html, stats, status
            
            except Exception as e:
                return paragraph_data, generate_paragraph_list_html(paragraph_data, selected_indices), f"总段落数: {len(paragraph_data)}", f"AI处理失败: {str(e)}"
        
        def update_manual_selection(paragraph_data, selection_text):
            """更新手动选择的段落"""
            if not paragraph_data:
                return [], generate_paragraph_list_html([])
            
            selected_indices = parse_manual_selection(selection_text, len(paragraph_data))
            paragraph_html = generate_paragraph_list_html(paragraph_data, selected_indices)
            
            return selected_indices, paragraph_html
        
        def update_paragraph_preview(paragraph_data, selected_index):
            if not paragraph_data or selected_index is None:
                return "", "", "", ""
            
            try:
                idx = int(selected_index.split(":")[0].replace("段落 ", "")) - 1
                if 0 <= idx < len(paragraph_data):
                    item = paragraph_data[idx]
                    return item['input'], item['response'], item['keywords'], item['instruction']
            except:
                pass
            
            return "", "", "", ""
        
        def update_paragraph_item(paragraph_data, selected_index, new_keywords, new_instruction):
            if not paragraph_data or selected_index is None:
                return paragraph_data, "请先选择段落"
            
            try:
                idx = int(selected_index.split(":")[0].replace("段落 ", "")) - 1
                if 0 <= idx < len(paragraph_data):
                    # 创建新的列表副本以避免状态管理问题
                    new_paragraph_data = paragraph_data.copy()
                    new_paragraph_data[idx] = paragraph_data[idx].copy()
                    new_paragraph_data[idx]['keywords'] = new_keywords
                    new_paragraph_data[idx]['instruction'] = new_instruction
                    return new_paragraph_data, f"段落 {idx+1} 更新成功"
            except:
                pass
            
            return paragraph_data, "更新失败"
        
        def regenerate_jsonl(paragraph_data):
            if not paragraph_data:
                return "", "没有数据可生成"
            
            jsonl = regenerate_jsonl_from_paragraph_data(paragraph_data)
            return jsonl, "JSONL重新生成成功"
        
        def convert_to_binidx(jsonl_file, prefix, tokenizer):
            if jsonl_file is None:
                return "请选择JSONL文件"
            
            return convert_jsonl_to_binidx(jsonl_file.name, prefix, tokenizer)
        
        # 绑定事件
        use_ai_keywords.change(
            fn=toggle_api_key_visibility,
            inputs=[use_ai_keywords],
            outputs=[api_key_input]
        )
        
        process_btn.click(
            fn=process_and_update,
            inputs=[file_input, min_length, max_length, use_smart_split, use_ai_keywords, api_key_input, custom_instruction],
            outputs=[status_output, jsonl_output, paragraph_data_state, single_paragraph_selector]
        )
        
        # JSONL文件上传事件
        load_jsonl_btn.click(
            fn=load_jsonl_and_update,
            inputs=[jsonl_upload],
            outputs=[paragraph_data_state, single_paragraph_selector, paragraph_stats, paragraph_list_html, operation_status]
        )
        
        # 全选/取消全选事件
        select_all_btn.click(
            fn=select_all_paragraphs,
            inputs=[paragraph_data_state],
            outputs=[selected_paragraphs_state, paragraph_list_html]
        )
        
        deselect_all_btn.click(
            fn=deselect_all_paragraphs,
            inputs=[paragraph_data_state],
            outputs=[selected_paragraphs_state, paragraph_list_html]
        )
        
        # 删除选中段落事件
        delete_selected_btn.click(
            fn=delete_selected_and_update,
            inputs=[paragraph_data_state, selected_paragraphs_state],
            outputs=[paragraph_data_state, single_paragraph_selector, selected_paragraphs_state, paragraph_list_html, paragraph_stats, operation_status]
        )
        
        # AI批量重新生成关键词事件
        ai_regenerate_btn.click(
            fn=ai_regenerate_and_update,
            inputs=[paragraph_data_state, selected_paragraphs_state, ai_api_key_edit, ai_prompt_template],
            outputs=[paragraph_data_state, paragraph_list_html, paragraph_stats, operation_status]
        )
        
        # 手动选择更新事件
        update_selection_btn.click(
            fn=update_manual_selection,
            inputs=[paragraph_data_state, manual_selection],
            outputs=[selected_paragraphs_state, paragraph_list_html]
        )
        
        single_paragraph_selector.change(
            fn=update_paragraph_preview,
            inputs=[paragraph_data_state, single_paragraph_selector],
            outputs=[input_preview, response_preview, edit_keywords, current_instruction_preview]
        )
        
        update_single_btn.click(
            fn=update_paragraph_item,
            inputs=[paragraph_data_state, single_paragraph_selector, edit_keywords, edit_instruction],
            outputs=[paragraph_data_state, operation_status]
        )
        
        regenerate_btn.click(
            fn=regenerate_jsonl,
            inputs=[paragraph_data_state],
            outputs=[jsonl_output, operation_status]
        )
        
        save_btn.click(
            fn=save_jsonl,
            inputs=[jsonl_output, filename_input],
            outputs=[save_status]
        )
        
        convert_btn.click(
            fn=convert_to_binidx,
            inputs=[convert_jsonl_file, output_prefix, tokenizer_type],
            outputs=[convert_status]
        )
        
        # 添加使用说明
        with gr.Accordion("📖 使用说明", open=False):
            gr.Markdown("""
            ## 🚀 功能特点
            
            ### 📚 文件处理
            1. **智能分段**: 考虑对话、场景转换等进行智能段落分割
            2. **AI关键词提取**: 使用DeepSeek API提取更准确的关键词
            3. **自定义指令**: 支持自定义指令模板，使用{keywords}作为占位符
            
            ### ✏️ 段落编辑
            1. **可视化编辑**: 查看和编辑每个段落的关键词和指令
            2. **实时预览**: 实时查看段落内容和当前指令
            3. **批量更新**: 支持重新生成整个JSONL文件
            
            ### 🔄 格式转换
            1. **BinIdx转换**: 将JSONL转换为RWKV训练格式
            2. **多种Tokenizer**: 支持RWKV和GPT-NeoX tokenizer
            
            ## 📝 数据格式
            
            生成的训练数据格式：
            ```
            Instruction: 请根据关键词'武功、内力、修炼'续写小说段落
            
            Input: 前一段小说内容...
            
            Response: 后一段小说内容...
            ```
            
            ## 🔧 使用步骤
            
            1. **上传文件**: 选择小说文本文件
            2. **配置参数**: 设置段落长度、分段方式等
            3. **AI设置**: 可选择使用DeepSeek API提取关键词
            4. **处理文件**: 生成训练数据
            5. **编辑优化**: 在段落编辑页面调整关键词和指令
            6. **保存导出**: 保存JSONL文件或转换为BinIdx格式
            """)
    
    return demo

if __name__ == "__main__":
    # 确保jieba已安装
    try:
        import jieba
    except ImportError:
        print("请先安装jieba: pip install jieba")
        exit(1)
    
    demo = create_interface()
    demo.launch(share=False, server_name="127.0.0.1", server_port=None, show_error=True)