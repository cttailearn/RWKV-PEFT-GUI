import gradio as gr
import json
import re
import jieba
from collections import Counter
import os
import requests
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path
import argparse
import multiprocessing

# 添加json2binidx_tool路径到sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'json2binidx_tool', 'tools'))

# AI API配置
DEFAULT_API_KEY = ""  # 用户需要填入自己的API密钥
DEFAULT_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEFAULT_MODEL = "deepseek-chat"

class DataProcessor:
    def __init__(self):
        self.vocab_file = os.path.join(os.path.dirname(__file__), 'json2binidx_tool', 'rwkv_vocab_v20230424.txt')
        
    def process_single_qa(self, text_content):
        """处理单轮问答格式"""
        lines = text_content.strip().split('\n')
        jsonl_data = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if '\t' in line:
                parts = line.split('\t', 1)
                if len(parts) == 2:
                    question, answer = parts
                    jsonl_data.append({
                        "text": f"User: {question.strip()}\n\nAssistant: {answer.strip()}"
                    })
            elif '|' in line:
                parts = line.split('|', 1)
                if len(parts) == 2:
                    question, answer = parts
                    jsonl_data.append({
                        "text": f"User: {question.strip()}\n\nAssistant: {answer.strip()}"
                    })
        
        return jsonl_data
    
    def process_novel_continuation(self, text_content):
        """处理小说段落续写数据格式"""
        jsonl_data = []
        lines = text_content.strip().split('\n')
        
        current_prompt = ""
        current_continuation = ""
        in_continuation = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith("段落开头:") or line.startswith("开头:"):
                # 保存之前的数据
                if current_prompt and current_continuation:
                    jsonl_data.append({
                        "text": f"User: {current_prompt}\n\nAssistant: {current_continuation}"
                    })
                
                current_prompt = line.replace("段落开头:", "").replace("开头:", "").strip()
                current_continuation = ""
                in_continuation = False
            elif line.startswith("续写:") or line.startswith("后续:") or line.startswith("继续:"):
                current_continuation = line.replace("续写:", "").replace("后续:", "").replace("继续:", "").strip()
                in_continuation = True
            elif in_continuation:
                current_continuation += "\n" + line
            else:
                if current_prompt:
                    current_prompt += "\n" + line
                else:
                    current_prompt = line
        
        # 处理最后一组数据
        if current_prompt and current_continuation:
            jsonl_data.append({
                "text": f"User: {current_prompt}\n\nAssistant: {current_continuation}"
            })
        
        return jsonl_data
    
    def process_chapter_expansion(self, text_content):
        """处理章节大纲扩写数据格式"""
        jsonl_data = []
        lines = text_content.strip().split('\n')
        
        current_outline = ""
        current_content = ""
        in_content = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith("大纲:") or line.startswith("章节大纲:") or line.startswith("outline:"):
                # 保存之前的数据
                if current_outline and current_content:
                    jsonl_data.append({
                        "text": f"User: {current_outline}\n\nAssistant: {current_content}"
                    })
                
                current_outline = line.replace("大纲:", "").replace("章节大纲:", "").replace("outline:", "").strip()
                current_content = ""
                in_content = False
            elif line.startswith("内容:") or line.startswith("章节内容:") or line.startswith("完整内容:") or line.startswith("content:"):
                current_content = line.replace("内容:", "").replace("章节内容:", "").replace("完整内容:", "").replace("content:", "").strip()
                in_content = True
            elif in_content:
                current_content += "\n" + line
            else:
                if current_outline:
                    current_outline += "\n" + line
                else:
                    current_outline = line
        
        # 处理最后一组数据
        if current_outline and current_content:
            jsonl_data.append({
                "text": f"User: {current_outline}\n\nAssistant: {current_content}"
            })
        
        return jsonl_data
    
    def process_multi_turn(self, text_content):
        """处理多轮对话格式"""
        conversations = text_content.strip().split('\n\n')
        jsonl_data = []
        
        for conv in conversations:
            if not conv.strip():
                continue
                
            lines = conv.strip().split('\n')
            dialogue_parts = []
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                    
                if i % 2 == 0:  # 用户输入
                    dialogue_parts.append(f"User: {line}")
                else:  # 助手回复
                    dialogue_parts.append(f"Assistant: {line}")
            
            if dialogue_parts:
                jsonl_data.append({
                    "text": "\n\n".join(dialogue_parts)
                })
        
        return jsonl_data
    
    def process_instruction(self, text_content):
        """处理指令问答格式"""
        blocks = text_content.strip().split('\n\n\n')
        jsonl_data = []
        
        for block in blocks:
            if not block.strip():
                continue
                
            lines = block.strip().split('\n\n')
            if len(lines) >= 3:
                instruction = lines[0].strip()
                input_text = lines[1].strip()
                response = lines[2].strip()
                
                jsonl_data.append({
                    "text": f"Instruction: {instruction}\n\nInput: {input_text}\n\nResponse: {response}"
                })
            elif len(lines) == 2:
                instruction = lines[0].strip()
                response = lines[1].strip()
                
                jsonl_data.append({
                    "text": f"Instruction: {instruction}\n\nResponse: {response}"
                })
        
        return jsonl_data
    
    def process_long_text(self, text_content):
        """处理长文本格式"""
        # 按段落分割
        paragraphs = text_content.strip().split('\n\n')
        jsonl_data = []
        
        for para in paragraphs:
            para = para.strip()
            if para and len(para) > 50:  # 只保留较长的段落
                jsonl_data.append({
                    "text": para
                })
        
        return jsonl_data
    
    def process_article_with_title(self, text_content):
        """处理带标题的文章格式"""
        lines = text_content.strip().split('\n')
        jsonl_data = []
        
        current_title = ""
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 检测标题（假设标题用特殊标记或格式）
            if line.startswith('#') or line.startswith('《') and line.endswith('》'):
                if current_title and current_content:
                    content = '\n'.join(current_content)
                    jsonl_data.append({
                        "text": f"{current_title}\n{content}"
                    })
                
                current_title = line.strip('#').strip()
                if current_title.startswith('《') and current_title.endswith('》'):
                    current_title = current_title
                else:
                    current_title = f"《{current_title}》"
                current_content = []
            else:
                current_content.append(line)
        
        # 处理最后一篇文章
        if current_title and current_content:
            content = '\n'.join(current_content)
            jsonl_data.append({
                "text": f"{current_title}\n{content}"
            })
        
        return jsonl_data
    
    def save_jsonl(self, data, output_path):
        """保存JSONL文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    def process_directory(self, directory_path, data_format):
        """处理目录下的所有文本文件"""
        supported_extensions = ['.txt', '.json', '.jsonl']
        all_jsonl_data = []
        processed_files = []
        
        try:
            # 遍历目录下的所有文件
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_ext = os.path.splitext(file)[1].lower()
                    
                    if file_ext in supported_extensions:
                        try:
                            print(f"正在处理文件: {file_path}")
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            
                            # 根据格式处理数据
                            if data_format == "单轮问答":
                                file_data = self.process_single_qa(content)
                            elif data_format == "多轮对话":
                                file_data = self.process_multi_turn(content)
                            elif data_format == "指令问答":
                                file_data = self.process_instruction(content)
                            elif data_format == "长文本":
                                file_data = self.process_long_text(content)
                            elif data_format == "带标题文章":
                                file_data = self.process_article_with_title(content)
                            elif data_format == "小说段落续写":
                                file_data = self.process_novel_continuation(content)
                            elif data_format == "章节大纲扩写":
                                file_data = self.process_chapter_expansion(content)
                            else:
                                continue
                            
                            # 为每条数据添加文件来源信息
                            for item in file_data:
                                item['source_file'] = os.path.relpath(file_path, directory_path)
                            
                            all_jsonl_data.extend(file_data)
                            processed_files.append(file_path)
                            print(f"文件 {file} 处理完成，生成 {len(file_data)} 条数据")
                            
                        except Exception as e:
                            print(f"处理文件 {file_path} 时出错: {str(e)}")
                            continue
            
            return all_jsonl_data, processed_files
            
        except Exception as e:
            print(f"处理目录时出错: {str(e)}")
            return [], []
    
    def convert_to_binidx(self, jsonl_path, output_prefix):
        """转换JSONL到binidx格式"""
        try:
            import subprocess
            
            # 构建preprocess_data.py的路径
            preprocess_script = os.path.join(os.path.dirname(__file__), 'json2binidx_tool', 'tools', 'preprocess_data.py')
            
            # 构建命令行参数
            cmd = [
                sys.executable,  # 使用当前Python解释器
                preprocess_script,
                '--input', jsonl_path,
                '--output-prefix', output_prefix,
                '--vocab', self.vocab_file,
                '--dataset-impl', 'mmap',
                '--tokenizer-type', 'RWKVTokenizer',
                '--append-eod'
            ]
            
            # 执行转换命令
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            
            if result.returncode == 0:
                return True, "转换成功完成！"
            else:
                error_msg = result.stderr if result.stderr else result.stdout
                return False, f"转换失败: {error_msg}"
            
        except Exception as e:
            return False, f"转换失败: {str(e)}"

# 小说处理相关函数
def extract_keywords_with_ai(text, api_key, api_url=None, model=None):
    """使用AI API提取关键词"""
    if not api_key:
        return extract_keywords(text)  # 回退到本地方法
    
    # 使用默认值或用户提供的值
    api_url = api_url or DEFAULT_API_URL
    model = model or DEFAULT_MODEL
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        prompt = f"请从以下文本中提取3-5个最重要的关键词，用顿号分隔，只返回关键词，不要其他内容：\n\n{text[:500]}"  # 限制文本长度
        
        data = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 50,
            "temperature": 0.3
        }
        
        response = requests.post(api_url, headers=headers, json=data, timeout=10)
        
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

def extract_keywords_with_ai_custom(text, api_key, custom_instruction, api_url=None, model=None):
    """使用自定义指令的AI关键词提取"""
    if not api_key:
        return extract_keywords(text)
    
    # 使用默认值或用户提供的值
    api_url = api_url or DEFAULT_API_URL
    model = model or DEFAULT_MODEL
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        prompt = f"{custom_instruction}\n\n文本内容：{text[:500]}"
        
        data = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 100,
            "temperature": 0.3
        }
        
        response = requests.post(api_url, headers=headers, json=data, timeout=10)
        
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

def process_novel(file_obj, min_paragraph_length, max_paragraph_length, use_smart_split, use_ai_keywords, api_key, custom_instruction, api_url=None, model=None):
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
                keywords = extract_keywords_with_ai(response_text, api_key, api_url, model)
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

# 段落编辑相关函数
def load_jsonl_for_editing(file_obj):
    """加载JSONL文件用于编辑"""
    if file_obj is None:
        return [], "请上传JSONL文件", [], "请先上传JSONL文件"
    
    try:
        # 读取文件内容
        if hasattr(file_obj, 'read'):
            content = file_obj.read()
        else:
            with open(file_obj, 'r', encoding='utf-8') as f:
                content = f.read()
        
        if isinstance(content, bytes):
            content = content.decode('utf-8', errors='ignore')
        
        # 解析JSONL
        paragraph_data = []
        lines = content.strip().split('\n')
        
        for i, line in enumerate(lines):
            if line.strip():
                try:
                    data = json.loads(line)
                    # 解析text字段中的Instruction、Input、Response
                    text = data.get('text', '')
                    
                    # 简单解析格式
                    instruction_match = re.search(r'Instruction: (.*?)\n\nInput:', text, re.DOTALL)
                    input_match = re.search(r'Input: (.*?)\n\nResponse:', text, re.DOTALL)
                    response_match = re.search(r'Response: (.*?)$', text, re.DOTALL)
                    
                    instruction = instruction_match.group(1).strip() if instruction_match else ""
                    input_text = input_match.group(1).strip() if input_match else ""
                    response_text = response_match.group(1).strip() if response_match else ""
                    
                    # 从instruction中提取关键词
                    keywords_match = re.search(r"关键词'([^']*)'|关键词'([^']*)'|关键词'([^']*)'|关键词\"([^\"]*)\"|关键词\"([^\"]*)\"|关键词\"([^\"]*)\"", instruction)
                    keywords = ""
                    if keywords_match:
                        for group in keywords_match.groups():
                            if group:
                                keywords = group
                                break
                    
                    paragraph_data.append({
                        "index": i,
                        "instruction": instruction,
                        "input": input_text,
                        "response": response_text,
                        "keywords": keywords,
                        "original_text": text
                    })
                except json.JSONDecodeError:
                    continue
        
        if not paragraph_data:
            return [], "JSONL文件格式错误或为空", [], "JSONL文件格式错误或为空"
        
        # 生成统计信息
        stats = f"总段落数: {len(paragraph_data)}"
        
        # 生成选择选项
        choices = [f"{i}: {data['input'][:50]}..." for i, data in enumerate(paragraph_data)]
        
        # 生成HTML列表
        html_content = generate_paragraph_list_html(paragraph_data, [])
        
        return paragraph_data, stats, choices, html_content
        
    except Exception as e:
        return [], f"加载文件时出错: {str(e)}", [], f"加载文件时出错: {str(e)}"

def generate_paragraph_list_html(paragraph_data, selected_indices):
    """生成段落列表的HTML"""
    if not paragraph_data:
        return "请先上传JSONL文件"
    
    html = "<div style='max-height: 400px; overflow-y: auto;'>"
    
    for i, data in enumerate(paragraph_data):
        selected_class = "background-color: #e3f2fd;" if i in selected_indices else ""
        
        html += f"""
        <div style='border: 1px solid #ddd; margin: 5px 0; padding: 10px; border-radius: 5px; {selected_class}'>
            <div style='font-weight: bold; color: #1976d2;'>段落 {i + 1}</div>
            <div style='margin: 5px 0;'><strong>关键词:</strong> {data.get('keywords', '无')}</div>
            <div style='margin: 5px 0;'><strong>输入:</strong> {data.get('input', '')[:100]}{'...' if len(data.get('input', '')) > 100 else ''}</div>
            <div style='margin: 5px 0;'><strong>响应:</strong> {data.get('response', '')[:100]}{'...' if len(data.get('response', '')) > 100 else ''}</div>
        </div>
        """
    
    html += "</div>"
    return html

def select_all_paragraphs(paragraph_data):
    """全选段落"""
    if not paragraph_data:
        return []
    return list(range(len(paragraph_data)))

def deselect_all_paragraphs():
    """取消全选"""
    return []

def delete_selected_paragraphs(paragraph_data, selected_indices):
    """删除选中的段落"""
    if not selected_indices or not paragraph_data:
        return paragraph_data, "没有选择要删除的段落", [], generate_paragraph_list_html(paragraph_data, [])
    
    # 按索引倒序删除，避免索引变化问题
    selected_indices = sorted(selected_indices, reverse=True)
    new_paragraph_data = paragraph_data.copy()
    
    for idx in selected_indices:
        if 0 <= idx < len(new_paragraph_data):
            del new_paragraph_data[idx]
    
    # 重新编号
    for i, data in enumerate(new_paragraph_data):
        data['index'] = i
    
    # 更新选择选项
    choices = [f"{i}: {data['input'][:50]}..." for i, data in enumerate(new_paragraph_data)]
    
    # 生成新的HTML
    html_content = generate_paragraph_list_html(new_paragraph_data, [])
    
    stats = f"总段落数: {len(new_paragraph_data)} (已删除 {len(selected_indices)} 个段落)"
    
    return new_paragraph_data, stats, choices, html_content

def ai_regenerate_keywords_for_selected(paragraph_data, selected_indices, api_key, prompt_template, api_url, model):
    """为选中段落AI重新生成关键词"""
    if not selected_indices or not paragraph_data or not api_key:
        return paragraph_data, "请选择段落并填入API密钥", generate_paragraph_list_html(paragraph_data, selected_indices)
    
    updated_count = 0
    new_paragraph_data = paragraph_data.copy()
    
    for idx in selected_indices:
        if 0 <= idx < len(new_paragraph_data):
            response_text = new_paragraph_data[idx]['response']
            
            # 使用AI重新生成关键词
            new_keywords = extract_keywords_with_ai_custom(
                response_text, api_key, prompt_template, api_url, model
            )
            
            if new_keywords and new_keywords != new_paragraph_data[idx]['keywords']:
                new_paragraph_data[idx]['keywords'] = new_keywords
                # 更新instruction
                new_instruction = f"请根据关键词'{new_keywords}'续写小说段落"
                new_paragraph_data[idx]['instruction'] = new_instruction
                updated_count += 1
    
    # 生成新的HTML
    html_content = generate_paragraph_list_html(new_paragraph_data, selected_indices)
    
    stats = f"已更新 {updated_count} 个段落的关键词"
    
    return new_paragraph_data, stats, html_content

def convert_jsonl_to_binidx(jsonl_file, output_prefix, tokenizer_type="RWKVTokenizer"):
    """将JSONL文件转换为binidx格式"""
    try:
        # 检查json2binidx工具是否存在，使用绝对路径
        current_dir = Path(__file__).parent
        tool_dir = current_dir / "json2binidx_tool"
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

# 创建主界面
def create_interface():
    processor = DataProcessor()
    
    with gr.Blocks(title="RWKV数据集处理工具", theme=gr.themes.Ocean()) as demo:
        gr.Markdown("# 🚀 RWKV数据集处理工具")
        gr.Markdown("集成通用数据处理、小说类增强处理和数据格式转换功能的一站式工具")
        
        with gr.Tabs():
            # 通用数据处理标签页
            with gr.TabItem("📋 通用数据处理"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📁 数据输入")
                        
                        input_mode = gr.Radio(
                            choices=["文件上传", "目录处理"],
                            label="输入模式",
                            value="文件上传",
                            info="选择处理单个文件还是整个目录"
                        )
                        
                        file_input = gr.File(
                            label="上传文本文件",
                            file_types=[".txt", ".json", ".jsonl"],
                            type="binary",
                            visible=True
                        )
                        
                        directory_input = gr.Textbox(
                            label="目录路径",
                            placeholder="请输入要处理的目录完整路径，例如：D:\\data\\texts",
                            info="目录下的所有.txt、.json、.jsonl文件将被处理并合并",
                            visible=False
                        )
                        
                        data_format = gr.Dropdown(
                            choices=["单轮问答", "多轮对话", "指令问答", "长文本", "带标题文章", "小说段落续写", "章节大纲扩写"],
                            label="数据格式",
                            value="单轮问答"
                        )
                        
                        system_prompt = gr.Textbox(
                            label="系统提示（可选）",
                            placeholder="例如：你是一个有用的AI助手...",
                            lines=3
                        )
                        
                        process_btn = gr.Button("🔄 处理数据", variant="primary")
                        
                    with gr.Column(scale=2):
                        gr.Markdown("### 📊 处理结果")
                        
                        status_output = gr.Textbox(
                            label="处理状态",
                            interactive=False
                        )
                        
                        preview_output = gr.Textbox(
                            label="数据预览",
                            lines=10,
                            interactive=False
                        )
                        
                        jsonl_content = gr.Textbox(
                            label="JSONL内容",
                            lines=5,
                            interactive=False
                        )
                
                # 格式说明
                with gr.Accordion("📖 数据格式说明", open=False):
                    gr.Markdown(
                        """
                        ### 单轮问答格式
                        每行一个问答对，用制表符或竖线分隔：
                        ```
                        问题1    答案1
                        问题2    答案2
                        ```
                        
                        ### 多轮对话格式
                        每个对话用空行分隔，奇数行为用户输入，偶数行为助手回复：
                        ```
                        你好
                        你好！很高兴见到你
                        今天天气怎么样？
                        今天天气很好
                        
                        新的对话开始
                        好的
                        ```
                        
                        ### 指令问答格式
                        每个样本用三个空行分隔：
                        ```
                        请总结以下内容
                        
                        这是需要总结的内容...
                        
                        这是总结结果...
                        
                        
                        
                        下一个指令
                        
                        输入内容
                        
                        响应内容
                        ```
                        
                        ### 长文本格式
                        直接输入文章内容，工具会自动按段落分割。
                        
                        ### 带标题文章格式
                        标题用#开头或用《》包围：
                        ```
                        # 文章标题1
                        文章内容...
                        
                        《文章标题2》
                        文章内容...
                        ```
                        """
                    )
            
            # 小说类增强处理标签页
            with gr.TabItem("📚 小说类增强处理"):
                # 状态变量存储段落数据
                paragraph_data_state = gr.State([])
                
                with gr.Tabs():
                    # 文件处理子标签页
                    with gr.TabItem("📄 文件处理"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("### 📁 文件上传")
                                novel_file_input = gr.File(
                                    label="上传小说文件",
                                    file_types=[".txt", ".md"],
                                    type="filepath"
                                )
                                
                                novel_directory_input = gr.Textbox(
                                    label="小说目录路径（可选）",
                                    placeholder="输入包含多个小说文件的目录路径",
                                    info="批量处理目录下的所有小说文件"
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
                                    info="使用AI API提取更准确的关键词"
                                )
                                
                                with gr.Group(visible=False) as ai_settings_group:
                                    api_key_input = gr.Textbox(
                                        label="API Key",
                                        type="password",
                                        placeholder="sk-...",
                                        info="输入您的API密钥"
                                    )
                                    
                                    api_url_input = gr.Textbox(
                                        label="API URL",
                                        value=DEFAULT_API_URL,
                                        placeholder="https://api.deepseek.com/v1/chat/completions",
                                        info="支持OpenAI兼容接口，如vLLM服务"
                                    )
                                    
                                    model_input = gr.Textbox(
                                        label="模型名称",
                                        value=DEFAULT_MODEL,
                                        placeholder="deepseek-chat",
                                        info="指定要使用的模型名称"
                                    )
                                
                                custom_instruction = gr.Textbox(
                                    label="自定义指令模板",
                                    placeholder="请根据关键词'{keywords}'续写小说段落",
                                    info="使用{keywords}作为关键词占位符"
                                )
                                
                                novel_process_btn = gr.Button("🚀 处理小说", variant="primary", size="lg")
                                
                            with gr.Column(scale=2):
                                novel_status_output = gr.Textbox(
                                    label="📊 处理状态",
                                    interactive=False
                                )
                                
                                novel_jsonl_output = gr.Textbox(
                                    label="📄 生成的JSONL数据",
                                    lines=15,
                                    max_lines=25,
                                    interactive=False
                                )
                                
                                with gr.Row():
                                    novel_filename_input = gr.Textbox(
                                        value="novel_training_data.jsonl",
                                        label="💾 保存文件名",
                                        scale=3
                                    )
                                    novel_save_btn = gr.Button("💾 保存JSONL", scale=1)
                                
                                novel_save_status = gr.Textbox(
                                    label="💾 保存状态",
                                    interactive=False
                                )
                    
                    # 段落编辑子标签页
                    with gr.TabItem("✏️ 段落编辑"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("### 📂 JSONL文件上传")
                                edit_jsonl_file = gr.File(
                                    label="上传JSONL文件",
                                    file_types=[".jsonl"]
                                )
                                
                                gr.Markdown("### 📊 段落统计")
                                paragraph_stats = gr.Textbox(
                                    label="段落统计信息",
                                    interactive=False
                                )
                                
                                gr.Markdown("### 🎯 选择管理")
                                with gr.Row():
                                    select_all_btn = gr.Button("全选", size="sm")
                                    deselect_all_btn = gr.Button("取消全选", size="sm")
                                
                                selected_indices = gr.CheckboxGroup(
                                    label="手动选择段落",
                                    choices=[],
                                    value=[]
                                )
                                
                                with gr.Row():
                                    delete_selected_btn = gr.Button("🗑️ 删除选中", variant="stop")
                                    ai_regenerate_btn = gr.Button("🤖 AI重新生成关键词", variant="secondary")
                                
                                gr.Markdown("### 🤖 AI设置")
                                edit_prompt_template = gr.Textbox(
                                    label="提示词模板",
                                    value="请从以下文本中提取3-5个最重要的关键词，用顿号分隔，只返回关键词：",
                                    lines=2
                                )
                                
                                edit_api_key = gr.Textbox(
                                    label="API Key",
                                    type="password",
                                    placeholder="sk-..."
                                )
                                
                                edit_api_url = gr.Textbox(
                                    label="API URL",
                                    value=DEFAULT_API_URL,
                                    placeholder="https://api.deepseek.com/v1/chat/completions"
                                )
                                
                                edit_model = gr.Textbox(
                                    label="模型名称",
                                    value=DEFAULT_MODEL,
                                    placeholder="deepseek-chat"
                                )
                            
                            with gr.Column(scale=2):
                                gr.Markdown("### 📝 段落列表")
                                paragraph_list = gr.HTML(
                                    label="段落列表",
                                    value="请先上传JSONL文件"
                                )
                                
                                gr.Markdown("### ✏️ 编辑指令")
                                edit_instruction = gr.Textbox(
                                    label="编辑说明",
                                    placeholder="选择段落后，可以在这里查看和编辑指令内容",
                                    lines=3
                                )
                                
                                with gr.Row():
                                    update_paragraph_btn = gr.Button("📝 更新段落", variant="primary")
                                    regenerate_paragraph_btn = gr.Button("🔄 重新生成", variant="secondary")
                                
                                gr.Markdown("### 👀 预览")
                                with gr.Row():
                                    with gr.Column():
                                        input_preview = gr.Textbox(
                                            label="输入内容",
                                            lines=5,
                                            interactive=False
                                        )
                                    with gr.Column():
                                        response_preview = gr.Textbox(
                                            label="响应内容",
                                            lines=5,
                                            interactive=False
                                        )
            
            # 数据格式转换标签页
            with gr.TabItem("🔄 数据格式转换"):
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
                            value="./data/training_data",
                            placeholder="./data/training_data"
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
                        
                        gr.Markdown("### 💾 下载选项")
                        
                        download_output_name = gr.Textbox(
                            label="输出文件名",
                            value="training_data",
                            placeholder="不包含扩展名"
                        )
                        
                        download_output_path = gr.Textbox(
                            label="输出路径（可选）",
                            placeholder="留空则使用默认路径（用户目录/RWKV_Downloads）",
                            info="指定自定义输出目录的完整路径"
                        )
                        
                        download_jsonl = gr.DownloadButton(
                            label="📥 下载JSONL文件",
                            variant="secondary"
                        )
                        
                        download_status = gr.Textbox(
                            label="下载状态",
                            interactive=False,
                            visible=False
                        )
        
        # 通用数据处理事件绑定
        def toggle_input_mode(mode):
            if mode == "文件上传":
                return gr.update(visible=True), gr.update(visible=False)
            else:  # 目录处理
                return gr.update(visible=False), gr.update(visible=True)
        
        def process_file(file_obj, data_format, system_prompt=""):
            if file_obj is None:
                return "请上传文件", "", ""
            
            try:
                # 读取文件内容
                if hasattr(file_obj, 'read'):
                    if hasattr(file_obj, 'name'):
                        print(f"正在处理文件: {file_obj.name}")
                    content = file_obj.read()
                    if isinstance(content, bytes):
                        content = content.decode('utf-8')
                elif isinstance(file_obj, bytes):
                    content = file_obj.decode('utf-8')
                elif isinstance(file_obj, str):
                    with open(file_obj, 'r', encoding='utf-8') as f:
                        content = f.read()
                else:
                    return f"不支持的文件类型: {type(file_obj)}", "", ""
                
                # 根据格式处理数据
                if data_format == "单轮问答":
                    jsonl_data = processor.process_single_qa(content)
                elif data_format == "多轮对话":
                    jsonl_data = processor.process_multi_turn(content)
                elif data_format == "指令问答":
                    jsonl_data = processor.process_instruction(content)
                elif data_format == "长文本":
                    jsonl_data = processor.process_long_text(content)
                elif data_format == "带标题文章":
                    jsonl_data = processor.process_article_with_title(content)
                elif data_format == "小说段落续写":
                    jsonl_data = processor.process_novel_continuation(content)
                elif data_format == "章节大纲扩写":
                    jsonl_data = processor.process_chapter_expansion(content)
                else:
                    return "不支持的数据格式", "", ""
                
                # 如果有系统提示，添加到每条数据中
                if system_prompt.strip():
                    for item in jsonl_data:
                        if data_format in ["单轮问答", "多轮对话"]:
                            item["text"] = f"System: {system_prompt.strip()}\n\n{item['text']}"
                
                # 生成预览
                preview = "\n".join([json.dumps(item, ensure_ascii=False, indent=2) for item in jsonl_data[:3]])
                if len(jsonl_data) > 3:
                    preview += f"\n\n... 还有 {len(jsonl_data) - 3} 条数据"
                
                # 生成下载内容
                download_content = "\n".join([json.dumps(item, ensure_ascii=False) for item in jsonl_data])
                
                return f"处理完成！共生成 {len(jsonl_data)} 条数据", preview, download_content
                
            except Exception as e:
                return f"处理失败: {str(e)}", "", ""
        
        def process_directory(directory_path, data_format, system_prompt=""):
            if not directory_path or not os.path.exists(directory_path):
                return "请提供有效的目录路径", "", ""
            
            if not os.path.isdir(directory_path):
                return "提供的路径不是目录", "", ""
            
            try:
                # 处理目录下的所有文件
                jsonl_data, processed_files = processor.process_directory(directory_path, data_format)
                
                if not jsonl_data:
                    return "目录中没有找到可处理的文件或处理失败", "", ""
                
                # 如果有系统提示，添加到每条数据中
                if system_prompt.strip():
                    for item in jsonl_data:
                        if data_format in ["单轮问答", "多轮对话"]:
                            item["text"] = f"System: {system_prompt.strip()}\n\n{item['text']}"
                
                # 生成预览
                preview = "\n".join([json.dumps(item, ensure_ascii=False, indent=2) for item in jsonl_data[:3]])
                if len(jsonl_data) > 3:
                    preview += f"\n\n... 还有 {len(jsonl_data) - 3} 条数据"
                
                # 添加处理文件列表到预览
                files_info = f"\n\n已处理的文件 ({len(processed_files)} 个):\n" + "\n".join([f"- {os.path.basename(f)}" for f in processed_files[:10]])
                if len(processed_files) > 10:
                    files_info += f"\n... 还有 {len(processed_files) - 10} 个文件"
                preview += files_info
                
                # 生成下载内容
                download_content = "\n".join([json.dumps(item, ensure_ascii=False) for item in jsonl_data])
                
                return f"处理完成！共处理 {len(processed_files)} 个文件，生成 {len(jsonl_data)} 条数据", preview, download_content
                
            except Exception as e:
                return f"处理失败: {str(e)}", "", ""
        
        def process_data(input_mode, file_input, directory_input, data_format, system_prompt):
            if input_mode == "文件上传":
                return process_file(file_input, data_format, system_prompt)
            else:  # 目录处理
                return process_directory(directory_input, data_format, system_prompt)
        
        def prepare_jsonl_download(jsonl_content, output_name, custom_output_path=""):
            """准备JSONL文件下载"""
            if not jsonl_content.strip():
                return None, gr.update(visible=False)
            
            if not output_name.strip():
                output_name = "training_data"
            
            # 确定保存路径
            if custom_output_path.strip():
                download_dir = custom_output_path.strip()
            else:
                user_home = os.path.expanduser("~")
                download_dir = os.path.join(user_home, "RWKV_Downloads")
            
            # 创建目录
            try:
                os.makedirs(download_dir, exist_ok=True)
            except Exception as e:
                # 如果创建目录失败，使用临时目录
                download_dir = tempfile.gettempdir()
            
            # 创建文件路径
            jsonl_file_path = os.path.join(download_dir, f"{output_name}.jsonl")
            
            # 保存文件
            try:
                with open(jsonl_file_path, 'w', encoding='utf-8') as f:
                    f.write(jsonl_content)
                status_msg = f"JSONL文件已保存到: {jsonl_file_path}"
                return jsonl_file_path, gr.update(value=status_msg, visible=True)
            except Exception as e:
                # 如果保存失败，创建临时文件
                temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8')
                temp_file.write(jsonl_content)
                temp_file.close()
                status_msg = f"保存到自定义路径失败，已保存到临时文件: {temp_file.name}"
                return temp_file.name, gr.update(value=status_msg, visible=True)
        
        def convert_to_binidx_wrapper(jsonl_content, output_name, custom_output_path=""):
            if not jsonl_content.strip():
                return "请先处理数据生成JSONL内容"
            
            if not output_name.strip():
                output_name = "output"
            
            try:
                # 创建临时目录
                temp_dir = tempfile.mkdtemp()
                jsonl_path = os.path.join(temp_dir, "temp.jsonl")
                output_prefix = os.path.join(temp_dir, output_name)
                
                # 保存JSONL文件
                with open(jsonl_path, 'w', encoding='utf-8') as f:
                    f.write(jsonl_content)
                
                # 转换为binidx
                success, message = processor.convert_to_binidx(jsonl_path, output_prefix)
                
                if success:
                    # 根据preprocess_data.py的输出格式，文件名是 {output_prefix}_text_document.bin/idx
                    bin_file = output_prefix + "_text_document.bin"
                    idx_file = output_prefix + "_text_document.idx"
                    
                    if os.path.exists(bin_file) and os.path.exists(idx_file):
                        # 确定输出目录
                        if custom_output_path.strip():
                            download_dir = custom_output_path.strip()
                        else:
                            user_home = os.path.expanduser("~")
                            download_dir = os.path.join(user_home, "RWKV_Downloads")
                        
                        # 创建输出目录
                        try:
                            os.makedirs(download_dir, exist_ok=True)
                        except PermissionError:
                            return f"权限错误：无法创建目录 {download_dir}，请检查路径权限或使用其他路径"
                        except Exception as e:
                            return f"创建目录失败：{str(e)}"
                        
                        # 复制文件到下载目录
                        final_bin = os.path.join(download_dir, f"{output_name}.bin")
                        final_idx = os.path.join(download_dir, f"{output_name}.idx")
                        
                        shutil.copy2(bin_file, final_bin)
                        shutil.copy2(idx_file, final_idx)
                        
                        return f"转换成功！文件已保存到:\n{download_dir}\\{output_name}.bin\n{download_dir}\\{output_name}.idx"
                    else:
                        return "转换失败：未找到生成的文件"
                else:
                    return message
                    
            except Exception as e:
                return f"转换失败: {str(e)}"
            finally:
                # 清理临时文件
                if 'temp_dir' in locals():
                    shutil.rmtree(temp_dir, ignore_errors=True)
        
        # 小说处理相关函数
        def toggle_api_key_visibility(use_ai):
            return gr.update(visible=use_ai)
        
        def process_novel_and_update(file_obj, min_len, max_len, smart_split, ai_keywords, api_key, custom_instr):
            status, jsonl, paragraphs = process_novel(file_obj, min_len, max_len, smart_split, ai_keywords, api_key, custom_instr)
            return status, jsonl, paragraphs
        
        # 事件绑定
        input_mode.change(
            fn=toggle_input_mode,
            inputs=[input_mode],
            outputs=[file_input, directory_input]
        )
        
        process_btn.click(
            fn=process_data,
            inputs=[input_mode, file_input, directory_input, data_format, system_prompt],
            outputs=[status_output, preview_output, jsonl_content]
        )
        
        # 小说处理事件
        use_ai_keywords.change(
            fn=toggle_api_key_visibility,
            inputs=[use_ai_keywords],
            outputs=[api_key_input]
        )
        
        novel_process_btn.click(
            fn=process_novel_and_update,
            inputs=[novel_file_input, min_length, max_length, use_smart_split, use_ai_keywords, api_key_input, custom_instruction],
            outputs=[novel_status_output, novel_jsonl_output, paragraph_data_state]
        )
        
        novel_save_btn.click(
            fn=save_jsonl,
            inputs=[novel_jsonl_output, novel_filename_input],
            outputs=[novel_save_status]
        )
        
        # 段落编辑相关事件
        edit_jsonl_file.upload(
            fn=load_jsonl_for_editing,
            inputs=[edit_jsonl_file],
            outputs=[edit_paragraph_data, edit_paragraph_stats, edit_paragraph_list]
        )
        
        edit_select_all_btn.click(
            fn=select_all_paragraphs,
            inputs=[edit_paragraph_data],
            outputs=[edit_selected_indices, edit_paragraph_list]
        )
        
        edit_deselect_all_btn.click(
            fn=deselect_all_paragraphs,
            inputs=[],
            outputs=[edit_selected_indices, edit_paragraph_list]
        )
        
        edit_delete_btn.click(
            fn=delete_selected_paragraphs,
            inputs=[edit_paragraph_data, edit_selected_indices],
            outputs=[edit_paragraph_data, edit_paragraph_stats, edit_paragraph_list, edit_selected_indices]
        )
        
        edit_ai_regen_btn.click(
            fn=ai_regenerate_keywords_for_selected,
            inputs=[edit_paragraph_data, edit_selected_indices, edit_api_key, edit_prompt_template, edit_api_url, edit_model],
            outputs=[edit_paragraph_data, edit_paragraph_list, edit_ai_status]
        )
        
        # 转换相关事件
        download_jsonl.click(
            fn=prepare_jsonl_download,
            inputs=[jsonl_content, download_output_name, download_output_path],
            outputs=[download_jsonl, download_status]
        )
        
        convert_btn.click(
            fn=convert_to_binidx_wrapper,
            inputs=[jsonl_content, download_output_name, download_output_path],
            outputs=[convert_status]
        )
    
    return demo

if __name__ == "__main__":
    # 检查必要文件
    vocab_file = os.path.join(os.path.dirname(__file__), 'json2binidx_tool', 'rwkv_vocab_v20230424.txt')
    if not os.path.exists(vocab_file):
        print(f"错误：找不到词汇文件 {vocab_file}")
        print("请确保json2binidx_tool目录存在且包含rwkv_vocab_v20230424.txt文件")
        sys.exit(1)
    
    # 确保jieba已安装
    try:
        import jieba
    except ImportError:
        print("请先安装jieba: pip install jieba")
        exit(1)
    
    demo = create_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7863,
        share=False,
        show_error=True
    )