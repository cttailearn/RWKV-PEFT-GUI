import gradio as gr
import json
import re
import os
import requests
import subprocess
import sys
from pathlib import Path
import openai
from typing import Dict, Any, Optional, List, Tuple, Generator
import logging
from dataclasses import dataclass
from urllib.parse import urlparse
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import gc
import psutil
from datetime import datetime

# 可选依赖
try:
    import chardet
    HAS_CHARDET = True
except ImportError:
    HAS_CHARDET = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('novel_dataset_processing.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# ==================== 进度管理器 ====================

class ProgressManager:
    """进度管理器 - 处理大文件时的进度显示和日志输出"""
    
    def __init__(self):
        self.total_files = 0
        self.processed_files = 0
        self.total_size = 0
        self.processed_size = 0
        self.start_time = None
        self.current_file = ""
        self.status_queue = Queue()
        self.is_processing = False
        
    def start_processing(self, total_files: int, total_size: int):
        """开始处理"""
        self.total_files = total_files
        self.processed_files = 0
        self.total_size = total_size
        self.processed_size = 0
        self.start_time = time.time()
        self.is_processing = True
        
        # 记录开始信息
        memory_info = psutil.virtual_memory()
        logger.info(f"="*60)
        logger.info(f"开始处理大量文本数据")
        logger.info(f"总文件数: {total_files}")
        logger.info(f"总大小: {self.format_size(total_size)}")
        logger.info(f"当前内存使用: {self.format_size(memory_info.used)} / {self.format_size(memory_info.total)}")
        logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"="*60)
        
        print(f"\n🚀 开始处理 {total_files} 个文件，总大小 {self.format_size(total_size)}")
        print(f"📊 内存使用: {self.format_size(memory_info.used)} / {self.format_size(memory_info.total)}")
        
    def update_file_progress(self, file_path: str, file_size: int):
        """更新文件处理进度"""
        self.processed_files += 1
        self.processed_size += file_size
        self.current_file = file_path
        
        # 计算进度
        file_progress = (self.processed_files / self.total_files) * 100
        size_progress = (self.processed_size / self.total_size) * 100 if self.total_size > 0 else 0
        
        # 计算速度和预估时间
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            speed = self.processed_size / elapsed_time
            remaining_size = self.total_size - self.processed_size
            eta = remaining_size / speed if speed > 0 else 0
        else:
            speed = 0
            eta = 0
            
        # 内存使用情况
        memory_info = psutil.virtual_memory()
        
        # 日志输出
        logger.info(f"处理文件 [{self.processed_files}/{self.total_files}]: {os.path.basename(file_path)}")
        logger.info(f"文件大小: {self.format_size(file_size)}, 累计处理: {self.format_size(self.processed_size)}")
        logger.info(f"进度: {file_progress:.1f}% (按文件) | {size_progress:.1f}% (按大小)")
        logger.info(f"处理速度: {self.format_size(speed)}/s, 预计剩余: {self.format_time(eta)}")
        logger.info(f"内存使用: {self.format_size(memory_info.used)} ({memory_info.percent:.1f}%)")
        
        # 终端输出
        print(f"\r📁 [{self.processed_files}/{self.total_files}] {os.path.basename(file_path)[:50]:<50} "
              f"{file_progress:5.1f}% | {self.format_size(speed)}/s | ETA: {self.format_time(eta)}", end="", flush=True)
        
        # 内存警告
        if memory_info.percent > 80:
            logger.warning(f"内存使用率过高: {memory_info.percent:.1f}%，建议释放内存")
            print(f"\n⚠️  内存使用率: {memory_info.percent:.1f}%")
            
    def update_paragraph_progress(self, current: int, total: int, file_name: str):
        """更新段落处理进度"""
        progress = (current / total) * 100 if total > 0 else 0
        
        if current % 100 == 0 or current == total:  # 每100个段落或完成时输出
            logger.info(f"文件 {os.path.basename(file_name)} - 段落处理进度: {current}/{total} ({progress:.1f}%)")
            print(f"\r  📝 处理段落: {current}/{total} ({progress:5.1f}%)", end="", flush=True)
            
    def log_memory_usage(self, context: str = ""):
        """记录内存使用情况"""
        memory_info = psutil.virtual_memory()
        process = psutil.Process()
        process_memory = process.memory_info()
        
        logger.info(f"内存使用情况 {context}:")
        logger.info(f"  系统内存: {self.format_size(memory_info.used)} / {self.format_size(memory_info.total)} ({memory_info.percent:.1f}%)")
        logger.info(f"  进程内存: {self.format_size(process_memory.rss)}")
        
    def finish_processing(self, total_paragraphs: int, total_training_data: int):
        """完成处理"""
        self.is_processing = False
        elapsed_time = time.time() - self.start_time
        
        logger.info(f"="*60)
        logger.info(f"处理完成！")
        logger.info(f"总耗时: {self.format_time(elapsed_time)}")
        logger.info(f"处理文件: {self.processed_files} 个")
        logger.info(f"处理数据: {self.format_size(self.processed_size)}")
        logger.info(f"生成段落: {total_paragraphs} 个")
        logger.info(f"训练数据: {total_training_data} 条")
        logger.info(f"平均速度: {self.format_size(self.processed_size / elapsed_time)}/s")
        logger.info(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"="*60)
        
        print(f"\n\n✅ 处理完成！")
        print(f"⏱️  总耗时: {self.format_time(elapsed_time)}")
        print(f"📊 处理了 {self.processed_files} 个文件，{self.format_size(self.processed_size)} 数据")
        print(f"📝 生成 {total_paragraphs} 个段落，{total_training_data} 条训练数据")
        print(f"🚀 平均速度: {self.format_size(self.processed_size / elapsed_time)}/s")
        
    @staticmethod
    def format_size(size_bytes: float) -> str:
        """格式化文件大小"""
        if size_bytes == 0:
            return "0B"
        
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f}PB"
        
    @staticmethod
    def format_time(seconds: float) -> str:
        """格式化时间"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"

# ==================== AI模型配置 ====================

@dataclass
class ModelConfig:
    """AI模型配置类"""
    name: str
    base_url: str
    api_key: str
    model_name: str
    timeout: int = 60
    max_retries: int = 3
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    
    def __post_init__(self):
        """配置验证"""
        if not self.base_url or not self.api_key or not self.model_name:
            raise ValueError("base_url, api_key, model_name 不能为空")
        
        # 验证URL格式
        parsed = urlparse(self.base_url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("base_url 格式无效")

class AIModelClient:
    """AI模型客户端"""
    @classmethod
    def load_custom_configs(cls) -> Dict[str, Dict]:
        """加载自定义模型配置"""
        config_file = "novel_model_configs.json"
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"加载模型配置失败: {str(e)}")
        return {}
    
    @classmethod
    def save_custom_config(cls, name: str, config: Dict[str, Any]):
        """保存自定义模型配置"""
        config_file = "novel_model_configs.json"
        try:
            configs = cls.load_custom_configs()
            configs[name] = config
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(configs, f, ensure_ascii=False, indent=2)
            logger.info(f"模型配置 {name} 已保存")
        except Exception as e:
            logger.error(f"保存模型配置失败: {str(e)}")
    
    @classmethod
    def get_all_configs(cls) -> Dict[str, Dict]:
        """获取所有模型配置（仅自定义）"""
        # 只返回自定义配置，不再有预设模型
        return cls.load_custom_configs()
    
    @classmethod
    def delete_config(cls, name: str) -> bool:
        """删除模型配置"""
        try:
            configs = cls.load_custom_configs()
            if name in configs:
                del configs[name]
                config_file = "novel_model_configs.json"
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(configs, f, ensure_ascii=False, indent=2)
                logger.info(f"模型配置 {name} 已删除")
                return True
            return False
        except Exception as e:
            logger.error(f"删除模型配置失败: {str(e)}")
            return False
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.client = openai.OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout,
            max_retries=config.max_retries
        )
        logger.info(f"AI模型客户端已初始化: {config.name}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError))
    )
    def extract_keywords(self, text: str, prompt: str) -> str:
        """使用AI模型提取关键词"""
        try:
            messages = [
                {"role": "system", "content": "你是一个专业的小说分析助手，擅长从文本中提取关键词。"},
                {"role": "user", "content": f"{prompt}\n\n文本内容：{text[:500]}"}
            ]
            
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens or 50
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"AI关键词提取失败: {str(e)}")
            raise
    
    def test_connection(self) -> bool:
        """测试连接"""
        try:
            response = self.extract_keywords("测试文本", "请回复'连接成功'")
            return "连接成功" in response or "成功" in response
        except Exception:
            return False

# ==================== 提示词任务管理 ====================

class PromptTaskManager:
    """提示词任务管理器"""
    
    def __init__(self):
        self.tasks_file = "novel_prompt_tasks.json"
        self.default_tasks = {
            "小说关键词提取": "请从以下小说文本中提取3-5个最重要的关键词，用顿号分隔，只返回关键词，不要其他内容。",
            "武侠小说关键词": "请从以下武侠小说文本中提取关键词，重点关注武功、人物、场景等元素，用顿号分隔。",
            "现代小说关键词": "请从以下现代小说文本中提取关键词，重点关注情感、场景、人物关系等，用顿号分隔。",
            "玄幻小说关键词": "请从以下玄幻小说文本中提取关键词，重点关注修炼、法术、境界等元素，用顿号分隔。",
            "言情小说关键词": "请从以下言情小说文本中提取关键词，重点关注情感、关系、场景等，用顿号分隔。",
            "科幻小说关键词": "请从以下科幻小说文本中提取关键词，重点关注科技、未来、探索等元素，用顿号分隔。"
        }
        self.tasks = self.load_tasks()
    
    def load_tasks(self) -> Dict[str, str]:
        """加载提示词任务"""
        tasks = self.default_tasks.copy()
        
        try:
            if os.path.exists(self.tasks_file):
                with open(self.tasks_file, 'r', encoding='utf-8') as f:
                    saved_tasks = json.load(f)
                tasks.update(saved_tasks)
                logger.info(f"已加载 {len(saved_tasks)} 个自定义任务")
        except Exception as e:
            logger.warning(f"加载任务文件失败: {str(e)}")
        
        return tasks
    
    def save_tasks(self):
        """保存提示词任务"""
        try:
            custom_tasks = {k: v for k, v in self.tasks.items() if k not in self.default_tasks or v != self.default_tasks.get(k)}
            with open(self.tasks_file, 'w', encoding='utf-8') as f:
                json.dump(custom_tasks, f, ensure_ascii=False, indent=2)
            logger.info(f"已保存 {len(custom_tasks)} 个任务")
        except Exception as e:
            logger.error(f"保存任务文件失败: {str(e)}")
    
    def add_task(self, name: str, prompt: str):
        """添加或更新任务"""
        self.tasks[name] = prompt
        self.save_tasks()
        logger.info(f"任务 '{name}' 已保存")
    
    def delete_task(self, name: str) -> bool:
        """删除任务"""
        if name in self.default_tasks:
            self.tasks[name] = self.default_tasks[name]
            self.save_tasks()
            logger.info(f"默认任务 '{name}' 已重置")
            return True
        
        if name in self.tasks:
            del self.tasks[name]
            self.save_tasks()
            logger.info(f"自定义任务 '{name}' 已删除")
            return True
        return False
    
    def get_task_names(self) -> List[str]:
        """获取所有任务名称"""
        return list(self.tasks.keys())
    
    def get_task_prompt(self, name: str) -> str:
        """获取任务提示词"""
        return self.tasks.get(name, "")
    
    def is_default_task(self, name: str) -> bool:
        """判断是否为默认任务"""
        return name in self.default_tasks

# ==================== 文件处理模块 ====================

class FileProcessor:
    """文件处理器 - 优化大文件处理"""
    
    # 大文件处理配置
    CHUNK_SIZE = 8 * 1024 * 1024  # 8MB 块大小
    MAX_MEMORY_USAGE = 2 * 1024 * 1024 * 1024  # 2GB 最大内存使用
    LARGE_FILE_THRESHOLD = 100 * 1024 * 1024  # 100MB 大文件阈值
    
    @staticmethod
    def detect_encoding(file_path: str) -> str:
        """检测文件编码"""
        if HAS_CHARDET:
            try:
                with open(file_path, 'rb') as f:
                    raw_data = f.read(10000)
                result = chardet.detect(raw_data)
                if result['encoding'] and result['confidence'] > 0.7:
                    return result['encoding']
            except Exception:
                pass
        
        encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'gb18030', 'big5', 'latin1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    f.read(1000)  # 只读取少量内容进行测试
                return encoding
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        return 'utf-8'
    
    @staticmethod
    def get_file_size(file_path: str) -> int:
        """获取文件大小"""
        try:
            return os.path.getsize(file_path)
        except Exception:
            return 0
    
    @staticmethod
    def read_text_file_streaming(file_path: str) -> Generator[str, None, None]:
        """流式读取大文本文件"""
        try:
            encoding = FileProcessor.detect_encoding(file_path)
            file_size = FileProcessor.get_file_size(file_path)
            
            logger.info(f"开始流式读取文件: {file_path} ({ProgressManager.format_size(file_size)})")
            
            with open(file_path, 'r', encoding=encoding) as f:
                while True:
                    chunk = f.read(FileProcessor.CHUNK_SIZE)
                    if not chunk:
                        break
                    yield chunk
                    
        except Exception as e:
            logger.error(f"流式读取文件失败 {file_path}: {str(e)}")
            raise Exception(f"读取文件失败: {str(e)}")
    
    @staticmethod
    def read_text_file(file_path: str) -> str:
        """读取文本文件 - 智能选择读取方式"""
        try:
            file_size = FileProcessor.get_file_size(file_path)
            
            # 小文件直接读取
            if file_size < FileProcessor.LARGE_FILE_THRESHOLD:
                encoding = FileProcessor.detect_encoding(file_path)
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            
            # 大文件流式读取并拼接
            logger.info(f"检测到大文件 {file_path} ({ProgressManager.format_size(file_size)})，使用流式读取")
            content_chunks = []
            total_size = 0
            
            for chunk in FileProcessor.read_text_file_streaming(file_path):
                content_chunks.append(chunk)
                total_size += len(chunk.encode('utf-8'))
                
                # 内存使用检查
                memory_info = psutil.virtual_memory()
                if memory_info.percent > 85:  # 内存使用超过85%
                    logger.warning(f"内存使用率过高 ({memory_info.percent:.1f}%)，暂停读取")
                    gc.collect()  # 强制垃圾回收
                    time.sleep(0.1)  # 短暂暂停
                    
                if total_size > FileProcessor.MAX_MEMORY_USAGE:
                    logger.warning(f"文件过大，已读取 {ProgressManager.format_size(total_size)}，停止读取")
                    break
            
            return ''.join(content_chunks)
            
        except Exception as e:
            raise Exception(f"读取文件失败: {str(e)}")
    
    @staticmethod
    def scan_directory(directory_path: str) -> List[Tuple[str, int]]:
        """扫描目录，获取文件列表和大小信息"""
        supported_extensions = ['.txt', '.md', '.text']
        files_info = []
        
        try:
            directory = Path(directory_path)
            logger.info(f"开始扫描目录: {directory_path}")
            
            for file_path in directory.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                    try:
                        file_size = FileProcessor.get_file_size(str(file_path))
                        files_info.append((str(file_path), file_size))
                    except Exception as e:
                        logger.warning(f"无法获取文件信息 {file_path}: {str(e)}")
            
            # 按文件大小排序，小文件优先处理
            files_info.sort(key=lambda x: x[1])
            
            total_size = sum(size for _, size in files_info)
            logger.info(f"扫描完成: 找到 {len(files_info)} 个文件，总大小 {ProgressManager.format_size(total_size)}")
            
            return files_info
            
        except Exception as e:
            raise Exception(f"扫描目录失败: {str(e)}")
    
    @staticmethod
    def process_directory_streaming(directory_path: str, progress_manager: ProgressManager) -> Generator[Tuple[str, str], None, None]:
        """流式处理目录中的所有文本文件"""
        try:
            # 扫描目录
            files_info = FileProcessor.scan_directory(directory_path)
            
            if not files_info:
                logger.warning(f"目录 {directory_path} 中没有找到支持的文本文件")
                return
            
            total_files = len(files_info)
            total_size = sum(size for _, size in files_info)
            
            # 开始处理
            progress_manager.start_processing(total_files, total_size)
            
            for file_path, file_size in files_info:
                try:
                    logger.info(f"开始处理文件: {file_path}")
                    
                    # 内存检查
                    memory_info = psutil.virtual_memory()
                    if memory_info.percent > 90:
                        logger.warning(f"内存使用率过高 ({memory_info.percent:.1f}%)，执行垃圾回收")
                        gc.collect()
                        time.sleep(0.5)
                    
                    # 读取文件内容
                    content = FileProcessor.read_text_file(file_path)
                    
                    # 更新进度
                    progress_manager.update_file_progress(file_path, file_size)
                    
                    yield (file_path, content)
                    
                    # 释放内存
                    del content
                    gc.collect()
                    
                except Exception as e:
                    logger.warning(f"跳过文件 {file_path}: {str(e)}")
                    progress_manager.update_file_progress(file_path, file_size)
                    continue
                    
        except Exception as e:
            raise Exception(f"流式处理目录失败: {str(e)}")
    
    @staticmethod
    def process_directory(directory_path: str) -> List[Tuple[str, str]]:
        """处理目录中的所有文本文件 - 兼容性方法"""
        progress_manager = ProgressManager()
        files_content = []
        
        try:
            for file_path, content in FileProcessor.process_directory_streaming(directory_path, progress_manager):
                files_content.append((file_path, content))
            
            return files_content
            
        except Exception as e:
            raise Exception(f"处理目录失败: {str(e)}")

# ==================== 文本处理模块 ====================

# 本地关键词提取功能已移除，现在完全依赖AI模型进行关键词提取

def clean_text(text):
    """清理文本"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'^第[一二三四五六七八九十\d]+章.*?\n', '', text, flags=re.MULTILINE)
    text = re.sub(r'^章节.*?\n', '', text, flags=re.MULTILINE)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return '\n'.join(lines)

def smart_split_paragraphs(text, min_length=100, max_length=500, use_smart=True):
    """智能分割段落 - 优化版"""
    if use_smart:
        return advanced_smart_split(text, min_length, max_length)
    else:
        return split_into_paragraphs(text, min_length, max_length)

def advanced_smart_split(text, min_length=100, max_length=500):
    """高级智能分段算法"""
    paragraphs = []
    
    # 1. 首先按自然段落分割
    natural_paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    for para in natural_paragraphs:
        if len(para) <= max_length and len(para) >= min_length:
            paragraphs.append(para)
        elif len(para) > max_length:
            # 对过长段落进行智能分割
            sub_paragraphs = intelligent_split_long_paragraph(para, min_length, max_length)
            paragraphs.extend(sub_paragraphs)
        else:
            # 短段落尝试合并
            if paragraphs and can_merge_paragraphs(paragraphs[-1], para, max_length):
                paragraphs[-1] = merge_paragraphs(paragraphs[-1], para)
            else:
                paragraphs.append(para)
    
    # 2. 后处理：处理过短的段落
    paragraphs = post_process_short_paragraphs(paragraphs, min_length, max_length)
    
    return paragraphs

def intelligent_split_long_paragraph(text, min_length=100, max_length=500):
    """智能分割长段落"""
    # 1. 优先按对话分割
    if has_dialogue(text):
        return split_by_dialogue(text, min_length, max_length)
    
    # 2. 按场景转换分割
    scene_splits = detect_scene_transitions(text)
    if scene_splits:
        return split_by_scenes(text, scene_splits, min_length, max_length)
    
    # 3. 按情感变化分割
    emotion_splits = detect_emotion_changes(text)
    if emotion_splits:
        return split_by_emotions(text, emotion_splits, min_length, max_length)
    
    # 4. 回退到基础分割
    return split_long_paragraph(text, min_length, max_length)

def has_dialogue(text):
    """检测是否包含对话"""
    dialogue_patterns = [
        r'[""''][^""'']*[""'']',  # 引号对话
        r'[：:]\s*[""''][^""'']*[""'']',  # 冒号+引号
        r'说道?[：:]',  # 说道/说:
        r'[问答回]道?[：:]',  # 问道/答道/回道
    ]
    
    for pattern in dialogue_patterns:
        if re.search(pattern, text):
            return True
    return False

def split_by_dialogue(text, min_length=100, max_length=500):
    """按对话分割"""
    # 按对话标记分割
    dialogue_pattern = r'([""''][^""'']*[""'']|[^""'']+)'
    parts = re.findall(dialogue_pattern, text)
    
    result = []
    current = ""
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        if len(current + part) <= max_length:
            current += part
        else:
            if current and len(current) >= min_length:
                result.append(current)
            current = part
    
    if current and len(current) >= min_length:
        result.append(current)
    
    return result if result else [text]

def detect_scene_transitions(text):
    """检测场景转换"""
    scene_markers = [
        r'(突然|忽然|这时|此时|接着|然后|于是|随即)',
        r'(一会儿|片刻|不久|过了|半晌|良久)',
        r'(转身|回头|抬头|低头|起身|坐下)',
        r'(走向|来到|到了|进入|离开|返回)',
        r'(第二天|次日|翌日|黄昏|夜晚|清晨)',
    ]
    
    splits = []
    for pattern in scene_markers:
        for match in re.finditer(pattern, text):
            splits.append(match.start())
    
    return sorted(set(splits))

def split_by_scenes(text, splits, min_length=100, max_length=500):
    """按场景分割"""
    if not splits:
        return [text]
    
    result = []
    start = 0
    
    for split_pos in splits:
        if split_pos - start >= min_length:
            segment = text[start:split_pos].strip()
            if segment:
                result.append(segment)
            start = split_pos
    
    # 添加最后一段
    if start < len(text):
        segment = text[start:].strip()
        if segment:
            if result and len(result[-1] + segment) <= max_length:
                result[-1] += segment
            else:
                result.append(segment)
    
    return result if result else [text]

def detect_emotion_changes(text):
    """检测情感变化"""
    emotion_markers = [
        r'(愤怒|生气|恼火|暴怒)',
        r'(高兴|开心|喜悦|兴奋)',
        r'(悲伤|难过|伤心|痛苦)',
        r'(惊讶|震惊|吃惊|诧异)',
        r'(恐惧|害怕|担心|紧张)',
        r'(平静|冷静|淡然|安详)',
    ]
    
    splits = []
    for pattern in emotion_markers:
        for match in re.finditer(pattern, text):
            splits.append(match.start())
    
    return sorted(set(splits))

def split_by_emotions(text, splits, min_length=100, max_length=500):
    """按情感变化分割"""
    return split_by_scenes(text, splits, min_length, max_length)

def can_merge_paragraphs(para1, para2, max_length):
    """判断两个段落是否可以合并"""
    if len(para1 + '\n' + para2) > max_length:
        return False
    
    # 检查语义连贯性
    # 如果第一段以句号结尾，第二段以大写字母开头，可能是新的主题
    if para1.endswith(('。', '！', '？')) and para2 and para2[0].isupper():
        return False
    
    return True

def merge_paragraphs(para1, para2):
    """合并两个段落"""
    return para1 + '\n' + para2

def post_process_short_paragraphs(paragraphs, min_length, max_length):
    """后处理过短的段落"""
    if not paragraphs:
        return paragraphs
    
    result = []
    i = 0
    
    while i < len(paragraphs):
        current = paragraphs[i]
        
        # 如果当前段落太短，尝试与下一段合并
        if len(current) < min_length and i + 1 < len(paragraphs):
            next_para = paragraphs[i + 1]
            if len(current + '\n' + next_para) <= max_length:
                result.append(current + '\n' + next_para)
                i += 2  # 跳过下一段
                continue
        
        result.append(current)
        i += 1
    
    return result

def split_long_paragraph(text, min_length=100, max_length=500):
    """分割过长段落"""
    if '"' in text or '"' in text or '"' in text:
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
    """按句子分割段落"""
    sentences = re.split(r'[。！？]', text)
    paragraphs = []
    current_paragraph = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        if len(current_paragraph + sentence) > max_length and len(current_paragraph) >= min_length:
            if current_paragraph:
                paragraphs.append(current_paragraph)
            current_paragraph = sentence
        else:
            if current_paragraph:
                current_paragraph += sentence + "。"
            else:
                current_paragraph = sentence + "。"
    
    if current_paragraph and len(current_paragraph) >= min_length:
        paragraphs.append(current_paragraph)
    
    return paragraphs

# ==================== 全局变量 ====================

current_model_client = None
task_manager = PromptTaskManager()
current_paragraph_data = []
progress_manager = ProgressManager()

# ==================== 界面函数 ====================

def load_model(selected_model: str, custom_name: str, custom_base_url: str, 
               custom_api_key: str, custom_model_name: str, save_custom: bool = False) -> Tuple[str, str]:
    """加载AI模型"""
    global current_model_client
    
    try:
        if selected_model and selected_model != "新建模型":
            # 加载已保存的模型配置
            all_configs = AIModelClient.get_all_configs()
            if selected_model not in all_configs:
                return f"错误：未找到模型配置 {selected_model}", ""
            
            config_dict = all_configs[selected_model].copy()
            config = ModelConfig(**config_dict)
        else:
            # 创建新的模型配置
            if not all([custom_name, custom_base_url, custom_api_key, custom_model_name]):
                return "错误：模型配置信息不完整", ""
            
            config = ModelConfig(
                name=custom_name,
                base_url=custom_base_url,
                api_key=custom_api_key,
                model_name=custom_model_name
            )
            
            # 保存新的模型配置
            if save_custom and custom_name:
                config_dict = {
                    "name": custom_name,
                    "base_url": custom_base_url,
                    "api_key": custom_api_key,
                    "model_name": custom_model_name
                }
                AIModelClient.save_custom_config(custom_name, config_dict)
        
        current_model_client = AIModelClient(config)
        
        logger.info(f"正在测试模型连接: {config.name}")
        if current_model_client.test_connection():
            message = f"✅ 模型 {config.name} 加载成功并连接正常"
            logger.info(message)
            updated_choices = list(AIModelClient.get_all_configs().keys()) + ["新建模型"]
            return message, gr.Dropdown(choices=updated_choices)
        else:
            message = f"⚠️ 模型 {config.name} 加载成功但连接测试失败，请检查配置"
            logger.warning(message)
            updated_choices = list(AIModelClient.get_all_configs().keys()) + ["新建模型"]
            return message, gr.Dropdown(choices=updated_choices)
            
    except Exception as e:
        error_msg = f"❌ 模型加载失败: {str(e)}"
        logger.error(error_msg)
        return error_msg, ""

def delete_model_config(model_name: str) -> Tuple[str, gr.Dropdown]:
    """删除模型配置"""
    if not model_name or model_name == "新建模型":
        return "请选择要删除的模型配置", gr.Dropdown()
    
    if AIModelClient.delete_config(model_name):
        updated_choices = list(AIModelClient.get_all_configs().keys()) + ["新建模型"]
        return f"模型配置 '{model_name}' 已删除", gr.Dropdown(choices=updated_choices)
    else:
        return f"删除模型配置 '{model_name}' 失败", gr.Dropdown()

def get_task_prompt(task_name: str) -> str:
    """获取任务提示词"""
    return task_manager.get_task_prompt(task_name)

def add_custom_task(task_name: str, task_prompt: str) -> Tuple[str, gr.Dropdown, gr.Dropdown]:
    """添加自定义任务"""
    if not task_name or not task_prompt:
        task_choices = task_manager.get_task_names()
        return "任务名称和提示词不能为空", gr.Dropdown(choices=task_choices), gr.Dropdown(choices=task_choices)
    
    task_manager.add_task(task_name, task_prompt)
    updated_choices = task_manager.get_task_names()
    return f"任务 '{task_name}' 已保存", gr.Dropdown(choices=updated_choices, value=task_name), gr.Dropdown(choices=updated_choices, value=task_name)

def edit_task(task_name: str, task_prompt: str) -> Tuple[str, gr.Dropdown, gr.Dropdown]:
    """编辑任务"""
    if not task_name:
        task_choices = task_manager.get_task_names()
        return "请选择要编辑的任务", gr.Dropdown(choices=task_choices), gr.Dropdown(choices=task_choices)
    
    if not task_prompt.strip():
        task_choices = task_manager.get_task_names()
        return "提示词不能为空", gr.Dropdown(choices=task_choices), gr.Dropdown(choices=task_choices)
    
    task_manager.add_task(task_name, task_prompt.strip())
    updated_choices = task_manager.get_task_names()
    return f"任务 '{task_name}' 已更新", gr.Dropdown(choices=updated_choices, value=task_name), gr.Dropdown(choices=updated_choices, value=task_name)

def delete_task(task_name: str) -> Tuple[str, gr.Dropdown, gr.Dropdown]:
    """删除任务"""
    if not task_name:
        task_choices = task_manager.get_task_names()
        return "请选择要删除的任务", gr.Dropdown(choices=task_choices), gr.Dropdown(choices=task_choices)
    
    if task_manager.delete_task(task_name):
        updated_choices = task_manager.get_task_names()
        return f"任务 '{task_name}' 已删除", gr.Dropdown(choices=updated_choices), gr.Dropdown(choices=updated_choices)
    else:
        task_choices = task_manager.get_task_names()
        return f"删除任务 '{task_name}' 失败", gr.Dropdown(choices=task_choices), gr.Dropdown(choices=task_choices)

def process_files_streaming(file_input, directory_input, min_length, max_length, use_smart_split, 
                           use_ai_keywords, prompt_task, custom_instruction, output_dir) -> Generator[str, None, Tuple[str, str, List]]:
    """流式处理文件或目录 - 大数据优化版本，支持增量保存"""
    global current_paragraph_data, progress_manager
    
    try:
        # 初始化
        all_paragraph_data = []
        training_data = []
        batch_size = 500  # 减小批处理大小，更频繁保存
        current_batch = []
        
        # 增量保存配置
        incremental_save_interval = 1000  # 每1000条数据保存一次
        temp_save_counter = 0
        temp_files = []  # 临时文件列表
        
        yield "🔍 开始扫描文件..."
        
        # 创建临时保存目录
        temp_dir = None
        if output_dir:
            temp_dir = os.path.join(output_dir, f"temp_{int(time.time())}")
            os.makedirs(temp_dir, exist_ok=True)
            yield f"📁 创建临时保存目录: {temp_dir}"
        
        # 获取文件流
        if file_input is not None:
            # 处理单个文件
            if hasattr(file_input, 'name'):
                file_path = file_input.name
            else:
                file_path = str(file_input)
            
            file_size = FileProcessor.get_file_size(file_path)
            progress_manager.start_processing(1, file_size)
            
            yield f"📁 处理单个文件: {os.path.basename(file_path)} ({ProgressManager.format_size(file_size)})"
            
            content = FileProcessor.read_text_file(file_path)
            file_stream = [(file_path, content)]
            progress_manager.update_file_progress(file_path, file_size)
            
        elif directory_input:
            # 处理目录 - 使用流式处理
            yield f"📂 扫描目录: {directory_input}"
            file_stream = FileProcessor.process_directory_streaming(directory_input, progress_manager)
            
        else:
            yield "❌ 请选择文件或输入目录路径"
            return "请选择文件或输入目录路径", "", []
        
        yield "📝 开始处理文本内容..."
        
        # 流式处理每个文件
        total_paragraphs = 0
        processed_files = 0
        
        for file_path, content in file_stream:
            try:
                processed_files += 1
                yield f"\n📄 处理文件 [{processed_files}]: {os.path.basename(file_path)}"
                
                # 内存使用检查
                progress_manager.log_memory_usage(f"处理文件 {os.path.basename(file_path)} 前")
                
                # 清理文本
                yield "  🧹 清理文本..."
                cleaned_text = clean_text(content)
                
                # 分割段落
                yield "  ✂️ 分割段落..."
                if use_smart_split:
                    paragraphs = smart_split_paragraphs(cleaned_text, min_length, max_length, True)
                else:
                    paragraphs = split_into_paragraphs(cleaned_text, min_length, max_length)
                
                if len(paragraphs) < 2:
                    logger.warning(f"文件 {file_path} 段落太少，跳过")
                    yield f"  ⚠️ 段落太少 ({len(paragraphs)})，跳过"
                    continue
                
                yield f"  📊 找到 {len(paragraphs)} 个段落，开始生成训练数据..."
                
                # 批量处理段落
                file_training_data = []
                for i in range(len(paragraphs) - 1):
                    # 更新段落进度
                    if i % 50 == 0 or i == len(paragraphs) - 2:
                        progress_manager.update_paragraph_progress(i + 1, len(paragraphs) - 1, file_path)
                    
                    input_text = paragraphs[i]
                    response_text = paragraphs[i + 1]
                    
                    # 提取关键词 - 仅使用AI模型
                    if use_ai_keywords:
                        if not current_model_client:
                            raise Exception("启用AI关键词提取但未配置AI模型，请先在AI设置中配置并加载模型")
                        if not prompt_task:
                            raise Exception("启用AI关键词提取但未选择提示词任务，请选择一个提示词任务")
                        
                        try:
                            prompt = task_manager.get_task_prompt(prompt_task)
                            keywords = current_model_client.extract_keywords(response_text, prompt)
                        except Exception as e:
                            logger.error(f"AI关键词提取失败: {str(e)}")
                            raise Exception(f"AI关键词提取失败: {str(e)}")
                    else:
                        # 不使用AI关键词提取时，使用默认关键词
                        keywords = "续写小说"
                    
                    # 生成指令
                    if custom_instruction:
                        instruction = custom_instruction.replace('{keywords}', keywords)
                    else:
                        instruction = f"请根据关键词'{keywords}'续写小说段落"
                    
                    data_item = {
                        "text": f"Instruction: {instruction}\n\nInput: {input_text}\n\nResponse: {response_text}"
                    }
                    file_training_data.append(data_item)
                    
                    # 保存段落数据
                    paragraph_item = {
                        "index": len(all_paragraph_data),
                        "file_path": file_path,
                        "input": input_text,
                        "response": response_text,
                        "keywords": keywords,
                        "instruction": instruction
                    }
                    all_paragraph_data.append(paragraph_item)
                    
                    # 批量处理和增量保存
                    current_batch.extend(file_training_data[-1:])
                    if len(current_batch) >= batch_size:
                        training_data.extend(current_batch)
                        current_batch = []
                        
                        # 增量保存检查
                        if temp_dir and len(training_data) - temp_save_counter >= incremental_save_interval:
                            temp_save_counter = len(training_data)
                            temp_file = os.path.join(temp_dir, f"batch_{len(temp_files)+1:04d}.jsonl")
                            
                            # 保存当前批次数据
                            with open(temp_file, 'w', encoding='utf-8') as f:
                                start_idx = max(0, len(training_data) - incremental_save_interval)
                                batch_data = training_data[start_idx:]
                                for item in batch_data:
                                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                            
                            temp_files.append(temp_file)
                            yield f"  💾 增量保存: {len(training_data)} 条数据已保存到临时文件 {os.path.basename(temp_file)}"
                        
                        gc.collect()  # 释放内存
                        yield f"  📊 已处理 {len(training_data)} 条训练数据 (内存: {psutil.virtual_memory().percent:.1f}%)"
                
                total_paragraphs += len(paragraphs) - 1
                yield f"  ✅ 完成文件处理，生成 {len(file_training_data)} 条训练数据"
                
                # 释放文件内容内存
                del content, cleaned_text, paragraphs, file_training_data
                gc.collect()
                
            except Exception as e:
                logger.error(f"处理文件 {file_path} 失败: {str(e)}")
                yield f"  ❌ 处理失败: {str(e)}"
                continue
        
        # 处理剩余批次
        if current_batch:
            training_data.extend(current_batch)
            yield f"💾 处理最后批次，总计 {len(training_data)} 条训练数据"
        
        # 最后一次增量保存
        if temp_dir and training_data and len(training_data) > temp_save_counter:
            temp_file = os.path.join(temp_dir, f"batch_{len(temp_files)+1:04d}.jsonl")
            with open(temp_file, 'w', encoding='utf-8') as f:
                remaining_data = training_data[temp_save_counter:]
                for item in remaining_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            temp_files.append(temp_file)
            yield f"💾 保存剩余 {len(training_data) - temp_save_counter} 条数据到 {os.path.basename(temp_file)}"
        
        current_paragraph_data = all_paragraph_data
        
        # 完成处理
        progress_manager.finish_processing(total_paragraphs, len(training_data))
        
        # 合并临时文件并生成最终JSONL
        yield "📄 合并临时文件并生成最终JSONL..."
        
        if output_dir and temp_files:
            # 合并所有临时文件
            output_file = os.path.join(output_dir, f"novel_dataset_{int(time.time())}.jsonl")
            yield f"💾 合并 {len(temp_files)} 个临时文件到最终文件..."
            
            with open(output_file, 'w', encoding='utf-8') as final_file:
                total_written = 0
                for i, temp_file in enumerate(temp_files):
                    yield f"  📄 合并文件 {i+1}/{len(temp_files)}: {os.path.basename(temp_file)}"
                    with open(temp_file, 'r', encoding='utf-8') as tf:
                        for line in tf:
                            final_file.write(line)
                            total_written += 1
                            if total_written % 5000 == 0:
                                yield f"    💾 已写入 {total_written} 条数据"
            
            # 清理临时文件
            yield "🧹 清理临时文件..."
            for temp_file in temp_files:
                try:
                    os.remove(temp_file)
                except Exception as e:
                    logger.warning(f"删除临时文件失败 {temp_file}: {str(e)}")
            
            try:
                os.rmdir(temp_dir)
            except Exception as e:
                logger.warning(f"删除临时目录失败 {temp_dir}: {str(e)}")
            
            # 生成内存中的JSONL用于显示（限制大小）
            if len(training_data) <= 1000:
                jsonl_output = "\n".join([json.dumps(item, ensure_ascii=False) for item in training_data])
            else:
                # 只显示前1000条
                sample_data = training_data[:1000]
                jsonl_output = "\n".join([json.dumps(item, ensure_ascii=False) for item in sample_data])
                jsonl_output += f"\n\n# 注意: 由于数据量过大，此处仅显示前1000条数据\n# 完整数据已保存到文件: {output_file}"
            
            status_message = f"✅ 处理完成！\n📊 处理了 {processed_files} 个文件\n📝 生成了 {len(training_data)} 条训练数据\n💾 已保存到: {output_file}\n🔄 使用了 {len(temp_files)} 个临时文件进行增量保存"
        else:
            # 无输出目录时，生成内存JSONL
            if len(training_data) <= 1000:
                jsonl_output = "\n".join([json.dumps(item, ensure_ascii=False) for item in training_data])
            else:
                sample_data = training_data[:1000]
                jsonl_output = "\n".join([json.dumps(item, ensure_ascii=False) for item in sample_data])
                jsonl_output += f"\n\n# 注意: 由于数据量过大，此处仅显示前1000条数据"
            
            status_message = f"✅ 处理完成！\n📊 处理了 {processed_files} 个文件\n📝 生成了 {len(training_data)} 条训练数据\n⚠️ 未指定输出目录，数据仅保存在内存中"
        
        yield f"\n{status_message}"
        return status_message, jsonl_output, all_paragraph_data
        
    except Exception as e:
        error_msg = f"❌ 处理失败: {str(e)}"
        logger.error(error_msg)
        yield error_msg
        return error_msg, "", []

def process_files_with_progress(file_input, directory_input, min_length, max_length, use_smart_split, 
                               use_ai_keywords, prompt_task, custom_instruction, output_dir):
    """带实时进度显示的文件处理函数 - 支持流式输出"""
    global progress_manager
    
    try:
        # 重置进度管理器
        progress_manager = ProgressManager()
        
        # 实时进度状态
        status_messages = []
        final_result = None
        current_progress = "等待开始处理..."
        current_file_progress = "0/0 文件"
        current_memory = "0 MB"
        last_update_time = time.time()
        
        # 流式处理生成器
        stream_generator = process_files_streaming(file_input, directory_input, min_length, max_length, 
                                                 use_smart_split, use_ai_keywords, prompt_task, 
                                                 custom_instruction, output_dir)
        
        for message in stream_generator:
            if isinstance(message, tuple):
                # 最终结果
                final_result = message
                break
            else:
                # 进度消息
                status_messages.append(message)
                print(message)  # 实时输出到终端
                
                # 更新进度信息
                current_time = time.time()
                if "扫描文件" in message or "扫描目录" in message:
                    current_progress = "🔍 扫描文件中..."
                elif "创建临时保存目录" in message:
                    current_progress = "📁 准备增量保存..."
                elif "处理文件" in message and "[" in message:
                    # 提取文件进度
                    if "处理文件 [" in message:
                        try:
                            file_num = message.split("[")[1].split("]")[0]
                            current_file_progress = f"{file_num} 文件处理中"
                        except:
                            pass
                    current_progress = "📄 处理文件中..."
                elif "清理文本" in message:
                    current_progress = "🧹 清理文本中..."
                elif "分割段落" in message:
                    current_progress = "✂️ 分割段落中..."
                elif "生成训练数据" in message:
                    current_progress = "📝 生成训练数据中..."
                elif "增量保存" in message:
                    current_progress = "💾 增量保存中..."
                elif "合并临时文件" in message:
                    current_progress = "📄 合并文件中..."
                elif "清理临时文件" in message:
                    current_progress = "🧹 清理临时文件中..."
                elif "处理完成" in message:
                    current_progress = "✅ 处理完成！"
                
                # 获取内存使用情况 (限制更新频率)
                if current_time - last_update_time >= 1.0:  # 每秒更新一次
                    try:
                        process = psutil.Process()
                        memory_mb = process.memory_info().rss / 1024 / 1024
                        system_memory = psutil.virtual_memory()
                        current_memory = f"进程: {memory_mb:.1f}MB | 系统: {system_memory.percent:.1f}%"
                        last_update_time = current_time
                    except:
                        pass
                
                # 实时返回当前状态 (每条消息都返回)
                full_log = "\n".join(status_messages[-50:])  # 只保留最近50条消息
                yield "", [], current_progress, current_file_progress, current_memory, full_log
        
        # 处理最终结果
        full_log = "\n".join(status_messages)
        
        if final_result:
            status, jsonl, paragraph_data = final_result
            current_progress = "✅ 处理完成！"
            yield jsonl, paragraph_data, current_progress, current_file_progress, current_memory, full_log
        else:
            yield "", [], current_progress, current_file_progress, current_memory, full_log
            
    except Exception as e:
        error_msg = f"❌ 处理失败: {str(e)}"
        logger.error(error_msg)
        yield "", [], "❌ 处理失败", "0/0 文件", "0 MB", error_msg
def process_files(file_input, directory_input, min_length, max_length, use_smart_split, 
                 use_ai_keywords, prompt_task, custom_instruction, output_dir):
    """处理文件或目录 - 兼容性包装器"""
    try:
        # 使用流式处理
        status_messages = []
        final_result = None
        
        for message in process_files_streaming(file_input, directory_input, min_length, max_length, 
                                             use_smart_split, use_ai_keywords, prompt_task, 
                                             custom_instruction, output_dir):
            if isinstance(message, tuple):
                # 最终结果
                final_result = message
                break
            else:
                # 进度消息
                status_messages.append(message)
                print(message)  # 实时输出到终端
        
        if final_result:
            return final_result
        else:
            return "处理完成", "", []
            
    except Exception as e:
        error_msg = f"❌ 处理失败: {str(e)}"
        logger.error(error_msg)
        return error_msg, "", []

def save_jsonl(jsonl_content, filename, output_dir):
    """保存JSONL文件"""
    if not jsonl_content:
        return "没有内容可保存"
    
    try:
        if not filename:
            filename = f"novel_dataset_{int(time.time())}.jsonl"
        
        if not filename.endswith('.jsonl'):
            filename += '.jsonl'
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, filename)
        else:
            filepath = filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(jsonl_content)
        
        return f"✅ 文件已保存: {filepath}"
    except Exception as e:
        return f"❌ 保存失败: {str(e)}"

def update_paragraph_keywords(paragraph_data, index, new_keywords, new_instruction):
    """更新段落关键词和指令"""
    try:
        if 0 <= index < len(paragraph_data):
            paragraph_data[index]['keywords'] = new_keywords
            paragraph_data[index]['instruction'] = new_instruction
            return paragraph_data, f"✅ 段落 {index+1} 更新成功"
        else:
            return paragraph_data, "❌ 段落索引无效"
    except Exception as e:
        return paragraph_data, f"❌ 更新失败: {str(e)}"

def regenerate_jsonl_from_paragraphs(paragraph_data):
    """从段落数据重新生成JSONL"""
    try:
        training_data = []
        for item in paragraph_data:
            data_item = {
                "text": f"Instruction: {item['instruction']}\n\nInput: {item['input']}\n\nResponse: {item['response']}"
            }
            training_data.append(data_item)
        
        jsonl_output = "\n".join([json.dumps(item, ensure_ascii=False) for item in training_data])
        return jsonl_output, f"✅ 重新生成完成，共 {len(training_data)} 条数据"
    except Exception as e:
        return "", f"❌ 生成失败: {str(e)}"

def convert_jsonl_to_binidx(jsonl_file, output_prefix, tokenizer_type="RWKVTokenizer"):
    """将JSONL文件转换为binidx格式"""
    try:
        current_dir = Path(__file__).parent
        tool_dir = current_dir / "json2binidx_tool"
        if not tool_dir.exists():
            return "错误：找不到json2binidx_tool目录"
        
        preprocess_script = tool_dir / "tools" / "preprocess_data.py"
        if not preprocess_script.exists():
            return "错误：找不到preprocess_data.py脚本"
        
        if tokenizer_type == "RWKVTokenizer":
            vocab_file = tool_dir / "rwkv_vocab_v20230424.txt"
        else:
            vocab_file = tool_dir / "20B_tokenizer.json"
        
        if not vocab_file.exists():
            return f"错误：找不到tokenizer文件: {vocab_file}"
        
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
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(tool_dir))
        
        if result.returncode == 0:
            return f"✅ 转换成功！生成文件: {output_prefix}.bin 和 {output_prefix}.idx"
        else:
            return f"❌ 转换失败: {result.stderr}"
            
    except Exception as e:
        return f"❌ 转换过程中出错: {str(e)}"

# ==================== 界面创建 ====================

def create_interface():
    """创建Gradio界面"""
    
    with gr.Blocks(title="小说训练数据生成器", theme=gr.themes.Ocean()) as demo:
        gr.Markdown("# 📚 小说训练数据生成器")
        gr.Markdown("智能处理小说文本，生成高质量的训练数据集")
        
        # 状态变量
        paragraph_data_state = gr.State([])
        
        with gr.Tabs():
            # AI设置标签页
            with gr.TabItem("🤖 AI设置"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 模型配置")
                        
                        model_preset = gr.Dropdown(
                            choices=list(AIModelClient.get_all_configs().keys()) + ["新建模型"],
                            value="新建模型" if not AIModelClient.get_all_configs() else list(AIModelClient.get_all_configs().keys())[0],
                            label="选择模型"
                        )
                        
                        with gr.Group(visible=True) as custom_model_group:
                            custom_model_name = gr.Textbox(label="模型名称")
                            custom_base_url = gr.Textbox(label="API地址")
                            custom_api_key = gr.Textbox(label="API密钥", type="password")
                            custom_model_id = gr.Textbox(label="模型ID")
                            save_custom_config = gr.Checkbox(label="保存自定义配置")
                        
                        with gr.Row():
                            load_model_btn = gr.Button("🔄 加载模型", variant="primary")
                            delete_model_btn = gr.Button("🗑️ 删除模型", variant="stop")
                        model_status = gr.Textbox(label="模型状态", interactive=False)
                        
                        def toggle_custom_model(preset):
                            return gr.Group(visible=(preset == "新建模型"))
                        
                        model_preset.change(
                            fn=toggle_custom_model,
                            inputs=[model_preset],
                            outputs=[custom_model_group]
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 提示词管理")
                        
                        task_dropdown = gr.Dropdown(
                            choices=task_manager.get_task_names(),
                            value="小说关键词提取",
                            label="选择提示词任务"
                        )
                        
                        current_prompt = gr.Textbox(
                            label="当前提示词",
                            value=task_manager.get_task_prompt("小说关键词提取"),
                            lines=3,
                            interactive=True
                        )
                        
                        with gr.Row():
                            new_task_name = gr.Textbox(label="新任务名称", scale=2)
                            add_task_btn = gr.Button("➕ 添加", scale=1)
                        
                        new_task_prompt = gr.Textbox(label="新任务提示词", lines=3)
                        
                        with gr.Row():
                            edit_task_btn = gr.Button("✏️ 编辑当前任务")
                            delete_task_btn = gr.Button("🗑️ 删除任务")
                        
                        task_status = gr.Textbox(label="操作状态", interactive=False)
            
            # 文件处理标签页
            with gr.TabItem("📁 文件处理"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 输入设置")
                        
                        file_input = gr.File(label="上传单个文件", file_types=[".txt", ".md"])
                        directory_input = gr.Textbox(label="或输入目录路径")
                        
                        gr.Markdown("### 处理参数")
                        
                        with gr.Row():
                            min_length = gr.Slider(50, 300, value=100, label="最小段落长度")
                            max_length = gr.Slider(300, 1000, value=500, label="最大段落长度")
                        
                        use_smart_split = gr.Checkbox(label="启用智能分段", value=True)
                        use_ai_keywords = gr.Checkbox(label="使用AI提取关键词", value=True)
                        
                        prompt_task_select = gr.Dropdown(
                            choices=task_manager.get_task_names(),
                            value="小说关键词提取",
                            label="关键词提取任务"
                        )
                        
                        custom_instruction = gr.Textbox(
                            label="自定义指令模板（使用{keywords}作为占位符）",
                            placeholder="请根据关键词'{keywords}'续写小说段落"
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 输出设置")
                        
                        output_dir = gr.Textbox(
                            label="保存目录",
                            value="./output",
                            placeholder="输入保存目录路径"
                        )
                        
                        process_btn = gr.Button("🚀 开始处理", variant="primary", size="lg")
                        
                        # 进度显示区域
                        gr.Markdown("### 处理进度")
                        
                        with gr.Group():
                            progress_info = gr.Textbox(
                                label="当前状态",
                                value="等待开始处理...",
                                interactive=False
                            )
                            
                            file_progress = gr.Textbox(
                                label="文件进度",
                                value="0/0 文件",
                                interactive=False
                            )
                            
                            memory_info = gr.Textbox(
                                label="内存使用",
                                value="0 MB",
                                interactive=False
                            )
                        
                        processing_status = gr.Textbox(
                            label="详细日志",
                            lines=8,
                            interactive=False,
                            max_lines=15
                        )
                        
                        jsonl_output = gr.Textbox(
                            label="生成的JSONL数据",
                            lines=6,
                            max_lines=15
                        )
            
            # 段落编辑标签页
            with gr.TabItem("✏️ 段落编辑"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 段落列表")
                        
                        paragraph_count = gr.Textbox(label="段落总数", interactive=False)
                        
                        paragraph_selector = gr.Dropdown(
                            label="选择段落",
                            choices=[],
                            interactive=True
                        )
                        
                        with gr.Row():
                            prev_btn = gr.Button("⬅️ 上一个")
                            next_btn = gr.Button("➡️ 下一个")
                        
                        file_source = gr.Textbox(label="文件来源", interactive=False)
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### 段落内容")
                        
                        input_preview = gr.Textbox(
                            label="输入段落",
                            lines=5,
                            interactive=False
                        )
                        
                        response_preview = gr.Textbox(
                            label="响应段落",
                            lines=5,
                            interactive=False
                        )
                        
                        edit_keywords = gr.Textbox(label="关键词")
                        edit_instruction = gr.Textbox(label="指令")
                        
                        with gr.Row():
                            update_btn = gr.Button("💾 更新段落", variant="primary")
                            regenerate_btn = gr.Button("🔄 重新生成JSONL")
                        
                        edit_status = gr.Textbox(label="编辑状态", interactive=False)
            
            # 数据导出标签页
            with gr.TabItem("💾 数据导出"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### JSONL保存")
                        
                        save_filename = gr.Textbox(
                            label="文件名",
                            placeholder="留空自动生成"
                        )
                        
                        save_output_dir = gr.Textbox(
                            label="保存目录",
                            value="./output"
                        )
                        
                        save_jsonl_btn = gr.Button("💾 保存JSONL", variant="primary")
                        save_status = gr.Textbox(label="保存状态", interactive=False)
                    
                    with gr.Column():
                        gr.Markdown("### BinIdx转换")
                        
                        convert_jsonl_file = gr.File(label="选择JSONL文件", file_types=[".jsonl"])
                        
                        convert_output_prefix = gr.Textbox(
                            label="输出前缀",
                            value="novel_dataset"
                        )
                        
                        tokenizer_type = gr.Dropdown(
                            choices=["RWKVTokenizer", "GPTNeoXTokenizer"],
                            value="RWKVTokenizer",
                            label="Tokenizer类型"
                        )
                        
                        convert_btn = gr.Button("🔄 转换为BinIdx")
                        convert_status = gr.Textbox(label="转换状态", interactive=False)
        
        # 事件绑定
        
        # AI设置事件
        load_model_btn.click(
            fn=load_model,
            inputs=[model_preset, custom_model_name, custom_base_url, custom_api_key, custom_model_id, save_custom_config],
            outputs=[model_status, model_preset]
        )
        
        task_dropdown.change(
            fn=get_task_prompt,
            inputs=[task_dropdown],
            outputs=[current_prompt]
        )
        
        add_task_btn.click(
            fn=add_custom_task,
            inputs=[new_task_name, new_task_prompt],
            outputs=[task_status, task_dropdown, prompt_task_select]
        )
        
        edit_task_btn.click(
            fn=edit_task,
            inputs=[task_dropdown, current_prompt],
            outputs=[task_status, task_dropdown, prompt_task_select]
        )
        
        delete_task_btn.click(
            fn=delete_task,
            inputs=[task_dropdown],
            outputs=[task_status, task_dropdown, prompt_task_select]
        )
        
        delete_model_btn.click(
            fn=delete_model_config,
            inputs=[model_preset],
            outputs=[model_status, model_preset]
        )
        
        # 文件处理事件 - 支持流式输出
        process_btn.click(
            fn=process_files_with_progress,
            inputs=[file_input, directory_input, min_length, max_length, use_smart_split, 
                   use_ai_keywords, prompt_task_select, custom_instruction, output_dir],
            outputs=[jsonl_output, paragraph_data_state, 
                    progress_info, file_progress, memory_info, processing_status],
            show_progress=True,
            queue=True
        )
        
        # 段落编辑事件
        def update_paragraph_selector(paragraph_data):
            if paragraph_data:
                choices = [f"段落 {i+1}: {item['file_path']}" for i, item in enumerate(paragraph_data)]
                return gr.Dropdown(choices=choices), f"共 {len(paragraph_data)} 个段落"
            return gr.Dropdown(choices=[]), "0 个段落"
        
        def load_paragraph_content(paragraph_data, selected):
            if not paragraph_data or not selected:
                return "", "", "", "", ""
            
            try:
                index = int(selected.split(":")[0].replace("段落 ", "")) - 1
                if 0 <= index < len(paragraph_data):
                    item = paragraph_data[index]
                    return (
                        item['input'],
                        item['response'],
                        item['keywords'],
                        item['instruction'],
                        item['file_path']
                    )
            except:
                pass
            
            return "", "", "", "", ""
        
        paragraph_data_state.change(
            fn=update_paragraph_selector,
            inputs=[paragraph_data_state],
            outputs=[paragraph_selector, paragraph_count]
        )
        
        paragraph_selector.change(
            fn=load_paragraph_content,
            inputs=[paragraph_data_state, paragraph_selector],
            outputs=[input_preview, response_preview, edit_keywords, edit_instruction, file_source]
        )
        
        def update_paragraph(paragraph_data, selected, keywords, instruction):
            if not paragraph_data or not selected:
                return paragraph_data, "请选择段落"
            
            try:
                index = int(selected.split(":")[0].replace("段落 ", "")) - 1
                return update_paragraph_keywords(paragraph_data, index, keywords, instruction)
            except:
                return paragraph_data, "更新失败"
        
        update_btn.click(
            fn=update_paragraph,
            inputs=[paragraph_data_state, paragraph_selector, edit_keywords, edit_instruction],
            outputs=[paragraph_data_state, edit_status]
        )
        
        regenerate_btn.click(
            fn=regenerate_jsonl_from_paragraphs,
            inputs=[paragraph_data_state],
            outputs=[jsonl_output, edit_status]
        )
        
        # 数据导出事件
        save_jsonl_btn.click(
            fn=save_jsonl,
            inputs=[jsonl_output, save_filename, save_output_dir],
            outputs=[save_status]
        )
        
        def convert_to_binidx_wrapper(jsonl_file, prefix, tokenizer):
            if jsonl_file is None:
                return "请选择JSONL文件"
            return convert_jsonl_to_binidx(jsonl_file.name, prefix, tokenizer)
        
        convert_btn.click(
            fn=convert_to_binidx_wrapper,
            inputs=[convert_jsonl_file, convert_output_prefix, tokenizer_type],
            outputs=[convert_status]
        )
        
        # 添加使用说明
        with gr.Accordion("📖 使用说明", open=False):
            gr.Markdown("""
            ## 🚀 功能特点
            
            ### 🤖 AI设置
            1. **多模型支持**: 支持DeepSeek、OpenAI、Claude等多种AI模型
            2. **提示词管理**: 内置多种小说类型的关键词提取模板
            3. **自定义配置**: 可保存和管理自定义模型配置
            
            ### 📁 文件处理
            1. **批量处理**: 支持单文件或整个目录的批量处理
            2. **智能分段**: 考虑对话、场景转换等进行智能段落分割
            3. **AI关键词**: 使用AI模型提取更准确的关键词
            4. **自定义指令**: 支持自定义指令模板
            
            ### ✏️ 段落编辑
            1. **可视化编辑**: 查看和编辑每个段落的关键词和指令
            2. **实时预览**: 实时查看段落内容和当前指令
            3. **批量更新**: 支持重新生成整个JSONL文件
            
            ### 💾 数据导出
            1. **JSONL保存**: 保存为标准JSONL格式
            2. **BinIdx转换**: 转换为RWKV训练格式
            3. **多种Tokenizer**: 支持不同的分词器
            
            ## 📝 数据格式
            
            生成的训练数据格式：
            ```
            {
              "text": "Instruction: 请根据关键词'武功、内力、修炼'续写小说段落\n\nInput: 前一段小说内容...\n\nResponse: 后一段小说内容..."
            }
            ```
            
            ## 🔧 使用步骤
            
            1. **AI设置**: 配置AI模型和提示词模板
            2. **文件处理**: 上传文件或指定目录，设置处理参数
            3. **段落编辑**: 在编辑页面优化关键词和指令
            4. **数据导出**: 保存JSONL文件或转换为训练格式
            """)
    
    return demo

if __name__ == "__main__":
    try:
        import jieba
    except ImportError:
        print("请先安装jieba: pip install jieba")
        exit(1)
    
    demo = create_interface()
    demo.launch(share=False, server_name="127.0.0.1", server_port=None, show_error=True)
