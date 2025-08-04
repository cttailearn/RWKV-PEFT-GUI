import os, subprocess, tempfile, shutil
import gradio as gr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GGUF_SCRIPT = os.path.join(BASE_DIR, "converter", "convert_rwkv_pth_to_gguf.py")
ST_SCRIPT = os.path.join(BASE_DIR, "converter", "convert_rwkv_to_safetensors.py")
DEFAULT_VOCAB_PATH = os.path.join(BASE_DIR, "converter", "rwkv_vocab_v20230424.txt")


def run_subprocess(cmd: list[str]):
    """Run a subprocess command and return (stdout + stderr)."""
    try:
        print(f"[执行命令] {' '.join(cmd)}")
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = res.stdout + "\n" + res.stderr
        print(f"[命令输出] {output}")
        return output
    except subprocess.CalledProcessError as e:
        error_msg = f"Error while executing {' '.join(cmd)}:\n{e.stdout}\n{e.stderr}"
        print(f"[命令错误] {error_msg}")
        return error_msg


def convert_model(pth_file, vocab_file, target_format, outtype, output_dir, upload_mode=None, folder_path=None):
    """Main conversion dispatcher."""
    
    # 处理不同的输入模式
    if upload_mode == "文件夹":
        if not folder_path or not os.path.exists(folder_path):
            return "请输入有效的文件夹路径。", [], "", ""
        
        # 查找文件夹中的PTH文件
        pth_files = [f for f in os.listdir(folder_path) if f.endswith('.pth')]
        if not pth_files:
            return "指定文件夹中未找到PTH文件。", [], folder_path, ""
        
        # 使用第一个找到的PTH文件
        pth_file_path = os.path.join(folder_path, pth_files[0])
        input_dir = folder_path
        print(f"[文件夹模式] 找到PTH文件: {pth_files}")
        print(f"[使用文件] {pth_file_path}")
        
        # 创建临时文件对象
        class TempFile:
            def __init__(self, path):
                self.name = path
        pth_file = TempFile(pth_file_path)
    else:
        if pth_file is None:
            return "请上传 PTH 模型文件。", [], "", ""
        input_dir = os.path.dirname(pth_file.name)
    
    # 确定输出目录 - 默认为代码文件所在目录的output/models
    if not output_dir or not os.path.exists(output_dir):
        default_output_dir = os.path.join(BASE_DIR, "output", "models")
        os.makedirs(default_output_dir, exist_ok=True)
        final_output_dir = default_output_dir
    else:
        final_output_dir = output_dir
    
    # 处理词表文件
    if target_format in ("GGUF", "Both") and vocab_file is None:
        if os.path.exists(DEFAULT_VOCAB_PATH):
            vocab_file = open(DEFAULT_VOCAB_PATH, "rb")
            print(f"[使用默认词表] {DEFAULT_VOCAB_PATH}")
        else:
            return "GGUF 转换需要词表文件，请上传或确保默认词表存在。", [], input_dir, final_output_dir

    logs = []
    outputs = []
    print(f"[开始转换] 输入文件: {pth_file.name}")
    print(f"[输出目录] {final_output_dir}")
    
    # 创建临时工作目录，拷贝文件进去，避免中文路径问题
    with tempfile.TemporaryDirectory() as tmpdir:
        pth_path = os.path.join(tmpdir, os.path.basename(pth_file.name))
        shutil.copy(pth_file.name, pth_path)

        vocab_path = None
        if vocab_file is not None:
            vocab_path = os.path.join(tmpdir, os.path.basename(vocab_file.name))
            shutil.copy(vocab_file.name, vocab_path)

        # GGUF
        if target_format in ("GGUF", "Both"):
            gguf_out = pth_path.replace(".pth", f"_{outtype}.gguf")
            cmd = [
                "python", GGUF_SCRIPT,
                "--outfile", gguf_out,
                "--outtype", outtype,
                pth_path, vocab_path,
            ]
            logs.append("运行 GGUF 转换:\n" + " ".join(cmd))
            logs.append(run_subprocess(cmd))
            if os.path.exists(gguf_out):
                outputs.append(gguf_out)

        # SafeTensors
        if target_format in ("SafeTensors", "Both"):
            st_out = pth_path.replace(".pth", ".safetensors")
            cmd = [
                "python", ST_SCRIPT,
                "--input", pth_path,
                "--output", st_out,
            ]
            logs.append("运行 SafeTensors 转换:\n" + " ".join(cmd))
            logs.append(run_subprocess(cmd))
            if os.path.exists(st_out):
                outputs.append(st_out)

        # 移动输出文件到指定目录
        downloadable = []
        for f in outputs:
            dest = os.path.join(final_output_dir, os.path.basename(f))
            shutil.move(f, dest)
            downloadable.append(dest)
            print(f"[输出文件] {dest}")

    return "\n\n".join(logs), downloadable, input_dir, final_output_dir


def generate_modelfile(gguf_file, mode, output_dir, upload_mode=None, folder_path=None):
    
    # 处理不同的输入模式
    if upload_mode == "文件夹":
        if not folder_path or not os.path.exists(folder_path):
            return "请输入有效的文件夹路径。", [], "", ""
        
        # 查找文件夹中的GGUF文件
        gguf_files = [f for f in os.listdir(folder_path) if f.endswith('.gguf')]
        if not gguf_files:
            return "指定文件夹中未找到GGUF文件。", [], folder_path, ""
        
        # 使用第一个找到的GGUF文件
        gguf_path = os.path.join(folder_path, gguf_files[0])
        input_dir = folder_path
        print(f"[文件夹模式] 找到GGUF文件: {gguf_files}")
        print(f"[使用文件] {gguf_path}")
    else:
        if gguf_file is None:
            return "请上传 GGUF 模型文件。", [], "", ""
        gguf_path = gguf_file.name
        input_dir = os.path.dirname(gguf_path)
    
    gguf_name = os.path.basename(gguf_path)
    
    # 确定输出目录 - 默认为代码文件所在目录的output/models
    if not output_dir or not os.path.exists(output_dir):
        default_output_dir = os.path.join(BASE_DIR, "output", "models")
        os.makedirs(default_output_dir, exist_ok=True)
        final_output_dir = default_output_dir
    else:
        final_output_dir = output_dir
    
    print(f"[生成 Modelfile] 输入文件: {gguf_path}")
    print(f"[输出目录] {final_output_dir}")
    if mode == "思考模式":
        content = f'''FROM {gguf_name}

TEMPLATE """{{- if .System }}System: {{ .System }}{{ end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1}}
{{- if eq .Role "user" }}
{{- if eq $i 0}}User: {{ .Content }}{{- else }}

User: {{ .Content }}{{- end }}
{{- else if eq .Role "assistant" }}

Assistant: <{{- if and $last .Thinking -}}think>{{ .Thinking }}</think>{{- else }}think>
</think>{{- end }}{{ .Content }}{{- end }}
{{- if and $last (ne .Role "assistant") }}

Assistant:{{- if $.IsThinkSet }} <{{- if not $.Think }}think>
</think>{{- end }}{{- end }}{{- end }}{{- end }}"""

PARAMETER stop """

"""
PARAMETER stop """
User"""

PARAMETER stop "User"
PARAMETER stop "Assistant"

PARAMETER temperature 1
PARAMETER top_p 0.5
PARAMETER repeat_penalty 1.2'''
    else:
        content = f'''FROM {gguf_name}

TEMPLATE """{{- if .System }}System: {{ .System }}{{ end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1}}
{{- if eq .Role "user" }}
{{- if eq $i 0}}User: {{ .Content }}{{- else }}

User: {{ .Content }}{{- end }}
{{- else if eq .Role "assistant" }}

Assistant:{{ .Content }}{{- end }}
{{- if and $last (ne .Role "assistant") }}

Assistant:{{- end -}}{{- end }}"""

PARAMETER stop """

"""
PARAMETER stop """
User"""

PARAMETER temperature 1
PARAMETER top_p 0.5
PARAMETER repeat_penalty 1.2'''
    modelfile_path = os.path.join(final_output_dir, "Modelfile")
    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[输出文件] {modelfile_path}")
    return f"已生成 Modelfile -> {modelfile_path}", [modelfile_path], input_dir, final_output_dir

def build_ui():
    """Build the Gradio UI."""
    with gr.Blocks(
        title="RWKV 模型工具箱",
        theme=gr.themes.Ocean(),
    ) as demo:
        gr.Markdown(
            """# 🚀 RWKV 模型工具箱
            
            <div class="info-box">
            💡 <strong>功能说明：</strong><br>
            • <strong>模型转换</strong>：将 PTH 格式的 RWKV 模型转换为 GGUF 或 SafeTensors 格式<br>
            • <strong>Modelfile 生成</strong>：为 GGUF 模型生成 Ollama 兼容的 Modelfile 配置文件
            </div>
            """
        )
        
        with gr.Tabs():
            # 模型转换 Tab
            with gr.TabItem("🔄 模型转换"):
                gr.Markdown("### PTH 模型转换为 GGUF/SafeTensors")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("**📁 文件上传方式**")
                        upload_mode = gr.Radio(
                            choices=["单个文件", "文件夹"],
                            value="单个文件",
                            label="上传模式"
                        )
                        
                        pth_input = gr.File(
                            label="📁 上传 PTH 模型文件", 
                            file_types=[".pth"],
                            height=100,
                            visible=True
                        )
                        
                        folder_input = gr.Textbox(
                            label="📂 输入文件夹路径 (包含PTH文件的目录)",
                            placeholder="例如: D:\\models\\rwkv",
                            visible=False
                        )
                        
                        vocab_input = gr.File(
                            label="📝 上传词表文件 (可选，默认使用内置词表)", 
                            file_types=[".txt"],
                            height=100
                        )
                        
                        def toggle_upload_mode(mode):
                            if mode == "单个文件":
                                return gr.update(visible=True), gr.update(visible=False)
                            else:
                                return gr.update(visible=False), gr.update(visible=True)
                        
                        upload_mode.change(
                            toggle_upload_mode,
                            inputs=[upload_mode],
                            outputs=[pth_input, folder_input]
                        )
                    
                    with gr.Column(scale=1):
                        format_radio = gr.Radio(
                            choices=["GGUF", "SafeTensors", "Both"],
                            value="GGUF",
                            label="🎯 目标格式"
                        )
                        outtype_dropdown = gr.Dropdown(
                             choices=["f16", "f32", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0"],
                             value="f16",
                             label="⚙️ GGUF 精度",
                             info="f16: 半精度浮点, q4_0/q4_1: 4位量化, q8_0: 8位量化"
                         )
                
                with gr.Row():
                    input_dir_display = gr.Textbox(
                        label="📂 输入文件目录",
                        interactive=False,
                        placeholder="选择文件后自动显示"
                    )
                    output_dir = gr.Textbox(
                         label="📁 输出目录 (可选，默认为output/models)",
                         placeholder="留空则输出到output/models目录",
                         value=os.path.join(BASE_DIR, "output", "models")
                     )
                
                convert_btn = gr.Button(
                    "🚀 开始转换", 
                    variant="primary", 
                    size="lg",
                    elem_classes=["primary-btn"]
                )
                
                with gr.Row():
                    with gr.Column(scale=2):
                        log_output = gr.Textbox(
                            label="📋 转换日志", 
                            lines=12, 
                            interactive=False,
                            show_copy_button=True
                        )
                    with gr.Column(scale=1):
                        files_output = gr.File(
                             label="⬇️ 下载转换后的文件", 
                             file_count="multiple",
                             height=200
                         )
                        output_dir_display = gr.Textbox(
                            label="📁 实际输出目录",
                            interactive=False,
                            placeholder="转换完成后显示"
                        )

                def on_format_change(fmt):
                    return gr.update(visible=fmt in ("GGUF", "Both"))

                format_radio.change(on_format_change, inputs=format_radio, outputs=outtype_dropdown)
                convert_btn.click(
                    convert_model, 
                    inputs=[pth_input, vocab_input, format_radio, outtype_dropdown, output_dir, upload_mode, folder_input], 
                    outputs=[log_output, files_output, input_dir_display, output_dir_display]
                )

            # Modelfile 生成 Tab
            with gr.TabItem("📄 Modelfile 生成"):
                gr.Markdown("### 从 GGUF 模型生成 Ollama Modelfile")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("**📁 文件上传方式**")
                        gen_upload_mode = gr.Radio(
                            choices=["单个文件", "文件夹"],
                            value="单个文件",
                            label="上传模式"
                        )
                        
                        gguf_input = gr.File(
                            label="📁 上传 GGUF 模型文件", 
                            file_types=[".gguf"],
                            height=120,
                            visible=True
                        )
                        
                        gen_folder_input = gr.Textbox(
                            label="📂 输入文件夹路径 (包含GGUF文件的目录)",
                            placeholder="例如: D:\\models\\gguf",
                            visible=False
                        )
                        
                        def toggle_gen_upload_mode(mode):
                            if mode == "单个文件":
                                return gr.update(visible=True), gr.update(visible=False)
                            else:
                                return gr.update(visible=False), gr.update(visible=True)
                        
                        gen_upload_mode.change(
                            toggle_gen_upload_mode,
                            inputs=[gen_upload_mode],
                            outputs=[gguf_input, gen_folder_input]
                        )
                    with gr.Column(scale=1):
                        mode_radio = gr.Radio(
                             choices=["思考模式", "非思考模式"],
                             value="非思考模式",
                             label="🧠 模式选择",
                             info="思考模式：启用 CoT 推理\n非思考模式：直接回答"
                         )
                
                with gr.Row():
                    gen_input_dir_display = gr.Textbox(
                        label="📂 输入文件目录",
                        interactive=False,
                        placeholder="选择文件后自动显示"
                    )
                    gen_output_dir = gr.Textbox(
                         label="📁 输出目录 (可选，默认为output/models)",
                         placeholder="留空则输出到output/models目录",
                         value=os.path.join(BASE_DIR, "output", "models")
                     )
                
                gen_btn = gr.Button(
                    "📄 生成 Modelfile", 
                    variant="primary", 
                    size="lg",
                    elem_classes=["primary-btn"]
                )
                
                with gr.Row():
                    with gr.Column(scale=2):
                        mf_log = gr.Textbox(
                            label="📋 生成结果", 
                            lines=8, 
                            interactive=False,
                            show_copy_button=True
                        )
                    with gr.Column(scale=1):
                        mf_file = gr.File(
                            label="⬇️ 下载 Modelfile", 
                            file_count="multiple",
                            height=150
                        )
                        gen_output_dir_display = gr.Textbox(
                            label="📁 实际输出目录",
                            interactive=False,
                            placeholder="生成完成后显示"
                        )
                
                gen_btn.click(
                    generate_modelfile, 
                    inputs=[gguf_input, mode_radio, gen_output_dir, gen_upload_mode, gen_folder_input], 
                    outputs=[mf_log, mf_file, gen_input_dir_display, gen_output_dir_display]
                )
        
        # 添加页脚信息
        gr.Markdown(
            """---
            <div style="text-align: center; color: #666; margin-top: 20px;">
            💻 RWKV 模型工具箱 | 🔧 支持 PTH → GGUF/SafeTensors 转换 | 📄 Ollama Modelfile 生成
            </div>
            """
        )
        
    return demo


def main():
    demo = build_ui()
    demo.launch()


if __name__ == "__main__":
    main()