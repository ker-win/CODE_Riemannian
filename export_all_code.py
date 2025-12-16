# -*- coding: utf-8 -*-
"""
將 CODE_Riemannian 的所有 Python 檔案整合到一個 Markdown 文件
"""

from pathlib import Path
from datetime import datetime


def collect_py_files_to_md(source_dir: str, output_file: str = None):
    """
    將指定目錄下的所有 .py 檔案整合到一個 Markdown 文件
    
    Parameters
    ----------
    source_dir : str
        來源目錄路徑
    output_file : str, optional
        輸出的 Markdown 檔案路徑
    """
    source_path = Path(source_dir)
    
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = source_path / f"all_code_{timestamp}.md"
    else:
        output_file = Path(output_file)
    
    # 收集所有 .py 檔案
    py_files = sorted(source_path.rglob("*.py"))
    
    print(f"找到 {len(py_files)} 個 Python 檔案")
    
    # 整合到 Markdown
    md_content = []
    md_content.append(f"# {source_path.name} 程式碼整合\n")
    md_content.append(f"產生時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    md_content.append(f"共 {len(py_files)} 個檔案\n")
    md_content.append("---\n")
    
    # 目錄
    md_content.append("## 目錄\n")
    for i, py_file in enumerate(py_files, 1):
        rel_path = py_file.relative_to(source_path)
        anchor = str(rel_path).replace("/", "-").replace("\\", "-").replace(".", "-")
        md_content.append(f"{i}. [{rel_path}](#{anchor})\n")
    md_content.append("\n---\n")
    
    # 各檔案內容
    for py_file in py_files:
        rel_path = py_file.relative_to(source_path)
        anchor = str(rel_path).replace("/", "-").replace("\\", "-").replace(".", "-")
        
        md_content.append(f"\n## {rel_path} {{#{anchor}}}\n")
        md_content.append(f"**路徑**: `{rel_path}`\n\n")
        md_content.append("```python\n")
        
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            md_content.append(content)
        except Exception as e:
            md_content.append(f"# 讀取失敗: {e}\n")
        
        if not content.endswith('\n'):
            md_content.append('\n')
        md_content.append("```\n")
        md_content.append("\n---\n")
    
    # 寫入檔案
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(md_content)
    
    print(f"已儲存至: {output_file}")
    return output_file


if __name__ == "__main__":
    # CODE_Riemannian 目錄
    code_dir = Path(__file__).parent
    
    # 輸出到 CODE_Riemannian 目錄
    output = code_dir / "CODE_Riemannian_全部程式碼.md"
    
    collect_py_files_to_md(code_dir, output)
