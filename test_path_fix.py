"""
测试路径提取修复，以及验证 pnsn/data/waveform 下的 SAC 文件能读取和绘图。

运行：
    cd /path/to/sage
    python3 test_path_fix.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from conversational_agent import IntentClassifier, SkillExecutor, ConversationContext


def test_path_extraction_with_chinese():
    """测试路径提取不会吞掉中文字符"""
    clf = IntentClassifier()

    cases = [
        # (输入文本, 期望提取出的路径)
        ("绘制 /Users/yuziye/Documents/GitHub/sage/pnsn/data/waveform 目录中的文件",
         "/Users/yuziye/Documents/GitHub/sage/pnsn/data/waveform"),
        ("查看 /pnsn/data/waveform目录中的文件",
         "/pnsn/data/waveform"),
        ("绘制 ./data/test.sac 文件",
         "./data/test.sac"),
        ("绘制 /pnsn/data/waveform/X1.53085.01.BHZ.D.20122080726235953.sac",
         "/pnsn/data/waveform/X1.53085.01.BHZ.D.20122080726235953.sac"),
    ]

    all_ok = True
    for text, expected in cases:
        result = clf._extract_entities(text)
        paths = result.get("file_paths", [])
        if paths and paths[0] == expected:
            print(f"  OK:   {text!r} -> {paths[0]}")
        else:
            print(f"  FAIL: {text!r} -> {paths} (expected {expected})")
            all_ok = False
    return all_ok


def _resolve_waveform_dir():
    """找到本机上的 pnsn/data/waveform 目录"""
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(here, "pnsn", "data", "waveform")
    return candidate


def test_directory_browsing():
    """测试 data_browsing 能正确扫描 waveform 目录"""
    directory = _resolve_waveform_dir()
    exists = os.path.isdir(directory)
    print(f"  目录 {directory} 存在: {exists}")
    if not exists:
        return False

    exec_ = SkillExecutor()
    ctx = ConversationContext()
    result = exec_._execute_data_browsing({"file_paths": [directory]}, ctx)
    print(f"  success={result['success']}, count={result.get('results', {}).get('count')}")
    print(f"  message 片段: {result['message'][:300]}")
    return result["success"]


def test_plot_directory():
    """测试对目录执行 waveform_plotting，会自动挑一个有效文件绘图"""
    directory = _resolve_waveform_dir()
    exec_ = SkillExecutor()
    ctx = ConversationContext()
    result = exec_._execute_waveform_plotting({"file_paths": [directory]}, ctx)
    print(f"  success={result['success']}")
    print(f"  message: {result['message'][:300]}")
    if result.get("results", {}).get("image_path"):
        print(f"  输出图片: {result['results']['image_path']}")
    return result["success"]


if __name__ == "__main__":
    print("=" * 60)
    print("1) 路径提取测试（不依赖外部库，可在任意环境运行）")
    print("=" * 60)
    ok1 = test_path_extraction_with_chinese()

    print()
    print("=" * 60)
    print("2) 目录浏览测试（需要 pnsn/data/waveform 目录存在）")
    print("=" * 60)
    ok2 = test_directory_browsing()

    print()
    print("=" * 60)
    print("3) 目录 -> 自动绘图测试（需要 obspy + matplotlib）")
    print("=" * 60)
    ok3 = test_plot_directory()

    print()
    print("=" * 60)
    print(f"总体结果: 路径={ok1}, 浏览={ok2}, 绘图={ok3}")
    print("=" * 60)
    sys.exit(0 if (ok1 and ok2 and ok3) else 1)
