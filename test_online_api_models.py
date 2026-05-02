#!/usr/bin/env python3
"""
测试在线 API 模型检测功能

使用方法：
    python test_online_api_models.py                    # 检测当前配置的API
    python test_online_api_models.py deepseek <api_key> # 检测DeepSeek API
    python test_online_api_models.py qwen <api_key>     # 检测通义千问（SiliconFlow）
"""

import sys
import json
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from config_manager import LLMConfigManager
from llm_agent import OpenAICompatibleClient


def test_current_config():
    """测试当前配置的在线 API"""
    config_mgr = LLMConfigManager()
    llm_cfg = config_mgr.get_llm_config()
    
    print("=" * 70)
    print("当前 LLM 配置")
    print("=" * 70)
    print(f"Provider: {llm_cfg.get('provider', 'N/A')}")
    print(f"Model:    {llm_cfg.get('model', 'N/A')}")
    print(f"API Base: {llm_cfg.get('api_base', 'N/A')}")
    print(f"API Key:  {'***' if llm_cfg.get('api_key') else '(未设置)'}")
    print()
    
    # 如果是在线 API，尝试获取模型列表
    provider = llm_cfg.get('provider', '').lower()
    if provider != 'ollama':
        print("=" * 70)
        print(f"检测 {provider} API 可用模型...")
        print("=" * 70)
        
        models = config_mgr.get_online_api_models()
        if models:
            print(f"✓ 成功获取 {len(models)} 个模型：")
            for i, model in enumerate(models[:20], 1):  # 显示前20个
                print(f"  {i}. {model}")
            if len(models) > 20:
                print(f"  ... 以及其他 {len(models) - 20} 个模型")
        else:
            print("✗ 无法获取模型列表，请检查 API Key 和 API Base")
    else:
        print("当前配置使用 Ollama 本地模型")


def test_provider(provider: str, api_key: str):
    """测试指定的在线 API provider"""
    print("=" * 70)
    print(f"测试 {provider.upper()} API")
    print("=" * 70)
    
    # 获取 API 配置
    from backend_manager import ONLINE_PROVIDERS
    
    if provider not in ONLINE_PROVIDERS:
        print(f"✗ 不支持的 provider: {provider}")
        print(f"  支持的 provider: {', '.join(ONLINE_PROVIDERS.keys())}")
        return
    
    provider_cfg = ONLINE_PROVIDERS[provider]
    api_base = provider_cfg["api_base"]
    
    print(f"Provider: {provider}")
    print(f"API Base: {api_base}")
    print()
    
    # 测试连接和获取模型列表
    print("正在检测可用模型...")
    models = OpenAICompatibleClient.list_models(api_base, api_key)
    
    if models:
        print(f"✓ 成功获取 {len(models)} 个模型：")
        for i, model in enumerate(models[:20], 1):
            print(f"  {i}. {model}")
        if len(models) > 20:
            print(f"  ... 以及其他 {len(models) - 20} 个模型")
        
        print()
        print("=" * 70)
        print("配置方法（可选）")
        print("=" * 70)
        print("如果你想使用这个 API，可以运行：")
        print()
        print("  from config_manager import LLMConfigManager")
        print("  cfg = LLMConfigManager()")
        print(f"  cfg.set_llm_provider('{provider}')")
        print(f"  cfg.set_api_key('{api_key[:20]}...')")
        print(f"  cfg.set_llm_model('{models[0]}')")  # 使用第一个可用模型
        print()
    else:
        print("✗ 无法获取模型列表")
        print("  请检查：")
        print("  1. API Key 是否正确")
        print("  2. 网络连接是否正常")
        print("  3. API 端点是否可访问")


def main():
    if len(sys.argv) == 1:
        # 没有参数，检测当前配置
        test_current_config()
    elif len(sys.argv) == 3:
        # python test_online_api_models.py <provider> <api_key>
        provider = sys.argv[1].lower()
        api_key = sys.argv[2]
        test_provider(provider, api_key)
    else:
        print("使用方法：")
        print(f"  {sys.argv[0]}                    # 检测当前配置")
        print(f"  {sys.argv[0]} <provider> <api_key>  # 检测指定 API")
        print()
        print("支持的 providers:")
        from backend_manager import ONLINE_PROVIDERS
        for p, cfg in ONLINE_PROVIDERS.items():
            print(f"  - {p:15} {cfg['display']}")


if __name__ == "__main__":
    main()
