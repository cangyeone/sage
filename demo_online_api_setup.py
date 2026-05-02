#!/usr/bin/env python3
"""
完整演示：配置和使用 Qwen/DeepSeek 等在线 API

演示场景：
1. 配置 DeepSeek API
2. 自动检测可用模型
3. 测试 LLM Agent 连接
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config_manager import LLMConfigManager
from backend_manager import ONLINE_PROVIDERS


def demo_setup_deepseek():
    """演示：配置 DeepSeek API"""
    print("=" * 70)
    print("演示 1: 配置 DeepSeek API")
    print("=" * 70)
    print()
    
    # 你需要从 https://platform.deepseek.com/ 获取 API Key
    api_key = input("请输入你的 DeepSeek API Key (sk-...): ").strip()
    
    if not api_key:
        print("❌ API Key 为空，跳过")
        return False
    
    cfg = LLMConfigManager()
    
    # 1. 设置 Provider
    print("正在配置 DeepSeek...")
    cfg.set_llm_provider('deepseek')
    print("✓ Provider 设置为 deepseek")
    
    # 2. 设置 API Key
    cfg.set_api_key(api_key)
    print("✓ API Key 已保存")
    
    # 3. 获取可用模型列表
    print("\n正在获取可用模型列表...")
    models = cfg.get_online_api_models()
    
    if not models:
        print("❌ 无法获取模型列表，请检查 API Key 是否正确")
        return False
    
    print(f"✓ 成功获取 {len(models)} 个模型：")
    for i, model in enumerate(models[:10], 1):
        print(f"  {i}. {model}")
    if len(models) > 10:
        print(f"  ... 以及其他 {len(models) - 10} 个模型")
    print()
    
    # 4. 选择一个模型
    if 'deepseek-v4-flash' in models:
        selected_model = 'deepseek-v4-flash'
        print("自动选择: deepseek-v4-flash (快速)")
    else:
        selected_model = models[0]
        print(f"自动选择: {selected_model}")
    
    cfg.set_llm_model(selected_model)
    print(f"✓ 模型已设置为: {selected_model}")
    print()
    
    # 5. 验证配置
    print("验证配置...")
    llm_cfg = cfg.get_llm_config()
    print(f"  Provider: {llm_cfg['provider']}")
    print(f"  Model:    {llm_cfg['model']}")
    print(f"  API Base: {llm_cfg['api_base']}")
    print(f"  API Key:  {llm_cfg['api_key'][:10]}... (已隐藏)")
    print()
    
    return True


def demo_test_agent():
    """演示：测试 LLM Agent 连接"""
    print("=" * 70)
    print("演示 2: 测试 LLM Agent 连接")
    print("=" * 70)
    print()
    
    from conversational_agent import ConversationalAgent
    
    print("初始化 ConversationalAgent...")
    agent = ConversationalAgent()
    
    if agent.llm_agent:
        print("✓ 在线 API 已连接！")
        print(f"  使用 provider: {agent.config_manager.get_llm_config()['provider']}")
        print(f"  使用 model: {agent.config_manager.get_llm_config()['model']}")
        print()
        
        # 测试对话
        print("测试对话...")
        print("-" * 70)
        
        test_message = "你好，请简短地介绍一下你能做什么"
        print(f"用户: {test_message}")
        print()
        
        result = agent.process_message(test_message)
        print(f"AI 助手: {result['response']}")
        print("-" * 70)
        
        return True
    else:
        print("❌ LLM Agent 连接失败")
        print(f"  错误: {agent.llm_error}")
        print()
        print("可能的原因：")
        print("  1. API Key 不正确")
        print("  2. 网络连接问题")
        print("  3. 模型名称不正确")
        return False


def demo_compare_providers():
    """演示：对比不同的在线 API provider"""
    print("=" * 70)
    print("演示 3: 支持的在线 API 服务商")
    print("=" * 70)
    print()
    
    for provider, cfg in ONLINE_PROVIDERS.items():
        if provider == 'custom':
            continue
        print(f"🔸 {provider.upper():20} {cfg['display']}")
        print(f"   API Base: {cfg['api_base']}")
        print(f"   默认模型: {cfg['default_model']}")
        print()


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                   SeismicX 在线 API 配置演示                          ║
║                                                                      ║
║ 本脚本演示如何：                                                     ║
║ 1. 配置 DeepSeek/Qwen/OpenAI 等在线 API                              ║
║ 2. 自动检测 API 的所有可用模型                                       ║
║ 3. 测试 LLM Agent 是否正常连接                                       ║
╚══════════════════════════════════════════════════════════════════════╝
""")
    
    while True:
        print("\n请选择操作:")
        print("1. 配置 DeepSeek API")
        print("2. 测试 LLM Agent 连接")
        print("3. 查看支持的服务商")
        print("4. 退出")
        print()
        
        choice = input("请输入选项 (1-4): ").strip()
        
        if choice == '1':
            print()
            if demo_setup_deepseek():
                print("✅ DeepSeek API 配置成功！")
            else:
                print("❌ DeepSeek API 配置失败")
        
        elif choice == '2':
            print()
            if demo_test_agent():
                print("\n✅ LLM Agent 测试成功！")
            else:
                print("\n❌ LLM Agent 测试失败，请先配置 API")
        
        elif choice == '3':
            print()
            demo_compare_providers()
        
        elif choice == '4':
            print("\n再见！")
            break
        
        else:
            print("\n❌ 无效选项，请重试")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序已中止")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
