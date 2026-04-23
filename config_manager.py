#!/usr/bin/env python3
"""
LLM Configuration Manager

Manages LLM model selection and configuration for SeismicX.
Supports Ollama / vLLM / online API (via BackendManager).
Configuration is saved to ~/.seismicx/config.json
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional


class LLMConfigManager:
    """Manages LLM configuration for SeismicX"""

    def __init__(self):
        self.config_dir = Path.home() / '.seismicx'
        self.config_file = self.config_dir / 'config.json'
        self.config_dir.mkdir(exist_ok=True)
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        """Load configuration from file"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        # Default configuration
        return {
            'llm': {
                'provider': 'ollama',  # ollama, openai, anthropic, etc.
                'model': 'qwen2.5:7b',  # Default model
                'api_base': 'http://localhost:11434',  # For Ollama
                'api_key': '',  # For online APIs
                'temperature': 0.7,
                'max_tokens': 2000
            },
            'first_run': True
        }

    def save_config(self):
        """Save configuration to file"""
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

    def get_llm_config(self) -> Dict:
        """
        Get current LLM configuration.
        Reads directly from config.json (set by the LLM settings UI).
        This is the single source of truth for all components.
        """
        return self.config.get('llm', {
            'provider': 'ollama',
            'model': '',
            'api_base': 'http://localhost:11434',
            'api_key': '',
        })

    def set_llm_provider(self, provider: str):
        """Set LLM provider"""
        valid_providers = ['ollama', 'openai', 'anthropic', 'azure', 'custom']
        if provider not in valid_providers:
            raise ValueError(f"Invalid provider: {provider}. Must be one of {valid_providers}")

        self.config['llm']['provider'] = provider

        # Set default API base for providers
        if provider == 'ollama':
            self.config['llm']['api_base'] = 'http://localhost:11434'
            self.config['llm']['api_key'] = ''
        elif provider == 'openai':
            self.config['llm']['api_base'] = 'https://api.openai.com/v1'
        elif provider == 'anthropic':
            self.config['llm']['api_base'] = 'https://api.anthropic.com'
        elif provider == 'azure':
            self.config['llm']['api_base'] = 'https://YOUR_RESOURCE.openai.azure.com/'

        self.save_config()

    def set_llm_model(self, model: str):
        """Set LLM model"""
        self.config['llm']['model'] = model
        self.save_config()

    def set_api_key(self, api_key: str):
        """Set API key for online providers"""
        self.config['llm']['api_key'] = api_key
        self.save_config()

    def set_api_base(self, api_base: str):
        """Set API base URL"""
        self.config['llm']['api_base'] = api_base
        self.save_config()

    def mark_first_run_complete(self):
        """Mark first run as complete"""
        self.config['first_run'] = False
        self.save_config()

    def is_first_run(self) -> bool:
        """Check if this is the first run"""
        return self.config.get('first_run', True)

    @staticmethod
    def check_ollama_available() -> bool:
        """Check if Ollama is installed and running"""
        try:
            result = subprocess.run(
                ['ollama', 'list'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    @staticmethod
    def get_ollama_models() -> List[str]:
        """Get list of available Ollama models"""
        try:
            result = subprocess.run(
                ['ollama', 'list'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                return []

            models = []
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            for line in lines:
                if line.strip():
                    # Extract model name (first column)
                    model_name = line.split()[0]
                    models.append(model_name)

            return models
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return []

    @staticmethod
    def pull_ollama_model(model_name: str) -> bool:
        """Pull an Ollama model"""
        try:
            print(f"Pulling model: {model_name}...")
            result = subprocess.run(
                ['ollama', 'pull', model_name],
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            print("Error: Pull timed out")
            return False
        except FileNotFoundError:
            print("Error: Ollama not found")
            return False

    def get_recommended_models(self) -> Dict[str, List[Dict]]:
        """Get recommended models for different use cases"""
        return {
            'ollama': [
                {'name': 'qwen2.5:7b', 'description': 'Qwen 2.5 7B - Good balance of speed and quality', 'size': '~4GB'},
                {'name': 'qwen2.5:14b', 'description': 'Qwen 2.5 14B - Better quality, slower', 'size': '~8GB'},
                {'name': 'llama3.2:3b', 'description': 'Llama 3.2 3B - Fast, lightweight', 'size': '~2GB'},
                {'name': 'llama3.1:8b', 'description': 'Llama 3.1 8B - Good general purpose', 'size': '~4.7GB'},
                {'name': 'mistral:7b', 'description': 'Mistral 7B - Efficient and capable', 'size': '~4.1GB'},
                {'name': 'deepseek-coder:6.7b', 'description': 'DeepSeek Coder - Great for code tasks', 'size': '~3.8GB'},
            ],
            'online': [
                {'name': 'gpt-4o', 'provider': 'OpenAI', 'description': 'GPT-4 Optimized - Best quality'},
                {'name': 'gpt-4o-mini', 'provider': 'OpenAI', 'description': 'GPT-4 Mini - Good balance'},
                {'name': 'claude-3-sonnet-20240229', 'provider': 'Anthropic', 'description': 'Claude 3 Sonnet - Excellent reasoning'},
                {'name': 'claude-3-haiku-20240307', 'provider': 'Anthropic', 'description': 'Claude 3 Haiku - Fast and efficient'},
            ]
        }

    def interactive_setup(self):
        """Interactive setup wizard for first-time users"""
        print("=" * 80)
        print("SeismicX - First Time Setup")
        print("=" * 80)
        print()

        # Check Ollama availability
        ollama_available = self.check_ollama_available()

        if ollama_available:
            print("✓ Ollama detected on your system")
            ollama_models = self.get_ollama_models()
            if ollama_models:
                print(f"✓ Found {len(ollama_models)} installed model(s):")
                for model in ollama_models[:5]:  # Show first 5
                    print(f"  - {model}")
                if len(ollama_models) > 5:
                    print(f"  ... and {len(ollama_models) - 5} more")
            else:
                print("⚠ No Ollama models installed yet")
        else:
            print("⚠ Ollama not detected")
            print("  To use local models, install Ollama from: https://ollama.ai")

        print()
        print("Choose your LLM provider:")
        print("  1. Ollama (Local models - Recommended)")
        print("  2. OpenAI (GPT-4, GPT-3.5)")
        print("  3. Anthropic (Claude)")
        print("  4. Azure OpenAI")
        print("  5. Custom API")
        print()

        choice = input("Enter choice (1-5) [1]: ").strip() or '1'

        provider_map = {
            '1': 'ollama',
            '2': 'openai',
            '3': 'anthropic',
            '4': 'azure',
            '5': 'custom'
        }

        provider = provider_map.get(choice, 'ollama')
        self.set_llm_provider(provider)

        if provider == 'ollama':
            self._setup_ollama(ollama_available)
        elif provider in ['openai', 'anthropic', 'azure', 'custom']:
            self._setup_online_api(provider)

        self.mark_first_run_complete()
        print()
        print("=" * 80)
        print("✓ Setup complete! Configuration saved.")
        print("=" * 80)

    def _setup_ollama(self, ollama_available: bool):
        """Setup Ollama configuration"""
        if not ollama_available:
            print()
            print("⚠ Ollama is not installed or not running.")
            print("  Please install Ollama from: https://ollama.ai")
            print("  Then run: ollama serve")
            print()
            install = input("Would you like instructions to install Ollama? (y/n) [n]: ").strip().lower()
            if install == 'y':
                print()
                print("Installation instructions:")
                print("  macOS: brew install ollama")
                print("  Linux: curl -fsSL https://ollama.ai/install.sh | sh")
                print("  Windows: Download from https://ollama.ai")
                print()
                print("After installation, run: ollama serve")
            return

        # Get available models
        models = self.get_ollama_models()
        recommended = self.get_recommended_models()['ollama']

        print()
        print("Available Ollama models:")
        if models:
            for i, model in enumerate(models, 1):
                print(f"  {i}. {model} (installed)")
        else:
            print("  No models installed yet")

        print()
        print("Recommended models to install:")
        for i, model_info in enumerate(recommended[:4], len(models) + 1):
            print(f"  {i}. {model_info['name']} - {model_info['description']} ({model_info['size']})")

        print()
        print("Select a model:")
        print("  - Enter number to choose from above")
        print("  - Or type model name directly (e.g., 'qwen2.5:7b')")
        print()

        model_choice = input("Your choice: ").strip()

        # Check if it's a number
        try:
            idx = int(model_choice) - 1
            all_models = models + [m['name'] for m in recommended]
            if 0 <= idx < len(all_models):
                model = all_models[idx]
            else:
                model = 'qwen2.5:7b'  # Default
        except ValueError:
            model = model_choice if model_choice else 'qwen2.5:7b'

        self.set_llm_model(model)

        # Ask to pull if not installed
        if model not in models:
            print()
            pull = input(f"Model '{model}' not installed. Pull now? (y/n) [y]: ").strip().lower()
            if pull != 'n':
                success = self.pull_ollama_model(model)
                if success:
                    print("✓ Model pulled successfully!")
                else:
                    print("✗ Failed to pull model. You can pull it manually later with:")
                    print(f"  ollama pull {model}")

    def _setup_online_api(self, provider: str):
        """Setup online API configuration"""
        provider_names = {
            'openai': 'OpenAI',
            'anthropic': 'Anthropic',
            'azure': 'Azure OpenAI',
            'custom': 'Custom API'
        }

        print()
        print(f"Configuring {provider_names[provider]} API")
        print()

        # Get API key
        api_key = input(f"Enter your {provider_names[provider]} API key: ").strip()
        if api_key:
            self.set_api_key(api_key)
        else:
            print("⚠ Warning: No API key provided. You can set it later.")

        # Get API base for custom/azure
        if provider in ['azure', 'custom']:
            default_base = self.config['llm']['api_base']
            api_base = input(f"Enter API base URL [{default_base}]: ").strip()
            if api_base:
                self.set_api_base(api_base)

        # Select model
        recommended = self.get_recommended_models()['online']
        print()
        print("Recommended models:")
        for i, model_info in enumerate(recommended, 1):
            print(f"  {i}. {model_info['name']} ({model_info['provider']}) - {model_info['description']}")

        print()
        model = input("Enter model name: ").strip()
        if model:
            self.set_llm_model(model)
        else:
            # Set default based on provider
            defaults = {
                'openai': 'gpt-4o-mini',
                'anthropic': 'claude-3-haiku-20240307',
                'azure': 'gpt-4o-mini',
                'custom': ''
            }
            self.set_llm_model(defaults.get(provider, ''))


# Singleton instance
_config_manager = None


def get_config_manager() -> LLMConfigManager:
    """Get or create config manager singleton"""
    global _config_manager
    if _config_manager is None:
        _config_manager = LLMConfigManager()
    return _config_manager


if __name__ == '__main__':
    # Run interactive setup
    manager = LLMConfigManager()
    manager.interactive_setup()

    # Show current config
    print()
    print("Current configuration:")
    print(json.dumps(manager.get_llm_config(), indent=2))
