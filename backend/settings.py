#!/usr/bin/env python3
"""
Settings management for Paper Labeler application.
Handles secure storage and retrieval of user configuration including OpenAI API settings.
"""
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from cryptography.fernet import Fernet
import base64
import copy

class SettingsManager:
    """Manages application settings with encryption for sensitive data."""
    
    def __init__(self, settings_dir: str = ".paper_labeler"):
        """Initialize settings manager with default directory."""
        self.settings_dir = Path.home() / settings_dir
        self.settings_dir.mkdir(exist_ok=True)
        
        self.settings_file = self.settings_dir / "settings.json"
        self.key_file = self.settings_dir / ".key"
        
        # Initialize encryption
        self._init_encryption()
        
        # Load existing settings
        self.settings = self._load_settings()
    
    def _init_encryption(self):
        """Initialize or load encryption key."""
        if self.key_file.exists():
            with open(self.key_file, 'rb') as f:
                key = f.read()
        else:
            key = Fernet.generate_key()
            with open(self.key_file, 'wb') as f:
                f.write(key)
            # Make key file readable only by owner
            os.chmod(self.key_file, 0o600)
        
        self.cipher = Fernet(key)
    
    def _encrypt_value(self, value: str) -> str:
        """Encrypt a sensitive value."""
        if not value:
            return ""
        encrypted = self.cipher.encrypt(value.encode())
        return base64.b64encode(encrypted).decode()
    
    def _decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt a sensitive value."""
        if not encrypted_value:
            return ""
        try:
            encrypted_bytes = base64.b64decode(encrypted_value.encode())
            decrypted = self.cipher.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception:
            return ""
    
    def _load_settings(self) -> Dict[str, Any]:
        """Load settings from file."""
        if not self.settings_file.exists():
            return self._get_default_settings()
        
        try:
            with open(self.settings_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return self._get_default_settings()
    
    def _save_settings(self):
        """Save settings to file."""
        with open(self.settings_file, 'w') as f:
            json.dump(self.settings, f, indent=2)
        # Make settings file readable only by owner
        os.chmod(self.settings_file, 0o600)
    
    def _get_default_settings(self) -> Dict[str, Any]:
        """Get default application settings."""
        return {
            "openai": {
                "api_key_encrypted": "",
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-4o-mini",
                "enabled": False
            },
            "processing": {
                "auto_generate_summary": False,
                "auto_generate_keywords": False,
                "batch_size": 10,
                "concurrent_requests": 3
            },
            "ui": {
                "theme": "light",
                "papers_per_page": 25,
                "auto_refresh": True
            },
            "database": {
                "backup_enabled": True,
                "backup_interval_hours": 24,
                "max_backups": 7
            }
        }
    
    def get_setting(self, key_path: str, default: Any = None) -> Any:
        """Get a setting value using dot notation (e.g., 'openai.model')."""
        keys = key_path.split('.')
        value = self.settings
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set_setting(self, key_path: str, value: Any):
        """Set a setting value using dot notation."""
        keys = key_path.split('.')
        setting = self.settings
        
        # Navigate to the parent dict
        for key in keys[:-1]:
            if key not in setting:
                setting[key] = {}
            setting = setting[key]
        
        # Set the value
        setting[keys[-1]] = value
        self._save_settings()
    
    def get_openai_config(self) -> Dict[str, Any]:
        """Get OpenAI configuration with decrypted API key."""
        config = self.settings.get("openai", {})
        
        # Decrypt API key
        encrypted_key = config.get("api_key_encrypted", "")
        api_key = self._decrypt_value(encrypted_key) if encrypted_key else ""
        
        return {
            "api_key": api_key,
            "base_url": config.get("base_url", "https://api.openai.com/v1"),
            "model": config.get("model", "gpt-4o-mini"),
            "enabled": config.get("enabled", False) and bool(api_key)
        }
    
    def set_openai_config(self, api_key: str = None, base_url: str = None, 
                         model: str = None, enabled: bool = None):
        """Set OpenAI configuration with encrypted API key."""
        if "openai" not in self.settings:
            self.settings["openai"] = {}
        
        if api_key is not None:
            # Encrypt and store API key
            encrypted_key = self._encrypt_value(api_key)
            self.settings["openai"]["api_key_encrypted"] = encrypted_key
        
        if base_url is not None:
            self.settings["openai"]["base_url"] = base_url
        
        if model is not None:
            self.settings["openai"]["model"] = model
        
        if enabled is not None:
            self.settings["openai"]["enabled"] = enabled
        
        self._save_settings()
    
    def test_openai_connection(self) -> Dict[str, Any]:
        """Test OpenAI connection with current settings."""
        config = self.get_openai_config()
        
        if not config["enabled"] or not config["api_key"]:
            return {
                "success": False,
                "error": "OpenAI API key not configured"
            }
        
        try:
            # Import here to avoid dependency if OpenAI not used
            import sys
            sys.path.append(str(Path(__file__).parent.parent))
            from src.openai_api import get_openai_client, get_openai_response
            
            client = get_openai_client(config["api_key"], config["base_url"])
            
            # Test with a simple prompt
            test_abstract = "This is a test abstract to verify API connectivity."
            response = get_openai_response(client, test_abstract, config["model"])
            
            return {
                "success": True,
                "message": "Connection successful",
                "model": config["model"]
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_all_settings(self) -> Dict[str, Any]:
        """Get all settings (with API key masked)."""
        settings_copy = copy.deepcopy(self.settings)
        
        # Mask API key in response
        if "openai" in settings_copy and "api_key_encrypted" in settings_copy["openai"]:
            encrypted_key = settings_copy["openai"]["api_key_encrypted"]
            if encrypted_key:
                settings_copy["openai"]["api_key_masked"] = "***" + encrypted_key[-4:]
            else:
                settings_copy["openai"]["api_key_masked"] = ""
            
            # Remove encrypted key from response
            del settings_copy["openai"]["api_key_encrypted"]
        
        return settings_copy
    
    def reset_settings(self):
        """Reset all settings to defaults."""
        self.settings = self._get_default_settings()
        self._save_settings()
    
    def export_settings(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Export settings for backup (optionally including sensitive data)."""
        if include_sensitive:
            return self.settings.copy()
        else:
            settings_copy = self.settings.copy()
            if "openai" in settings_copy:
                settings_copy["openai"]["api_key_encrypted"] = ""
            return settings_copy
    
    def import_settings(self, settings_data: Dict[str, Any]):
        """Import settings from backup."""
        # Validate settings structure
        default_settings = self._get_default_settings()
        
        # Merge with defaults to ensure all keys exist
        def merge_settings(default: dict, imported: dict) -> dict:
            result = default.copy()
            for key, value in imported.items():
                if key in result:
                    if isinstance(result[key], dict) and isinstance(value, dict):
                        result[key] = merge_settings(result[key], value)
                    else:
                        result[key] = value
            return result
        
        self.settings = merge_settings(default_settings, settings_data)
        self._save_settings()

# Global settings instance
_settings_manager = None

def get_settings_manager() -> SettingsManager:
    """Get global settings manager instance."""
    global _settings_manager
    if _settings_manager is None:
        _settings_manager = SettingsManager()
    return _settings_manager