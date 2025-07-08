# scripts/gradual_rollout.py - скрипт для постепенного внедрения

import json
import os
from datetime import datetime

class GradualRollout:
    """Управление постепенным внедрением новых функций"""
    
    def __init__(self):
        self.config_path = "configs/feature_flags.json"
        self.load_config()
    
    def load_config(self):
        """Загрузка конфигурации"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {
                'enhanced_analysis': {
                    'enabled': False,
                    'rollout_percentage': 0,
                    'test_users': [],
                    'start_date': None
                }
            }
    
    def enable_for_testing(self, user_id=None):
        """Включить для тестирования"""
        self.config['enhanced_analysis']['test_users'].append(user_id or 'test')
        self.config['enhanced_analysis']['start_date'] = datetime.now().isoformat()
        self.save_config()
        
    def increase_rollout(self, percentage):
        """Увеличить процент пользователей"""
        self.config['enhanced_analysis']['rollout_percentage'] = min(percentage, 100)
        self.save_config()
    
    def save_config(self):
        """Сохранить конфигурацию"""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)