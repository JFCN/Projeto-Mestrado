"""
Gerenciador de Configuração do Sistema IA Tuberculose
Carrega e valida configurações de múltiplas fontes

Prioridade:
1. Argumentos da linha de comando
2. Variáveis de ambiente
3. Arquivo config_local.yaml (ignorado pelo git)
4. Arquivo config.yaml (padrão)
5. Configurações hardcoded (fallback)
"""

import os
import yaml
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
import logging
from datetime import datetime

logger = logging.getLogger('ConfigManager')

@dataclass
class SystemConfig:
    """Configurações do sistema"""
    name: str = "Sistema IA Tuberculose Infantil"
    version: str = "1.0.0"
    author: str = "Janduhy Finizola da Cunha Neto"
    log_level: str = "INFO"
    max_execution_time: int = 3600

@dataclass
class DataConfig:
    """Configurações de dados"""
    input_path: str = ""
    output_base_dir: str = ""
    required_columns: list = field(default_factory=lambda: ["SITUA_ENCE", "IDADE", "CS_SEXO"])
    target_mapping: dict = field(default_factory=lambda: {"cura": 1, "obito": 0})
    missing_strategy: dict = field(default_factory=lambda: {"categorical": "unknown", "numerical": "median"})

@dataclass
class ModelConfig:
    """Configurações do modelo"""
    use_tabpfn: bool = True
    tabpfn_fallback: str = "xgboost"
    max_features: int = 15
    feature_selection_method: str = "f_classif"
    cv_folds: int = 10
    cv_strategy: str = "stratified"
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    balancing_strategy: str = "undersample_exact"
    meta_model_selection: str = "auto"
    meta_model_cv_folds: int = 5

@dataclass
class LeakageConfig:
    """Configurações anti-leakage"""
    temporal_threshold: float = 0.05
    missing_threshold: float = 0.1
    auc_difference_threshold: float = 0.10
    suspicious_patterns: list = field(default_factory=lambda: [
        "BACILOSC_", "BAC_APOS", "TRANSF", "DIAS", "_APOS_",
        "RESULTADO", "DESFECHO", "ENCERRA", "TRAT"
    ])
    stop_on_critical: bool = True
    continue_on_high: bool = True
    require_manual_approval: bool = False

@dataclass
class ReportsConfig:
    """Configurações de relatórios"""
    generate_word: bool = True
    generate_technical: bool = True
    generate_executive: bool = True
    generate_json_export: bool = True
    create_visualizations: bool = True
    visualization_formats: list = field(default_factory=lambda: ["png"])
    visualization_dpi: int = 300

@dataclass
class PipelineConfig:
    """Configurações do pipeline"""
    save_checkpoints: bool = True
    resume_from_checkpoint: bool = False
    stop_on_failure: bool = False
    parallel_execution: bool = False
    max_workers: int = 4
    log_detailed_progress: bool = True

class ConfigManager:
    """
    Gerenciador centralizado de configurações
    """
    
    def __init__(self, config_file: Optional[str] = None, local_config_file: Optional[str] = None):
        self.config_file = config_file or "config.yaml"
        self.local_config_file = local_config_file or "config_local.yaml"
        self.config = {}
        self._load_configurations()
        
    def _load_configurations(self):
        """Carrega configurações em ordem de prioridade"""

        self.config = self._get_default_config()
        
        main_config = self._load_yaml_config(self.config_file)
        if main_config:
            self.config = self._deep_merge(self.config, main_config)
            logger.info(f"✅ Configuração principal carregada: {self.config_file}")
        
        local_config = self._load_yaml_config(self.local_config_file)
        if local_config:
            self.config = self._deep_merge(self.config, local_config)
            logger.info(f"✅ Configuração local carregada: {self.local_config_file}")
        
        env_config = self._load_env_config()
        if env_config:
            self.config = self._deep_merge(self.config, env_config)
            logger.info("✅ Variáveis de ambiente aplicadas")
        
        args_config = self._load_args_config()
        if args_config:
            self.config = self._deep_merge(self.config, args_config)
            logger.info("✅ Argumentos da linha de comando aplicados")
        
        self._validate_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Configurações padrão hardcoded como fallback"""

        return {
            'system': {
                'name': 'Sistema IA Tuberculose Infantil',
                'version': '1.0.0',
                'author': 'Janduhy Finizola da Cunha Neto',
                'log_level': 'INFO',
                'max_execution_time': 3600
            },
            'model': {
                'use_tabpfn': True,
                'meta_model_type': 'auto',
                'cv_folds': 10,
                'max_features': 15,
                'test_size': 0.2,
                'validation_size': 0.2
            },
            'leakage_detection': {
                'temporal_threshold': 0.05,
                'missing_threshold': 0.1,
                'suspicious_patterns': [
                    'BACILOSC_', 'BAC_APOS', 'TRANSF', 'DIAS',
                    '_APOS_', 'RESULTADO', 'DESFECHO', 'ENCERRA', 'TRAT'
                ]
            },
            'reports': {
                'generate_word': True,
                'generate_technical': True,
                'generate_executive': True,
                'create_visualizations': True
            },
            'pipeline': {
                'stop_on_critical_leakage': True,
                'continue_on_high_leakage': True,
                'save_checkpoints': True,
                'parallel_validation': False
            }
        }
    
    def _load_yaml_config(self, filepath: str) -> Optional[Dict[str, Any]]:
        """Carrega configuração de arquivo YAML"""

        try:
            config_path = Path(filepath)
            if not config_path.exists():
                logger.debug(f"Arquivo de configuração não encontrado: {filepath}")
                return None
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
            logger.debug(f"Configuração carregada de: {filepath}")
            return config
            
        except Exception as e:
            logger.warning(f"Erro ao carregar {filepath}: {e}")
            return None
    
    def _load_env_config(self) -> Dict[str, Any]:
        """Carrega configurações de variáveis de ambiente"""

        env_config = {}
        
        env_mappings = {
            'TB_AI_LOG_LEVEL': ['system', 'log_level'],
            'TB_AI_USE_TABPFN': ['model', 'use_tabpfn'],
            'TB_AI_MAX_FEATURES': ['model', 'max_features'],
            'TB_AI_CV_FOLDS': ['model', 'cv_folds'],
            'TB_AI_INPUT_PATH': ['data', 'input_path'],
            'TB_AI_OUTPUT_DIR': ['data', 'output_base_dir'],
            'TB_AI_TEMPORAL_THRESHOLD': ['leakage_detection', 'temporal_threshold'],
            'TB_AI_GENERATE_WORD': ['reports', 'generate_word'],
            'TB_AI_STOP_ON_CRITICAL': ['pipeline', 'stop_on_critical_leakage']
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                converted_value = self._convert_env_value(value)
                self._set_nested_value(env_config, config_path, converted_value)
        
        return env_config
    
    def _load_args_config(self) -> Dict[str, Any]:
        """Carrega configurações de argumentos da linha de comando"""

        parser = argparse.ArgumentParser(description='Sistema IA Tuberculose Infantil')
        
        parser.add_argument('--data-path', type=str, help='Caminho para arquivo de dados')
        parser.add_argument('--output-dir', type=str, help='Diretório de saída')
        parser.add_argument('--config', type=str, help='Arquivo de configuração')
        
        parser.add_argument('--no-tabpfn', action='store_true', help='Desabilitar TabPFN')
        parser.add_argument('--max-features', type=int, help='Máximo de features')
        parser.add_argument('--cv-folds', type=int, help='Número de folds CV')
        
        parser.add_argument('--no-word', action='store_true', help='Não gerar relatório Word')
        parser.add_argument('--no-viz', action='store_true', help='Não gerar visualizações')
        
        parser.add_argument('--debug', action='store_true', help='Modo debug')
        parser.add_argument('--verbose', action='store_true', help='Log verboso')
        
        try:
            args, unknown = parser.parse_known_args()
        except SystemExit:
            return {}
        
        args_config = {}
        
        if args.data_path:
            self._set_nested_value(args_config, ['data', 'input_path'], args.data_path)
        
        if args.output_dir:
            self._set_nested_value(args_config, ['data', 'output_base_dir'], args.output_dir)
        
        if args.no_tabpfn:
            self._set_nested_value(args_config, ['model', 'use_tabpfn'], False)
        
        if args.max_features:
            self._set_nested_value(args_config, ['model', 'max_features'], args.max_features)
        
        if args.cv_folds:
            self._set_nested_value(args_config, ['model', 'cv_folds'], args.cv_folds)
        
        if args.no_word:
            self._set_nested_value(args_config, ['reports', 'generate_word'], False)
        
        if args.no_viz:
            self._set_nested_value(args_config, ['reports', 'create_visualizations'], False)
        
        if args.debug:
            self._set_nested_value(args_config, ['system', 'log_level'], 'DEBUG')
            self._set_nested_value(args_config, ['debug', 'verbose_logging'], True)
        
        return args_config
    
    def _convert_env_value(self, value: str) -> Union[str, int, float, bool]:
        """Converte string de variável de ambiente para tipo apropriado"""

        if value.lower() in ['true', '1', 'yes', 'on']:
            return True
        elif value.lower() in ['false', '0', 'no', 'off']:
            return False
        
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        return value
    
    def _set_nested_value(self, config: Dict, path: List[str], value: Any):
        """Define valor em configuração"""

        current = config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Merge recursivo de dicionários"""

        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _validate_config(self):
        """Valida configuração final"""

        errors = []
        
        if not self.config.get('system', {}).get('name'):
            errors.append("system.name é obrigatório")
        
        max_features = self.config.get('model', {}).get('max_features', 0)
        if max_features <= 0 or max_features > 200:
            errors.append("model.max_features deve estar entre 1 e 200")
        
        cv_folds = self.config.get('model', {}).get('cv_folds', 0)
        if cv_folds < 3 or cv_folds > 20:
            errors.append("model.cv_folds deve estar entre 3 e 20")
        
        test_size = self.config.get('model', {}).get('test_size', 0)
        if test_size <= 0 or test_size >= 1:
            errors.append("model.test_size deve estar entre 0 e 1")
        
        if errors:
            error_msg = "Erros de configuração:\n" + "\n".join(f"- {error}" for error in errors)
            raise ValueError(error_msg)
        
        logger.info("✅ Configuração validada com sucesso")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Obtém valor de configuração por caminho"""

        keys = key_path.split('.')
        current = self.config
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any):
        """Define valor de configuração por caminho"""

        keys = key_path.split('.')
        self._set_nested_value(self.config, keys, value)
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Obtém seção completa de configuração"""

        return self.config.get(section, {})
    
    def save_config(self, filepath: str):
        """Salva configuração atual em arquivo"""

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"✅ Configuração salva em: {filepath}")
        except Exception as e:
            logger.error(f"❌ Erro ao salvar configuração: {e}")
    
    def print_config(self):
        """Imprime configuração atual"""

        print("="*60)
        print("CONFIGURAÇÃO ATUAL")
        print("="*60)
        print(yaml.dump(self.config, default_flow_style=False, allow_unicode=True))
        print("="*60)
    
    def get_runtime_info(self) -> Dict[str, Any]:
        """Retorna informações de runtime"""

        return {
            'config_loaded_at': datetime.now().isoformat(),
            'config_sources': {
                'main_config_exists': Path(self.config_file).exists(),
                'local_config_exists': Path(self.local_config_file).exists(),
                'env_vars_detected': len([k for k in os.environ.keys() if k.startswith('TB_AI_')]),
            },
            'final_config_size': len(str(self.config)),
            'log_level': self.get('system.log_level', 'INFO')
        }

_config_manager = None

def get_config_manager(config_file: Optional[str] = None, 
                      local_config_file: Optional[str] = None, 
                      force_reload: bool = False) -> ConfigManager:
    """
    Retorna instância singleton do ConfigManager
    
    Args:
        config_file: Arquivo de configuração principal
        local_config_file: Arquivo de configuração local
        force_reload: Força recarga da configuração
    """

    global _config_manager
    
    if _config_manager is None or force_reload:
        _config_manager = ConfigManager(config_file, local_config_file)
    
    return _config_manager

def load_config(config_file: str = "config.yaml", **kwargs) -> Dict[str, Any]:
    """
    Função de conveniência para carregar configuração
    """

    config_manager = get_config_manager(config_file)
    
    for key, value in kwargs.items():
        key_path = key.replace('_', '.')
        config_manager.set(key_path, value)
    
    return config_manager.config