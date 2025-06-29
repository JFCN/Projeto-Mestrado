"""
Sistema de IA para Predição de Sobrevivência em Tuberculose Infantil
COM SUBSTITUIÇÃO INTELIGENTE DE FEATURES

Fluxo:
1. Seleção de Features com Substituição Inteligente
2. Análise Temporal Clínica
3. Detecção de Data Leakage
4. Treinamento com TabPFN + Meta-Models
5. Geração de Relatórios

Autor: Janduhy Finizola da Cunha Neto
"""

import os
import sys
import time
import io
import locale
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
import traceback
import json


def ensure_feature_selector_compatibility(feature_selector):
    """
    Garante que o feature_selector tenha os métodos necessários
    """
    if feature_selector is None:
        return None

    if not hasattr(feature_selector, 'transform'):
        logger.warning("⚠️ Feature selector não tem método transform. Adicionando...")

        # Adiciona os métodos necessários dinamicamente
        def fit_method(self, X, y=None):
            if hasattr(self, 'selected_features') and self.selected_features:
                self.feature_names_ = self.selected_features
            else:
                self.feature_names_ = list(X.columns) if hasattr(X, 'columns') else [f'feature_{i}' for i in
                                                                                     range(X.shape[1])]
            return self

        def transform_method(self, X):
            if not hasattr(self, 'feature_names_'):
                return X

            if hasattr(X, 'columns'):
                available_features = [f for f in self.feature_names_ if f in X.columns]
                return X[available_features]
            else:
                return X

        def fit_transform_method(self, X, y=None):
            return self.fit(X, y).transform(X)

        # Adiciona os métodos à instância
        feature_selector.fit = fit_method.__get__(feature_selector, type(feature_selector))
        feature_selector.transform = transform_method.__get__(feature_selector, type(feature_selector))
        feature_selector.fit_transform = fit_transform_method.__get__(feature_selector, type(feature_selector))

        logger.info("✅ Métodos de compatibilidade adicionados ao feature_selector")

    return feature_selector

def apply_categorical_mapping_patch():
    """
    Aplica correção imediata para mapeamento categórico
    """
    import pandas as pd
    import numpy as np
    import logging
    import types

    logger = logging.getLogger('FeatureSelector')

    def _create_score_comorbidades_patched(self):
        """SCORE_COMORBIDADES com mapeamento categórico"""
        try:
            comorbidades = ['AGRAVAIDS', 'AGRAVALCOO', 'AGRAVDIABE', 'AGRAVDOENC',
                            'AGRAVDROGA', 'AGRAVTABAC', 'AGRAVOUTRA']

            logger.info("🔧 Criando SCORE_COMORBIDADES com mapeamento categórico ...")

            # Mapeamento robusto para valores categóricos
            mapping_values = {
                'SIM': 1, 'sim': 1, 'Sim': 1,
                'NÃO': 0, 'não': 0, 'Não': 0, 'NAO': 0, 'nao': 0, 'Nao': 0,
                'IGNORADO': 0, 'ignorado': 0, 'Ignorado': 0,
                '1': 1, '2': 0, '9': 0,
                1: 1, 2: 0, 9: 0,
                np.nan: 0, None: 0, '': 0
            }

            score = pd.Series(0, index=self.df.index)
            valid_comorbidades = []

            for col in comorbidades:
                if col not in self.df.columns:
                    continue

                # Aplicar mapeamento categórico
                col_data = self.df[col].copy()
                col_mapped = col_data.map(mapping_values).fillna(0).astype(int)

                positive_cases = (col_mapped == 1).sum()
                score += col_mapped
                valid_comorbidades.append(col)

                logger.info(f"   ✅ {col}: {positive_cases} casos positivos de {len(col_data)}")

            if not valid_comorbidades:
                logger.error("❌ Nenhuma coluna de comorbidade válida encontrada")
                return False

            # Criar a feature derivada
            self.df['SCORE_COMORBIDADES'] = score
            self.derived_features.append('SCORE_COMORBIDADES')

            # Configurar substituição
            self._register_replacement('SCORE_COMORBIDADES', valid_comorbidades, 'high', 'full')

            # Log do resultado
            score_stats = score.value_counts().sort_index()
            total_cases = len(self.df)

            logger.info("✅ SCORE_COMORBIDADES criado com sucesso!")
            logger.info(f"   📊 Componentes: {len(valid_comorbidades)} comorbidades")
            logger.info(f"   📈 Distribuição:")
            for score_val, count in score_stats.items():
                percentage = (count / total_cases) * 100
                logger.info(f"      Score {score_val}: {count} casos ({percentage:.1f}%)")

            if self.replacement_config['log_replacements']:
                logger.info(f"   🔄 Substituirá: {valid_comorbidades}")

            return True

        except Exception as e:
            logger.error(f"❌ Erro ao criar SCORE_COMORBIDADES: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _create_risco_social_patched(self):
        """RISCO_SOCIAL com mapeamento categórico """
        try:
            pop_especiais = ['POP_LIBER', 'POP_RUA', 'POP_IMIG']

            logger.info("🔧 Criando RISCO_SOCIAL com mapeamento categórico...")

            mapping_values = {
                'SIM': 1, 'sim': 1, 'Sim': 1,
                'NÃO': 0, 'não': 0, 'Não': 0, 'NAO': 0, 'nao': 0, 'Nao': 0,
                'IGNORADO': 0, 'ignorado': 0, 'Ignorado': 0,
                '1': 1, '2': 0, '9': 0,
                1: 1, 2: 0, 9: 0,
                np.nan: 0, None: 0, '': 0
            }

            risco = pd.Series(0, index=self.df.index)
            valid_pops = []

            for col in pop_especiais:
                if col not in self.df.columns:
                    continue

                col_data = self.df[col].copy()
                col_mapped = col_data.map(mapping_values).fillna(0).astype(int)

                positive_cases = (col_mapped == 1).sum()
                risco += col_mapped
                valid_pops.append(col)

                logger.info(f"   ✅ {col}: {positive_cases} casos positivos")

            if not valid_pops:
                logger.warning("⚠️ Nenhuma coluna de população especial encontrada")
                return False

            self.df['RISCO_SOCIAL'] = risco
            self.derived_features.append('RISCO_SOCIAL')
            self._register_replacement('RISCO_SOCIAL', valid_pops, 'high', 'full')

            risco_stats = risco.value_counts().sort_index()
            total_cases = len(self.df)

            logger.info("✅ RISCO_SOCIAL criado com sucesso!")
            logger.info(f"   📊 Componentes: {len(valid_pops)} populações especiais")
            for risco_val, count in risco_stats.items():
                percentage = (count / total_cases) * 100
                logger.info(f"      Risco {risco_val}: {count} casos ({percentage:.1f}%)")

            return True

        except Exception as e:
            logger.error(f"❌ Erro ao criar RISCO_SOCIAL: {e}")
            return False

    def _create_caso_complexo_patched(self):
        """CASO_COMPLEXO com mapeamento categórico """
        try:
            logger.info("🔧 Criando CASO_COMPLEXO com mapeamento categórico ...")

            complexidade = []
            mapping_values = {
                'SIM': 1, 'sim': 1, 'Sim': 1,
                'NÃO': 0, 'não': 0, 'Não': 0, 'NAO': 0, 'nao': 0, 'Nao': 0,
                'IGNORADO': 0, 'ignorado': 0, 'Ignorado': 0,
                '1': 1, '2': 0, '9': 0,
                1: 1, 2: 0, 9: 0,
                np.nan: 0, None: 0, '': 0
            }

            # Forma clínica extrapulmonar
            if 'FORMA' in self.df.columns:
                forma_data = self.df['FORMA'].astype(str).str.strip().str.lower()
                formas_complexas = forma_data.isin([
                    'extrapulmonar', 'extra-pulmonar', 'extra pulmonar',
                    'pulmonar + extrapulmonar', 'pulmonar+extrapulmonar',
                    'pulmonar e extrapulmonar', '3'
                ])
                complexidade.append(formas_complexas)
                logger.info(f"   ✅ FORMA: {formas_complexas.sum()} casos extrapulmonares")

            # HIV positivo
            if 'HIV' in self.df.columns:
                hiv_data = self.df['HIV']
                if hiv_data.dtype == 'object':
                    hiv_mapped = hiv_data.map(mapping_values).fillna(0).astype(int)
                else:
                    hiv_mapped = pd.to_numeric(hiv_data, errors='coerce').fillna(0).astype(int)

                hiv_positivo = (hiv_mapped == 1)
                complexidade.append(hiv_positivo)
                logger.info(f"   ✅ HIV: {hiv_positivo.sum()} casos positivos")

            # Múltiplas comorbidades
            if 'SCORE_COMORBIDADES' in self.df.columns:
                multiplas_comorbidades = (self.df['SCORE_COMORBIDADES'] >= 2)
                complexidade.append(multiplas_comorbidades)
                logger.info(f"   ✅ COMORBIDADES: {multiplas_comorbidades.sum()} casos com ≥2")

            if not complexidade:
                logger.warning("⚠️ Nenhum critério de complexidade disponível")
                return False

            caso_complexo = np.any(complexidade, axis=0).astype(int)
            self.df['CASO_COMPLEXO'] = caso_complexo
            self.derived_features.append('CASO_COMPLEXO')
            self._register_replacement('CASO_COMPLEXO', [], 'medium', 'partial')

            casos_complexos = caso_complexo.sum()
            total_casos = len(self.df)
            percentage = (casos_complexos / total_casos) * 100

            logger.info("✅ CASO_COMPLEXO criado com sucesso!")
            logger.info(f"   📊 Casos complexos: {casos_complexos}/{total_casos} ({percentage:.1f}%)")

            return True

        except Exception as e:
            logger.error(f"❌ Erro ao criar CASO_COMPLEXO: {e}")
            return False

    # Aplicar os patches
    try:
        from features.selecao_features import TuberculosisFeatureSelector

        # Substituir métodos da classe
        TuberculosisFeatureSelector._create_score_comorbidades = _create_score_comorbidades_patched
        TuberculosisFeatureSelector._create_risco_social = _create_risco_social_patched
        TuberculosisFeatureSelector._create_caso_complexo = _create_caso_complexo_patched

        logger.info("✅ Patches categóricos aplicados com sucesso à classe TuberculosisFeatureSelector!")
        return True

    except ImportError as e:
        logger.error(f"❌ Erro ao importar TuberculosisFeatureSelector: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Erro ao aplicar patches: {e}")
        return False


if sys.platform == "win32":
    os.system("chcp 65001 > nul 2>&1")

os.environ["PYTHONIOENCODING"] = "utf-8"

# Adiciona path dos módulos
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "modules"))

def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Configura sistema de logging """

    # Mapear os níveis
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }

    level = level_map.get(log_level.upper(), logging.INFO)

    # LIMPA TODOS OS HANDLERS EXISTENTES
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # REMOVE HANDLERS DE TODOS OS LOGGERS EXISTENTES
    for logger_name in logging.Logger.manager.loggerDict:
        logger_obj = logging.getLogger(logger_name)
        logger_obj.handlers.clear()
        logger_obj.propagate = True

    # CRIA HANDLER CUSTOM
    class EmojiSafeHandler(logging.StreamHandler):
        def emit(self, record):
            try:
                msg = self.format(record)
                if sys.platform == "win32":
                    print(msg, file=sys.stderr, flush=True)
                else:
                    sys.stderr.write(msg + '\n')
                    sys.stderr.flush()
            except (UnicodeEncodeError, AttributeError):
                # Se falhar, substituir emojis
                safe_msg = self._make_emoji_safe(msg)
                print(safe_msg, file=sys.stderr, flush=True)
            except Exception:
                try:
                    fallback_msg = record.getMessage().encode('ascii', 'replace').decode('ascii')
                    print(f"{record.levelname}: {fallback_msg}", file=sys.stderr, flush=True)
                except:
                    print(f"{record.levelname}: [ERRO DE ENCODING]", file=sys.stderr, flush=True)

        def _make_emoji_safe(self, text):
            """Substitui emojis por texto ASCII"""
            emoji_map = {
                "🚨": "[ALERTA]", "❌": "[ERRO]", "⚠️": "[AVISO]", "✅": "[OK]",
                "🔧": "[CONFIG]", "📊": "[DADOS]", "🎯": "[META]", "🏆": "[MELHOR]",
                "📄": "[RELATORIO]", "🔍": "[BUSCA]", "🕐": "[TEMPO]", "🤖": "[IA]",
                "📋": "[LISTA]", "🎉": "[SUCESSO]", "🏥": "[HOSPITAL]", "⚙️": "[GEAR]",
                "📁": "[PASTA]", "📝": "[NOTA]", "🚀": "[ROCKET]", "▶️": "[PLAY]",
                "⏱️": "[TIMER]", "💾": "[SAVE]", "🔥": "[FIRE]", "📧": "[EMAIL]", "📱": "[PHONE]",
                "🔄": "[DERIVADA]"  # Adiciona ícone para features derivadas
            }

            result = text
            for emoji, replacement in emoji_map.items():
                result = result.replace(emoji, replacement)
            return result

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Configura handler principal
    console_handler = EmojiSafeHandler()
    console_handler.setFormatter(formatter)

    # File handler
    handlers = [console_handler]
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)
        except Exception as e:
            print(f"Aviso: Erro no log de arquivo: {e}")

    # Configura logging root
    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True
    )

    # TODOS os loggers usem este handler
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Configura propagação para todos os loggers existentes
    for logger_name in logging.Logger.manager.loggerDict:
        logger_obj = logging.getLogger(logger_name)
        logger_obj.propagate = True
        logger_obj.setLevel(logging.NOTSET)

# Configura logging inicial GLOBAL
setup_logging()

# DEFINE LOGGER APÓS CONFIGURAÇÃO
logger = logging.getLogger('TuberculosisAI')

for logger_name in ['AdvancedLeakageDetector', 'DataLeakageDetector', 'ClinicalTimelineAnalyzer',
                   'TuberculosisPredictor', 'ComprehensiveReportGenerator', 'FeatureSelector']:
    temp_logger = logging.getLogger(logger_name)
    temp_logger.propagate = True

try:
    from configs.config_manager import get_config_manager, ConfigManager
except ImportError:
    logger.warning("⚠️ Sistema de configuração não encontrado. Usando configuração básica.")
    ConfigManager = None

try:
    from features.selecao_features import run_feature_selection_for_main, TuberculosisFeatureSelector
    FEATURE_SELECTOR_AVAILABLE = True
    logger.info("✅ Sistema de seleção de features disponível")
except ImportError:
    FEATURE_SELECTOR_AVAILABLE = False
    logger.error("❌ selecao_features.py não encontrado!")

try:
    from modules.validacao_hipotese import run_statistical_validation as statistical_validation_func
    STATISTICAL_VALIDATION_AVAILABLE = True
    logger.info("✅ Módulo de validação estatística importado com sucesso")
except ImportError as e:
    STATISTICAL_VALIDATION_AVAILABLE = False
    logger.error(f"❌ Erro ao importar modules.validacao_hipotese: {e}")

class TuberculosisAISystem:
    """
    Sistema Principal da IA para Tuberculose Infantil
    COM SUBSTITUIÇÃO INTELIGENTE DE FEATURES
    Orquestra todos os microserviços em sequência lógica
    """

    def __init__(self, config_file: str = None, config_overrides: Dict[str, Any] = None):

        # Carrega configuração
        if ConfigManager:
            self.config_manager = get_config_manager(config_file)
            self.config = self.config_manager.config
        else:
            self.config = self._load_default_config()

        if config_overrides:
            self._apply_config_overrides(config_overrides)

        # Reconfigura logging com nível do config
        log_level = self.config.get('system', {}).get('log_level', 'INFO')
        setup_logging(log_level)

        # Inicializa sistema
        self.results = {}
        self.data_path = None
        self.output_dir = None
        self.start_time = None
        self.step_times = {}
        self.feature_selector = None

        # Verifica disponibilidade do seletor de features PRIMEIRO
        if not FEATURE_SELECTOR_AVAILABLE:
            logger.error("❌ SISTEMA NÃO PODE CONTINUAR SEM selecao_features.py!")
            raise ImportError("selecao_features.py é obrigatório para o funcionamento")

        # Status do pipeline
        self.pipeline_status = {
            'feature_selection': False,
            'clinical_analysis': False,
            'leakage_detection': False,
            'model_training': False,
            'validacao_hipotese': False,
            'report_generation': False
        }

        # Valida dependências
        self._validate_dependencies()

        # Mostra cenários que serão executados
        scenarios = self._get_scenarios_from_config()
        logger.info(f"🎯 Cenários configurados: {scenarios}")

        # Log inicial com foco na substituição inteligente
        system_info = self.config.get('system', {})
        logger.info(f"🏥 {system_info.get('name', 'Sistema IA Tuberculose')} v{system_info.get('version', '1.0')}")
        logger.info("📋 Arquitetura: Microserviços com SUBSTITUIÇÃO INTELIGENTE DE FEATURES")
        logger.info(f"⚙️ Configuração carregada: {len(self.config)} seções")

        # Log da configuração de substituição
        replacement_config = self.config.get('feature_selection', {}).get('feature_replacement', {})
        if replacement_config.get('enable_smart_replacement', True):
            logger.info("🔄 SUBSTITUIÇÃO INTELIGENTE DE FEATURES HABILITADA")
            logger.info(f"   Estratégia: {replacement_config.get('replacement_strategy', 'replace_originals')}")
            logger.info(f"   Priorizar derivadas: {replacement_config.get('prioritize_derived_features', True)}")
        else:
            logger.warning("⚠️ Substituição inteligente desabilitada")

        if self.config_manager:
            runtime_info = self.config_manager.get_runtime_info()
            logger.info(f"📊 Fontes de configuração: {runtime_info['config_sources']}")

    def _apply_config_overrides(self, overrides: Dict[str, Any]):
        """Aplica overrides de configuração"""
        for key, value in overrides.items():
            # Converte chaves com underscore para pontos
            if '_' in key:
                key = key.replace('_', '.')

            # Aplica override
            keys = key.split('.')
            current = self.config
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value

            logger.info(f"🔧 Override aplicado: {key} = {value}")

    def _validate_dependencies(self) -> bool:
        """Valida se todos os módulos necessários existem"""
        required_modules = [
            'modules.analisador_temporal',
            'modules.detector_dataleak',
            'modules.treino',
            'modules.relatorio'
        ]

        missing_modules = []
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing_modules.append(module)

        if missing_modules:
            logger.error(f"❌ Módulos não encontrados: {missing_modules}")
            return False

        logger.info("✅ Todos os módulos necessários encontrados")
        return True

    def _load_default_config(self) -> Dict[str, Any]:
        return {
            'system': {
                'name': 'Sistema de Predição Tuberculose Infantil',
                'version': '1.0.0',
                'log_level': 'INFO'
            },
            'feature_selection': {
                'scenarios': ['GERAL'],
                'max_features': 20,
                'use_temporal_critical': False,
                'feature_replacement': {
                    'enable_smart_replacement': True,
                    'replacement_strategy': 'replace_originals',
                    'prioritize_derived_features': True
                }
            },
            'model': {
                'use_tabpfn': True,
                'meta_model_type': 'auto',
                'cv_folds': 10,
                'max_features': 20,
                'test_size': 0.2,
                'validation_size': 0.2
            },
            'leakage_detection': {
                'temporal_threshold': 0.10,
                'missing_threshold': 0.20
            },
            'reports': {
                'generate_word': True,
                'generate_technical': True,
                'generate_executive': True,
                'create_visualizations': True
            },
            'pipeline': {
                'stop_on_critical_leakage': False,
                'save_checkpoints': True,
            }
        }

    def setup_environment(self, data_path: str = None, output_dir: str = None) -> bool:
        """
        Configuração inicial do ambiente

        Args:
            data_path: Caminho para arquivo de dados (caso não esteja setado em config.yaml)
            output_dir: Diretório de saída (caso não esteja setado em config.yaml)
        """
        try:
            # Usa caminhos da configuração se não setados
            if not data_path:
                data_path = self.config.get('data', {}).get('input_path')

            if not output_dir:
                output_dir = self.config.get('data', {}).get('output_base_dir')

            # Valida arquivo de dados
            if not data_path:
                logger.error("❌ Caminho dos dados não especificado")
                logger.error("💡 Use --data-path ou configure em config_local.yaml")
                return False

            self.data_path = Path(data_path)
            if not self.data_path.exists():
                logger.error(f"❌ Arquivo de dados não encontrado: {data_path}")
                return False

            # Configura diretório de saída
            if output_dir:
                self.output_dir = Path(output_dir)
            else:
                self.output_dir = self.data_path.parent / "tb_ia_resultados"

            self.output_dir.mkdir(exist_ok=True)

            # Cria subdiretórios
            subdirs = ["reports", "models", "logs", "visualizations", "checkpoints"]
            for subdir in subdirs:
                (self.output_dir / subdir).mkdir(exist_ok=True)

            # Configura log de arquivo
            log_file = self.output_dir / "logs" / f"tb_ia_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

            # Adiciona handler de arquivo ao logger
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            logging.getLogger().addHandler(file_handler)

            logger.info(f"✅ Ambiente configurado:")
            logger.info(f"   📁 Dados: {self.data_path}")
            logger.info(f"   📁 Saída: {self.output_dir}")
            logger.info(f"   📝 Log: {log_file}")

            # Salva configuração usada
            if self.config_manager:
                config_used_path = self.output_dir / "logs" / "config_used.yaml"
                self.config_manager.save_config(str(config_used_path))
                logger.info(f"   ⚙️ Configuração salva: {config_used_path}")

            return True

        except Exception as e:
            logger.error(f"❌ Erro na configuração: {e}")
            return False

    def _get_scenarios_from_config(self) -> List[str]:
        """Obtém cenários configurados"""
        feature_config = self.config.get('feature_selection', {})
        scenarios = feature_config.get('scenarios', ['GERAL'])

        # Valida cenários
        valid_scenarios = ['GERAL', 'MASCULINO', 'FEMININO', 'NEGROS_PARDOS', 'OUTROS_RACA']
        validated_scenarios = [s for s in scenarios if s in valid_scenarios]

        if not validated_scenarios:
            logger.warning("⚠️ Nenhum cenário válido encontrado, usando GERAL")
            validated_scenarios = ['GERAL']

        return validated_scenarios

    def run_feature_selection(self) -> bool:
        """ETAPA 1: Seleção de Features com SUBSTITUIÇÃO INTELIGENTE (SEMPRE OBRIGATÓRIA)"""
        try:
            step_start = time.time()
            logger.info("\n" + "=" * 70)
            logger.info("🔧 ETAPA 1: SELEÇÃO DE FEATURES COM SUBSTITUIÇÃO INTELIGENTE")
            logger.info("=" * 70)

            # Obter cenários da configuração
            scenarios_to_run = self._get_scenarios_from_config()

            logger.info(f"📊 Cenários selecionados: {scenarios_to_run}")
            logger.info(f"📁 Arquivo de dados: {self.data_path}")

            # Log da estratégia de substituição
            replacement_config = self.config.get('feature_selection', {}).get('feature_replacement', {})
            if replacement_config.get('enable_smart_replacement', True):
                logger.info("🔄 ESTRATÉGIA DE SUBSTITUIÇÃO:")
                logger.info(f"   ✅ SCORE_COMORBIDADES → substitui 7 features individuais")
                logger.info(f"   ✅ RISCO_SOCIAL → substitui populações especiais")
                logger.info(f"   ✅ TEMPO_INICIO_CAT → substitui versão contínua")
                logger.info(f"   ✅ Features derivadas priorizadas na seleção")

            # Executar seleção de features com substituições
            self.feature_selector = run_feature_selection_for_main(
                data_path=str(self.data_path),
                config=self.config,
                scenarios=scenarios_to_run
            )

            if not self.feature_selector:
                logger.error("❌ Falha na seleção de features")
                return False

            # Armazenar resultados incluindo informações de substituição
            replacement_summary = self.feature_selector.get_replacement_summary()

            self.results['feature_selection'] = {
                'selector': self.feature_selector,
                'scenarios_processed': self.feature_selector.list_processed_scenarios(),
                'derived_features': self.feature_selector.derived_features,
                'replaced_features': list(self.feature_selector.replaced_features),
                'replacement_mappings': self.feature_selector.replacement_mappings,
                'replacement_summary': replacement_summary,
                'scenarios_configured': scenarios_to_run,
                'config_used': self.config.get('feature_selection', {}),
                'timestamp': datetime.now()
            }

            self.pipeline_status['feature_selection'] = True
            self.step_times['feature_selection'] = time.time() - step_start

            if self.config.get('pipeline', {}).get('save_checkpoints', True):
                self.save_checkpoint('feature_selection')

            # Salvar resumo da seleção COM INFORMAÇÕES DE SUBSTITUIÇÃO
            summary_path = self.output_dir / "reports" / "01_feature_selection_summary.txt"
            self.feature_selector.save_results_summary(str(summary_path))

            # Mostrar resumo com foco nas substituições
            processed_scenarios = self.feature_selector.list_processed_scenarios()
            logger.info(f"✅ Seleção concluída para {len(processed_scenarios)} cenários")
            logger.info(f"🔄 Features derivadas criadas: {len(self.feature_selector.derived_features)}")
            logger.info(f"🔄 Features originais substituídas: {len(self.feature_selector.replaced_features)}")

            # Resumo de efetividade das substituições
            total_derived_used = 0
            total_replaced_still_used = 0

            for scenario in processed_scenarios:
                data, features = self.feature_selector.get_data_for_scenario(scenario)
                derived_count = len([f for f in features if f in self.feature_selector.derived_features])
                replaced_count = len([f for f in features if f in self.feature_selector.replaced_features])

                total_derived_used += derived_count
                total_replaced_still_used += replaced_count

                logger.info(f"   🎯 {scenario}: {len(features)} features ({derived_count} derivadas), {len(data)} registros")
                if replaced_count > 0:
                    logger.warning(f"      ⚠️ {replaced_count} features substituídas ainda em uso")
                else:
                    logger.info(f"      ✅ Substituições efetivas")

            # Análise geral de efetividade
            if total_replaced_still_used == 0:
                logger.info(f"   ✅ SUBSTITUIÇÕES 100% EFETIVAS - features originais não estão sendo usadas")
            else:
                logger.warning(f"   ⚠️ {total_replaced_still_used} features substituídas ainda em uso - revisar configuração")

            # Log das substituições realizadas
            if replacement_summary['replacement_enabled']:
                logger.info(f"\n🔄 RESUMO DAS SUBSTITUIÇÕES APLICADAS:")
                logger.info(f"   Features derivadas: {replacement_summary['derived_features_created']}")
                if replacement_summary['replaced_features']:
                    logger.info(f"   Features substituídas: {replacement_summary['replaced_features']}")
                logger.info(f"   Estratégia: {replacement_summary['replacement_strategy']}")

            return True

        except Exception as e:
            logger.error(f"❌ Erro na seleção de features: {e}")
            logger.error(traceback.format_exc())
            if self.config.get('pipeline', {}).get('stop_on_failure', False):
                raise
            return False

    def run_clinical_timeline_analysis(self) -> bool:
        """
        ETAPA 2: Análise da Linha Temporal Clínica das Features Selecionadas E DERIVADAS
        Identifica quando cada informação está disponível, considerando substituições
        """
        try:
            step_start = time.time()
            logger.info("\n" + "=" * 70)
            logger.info("🕐 ETAPA 2: ANÁLISE TEMPORAL COM FEATURES DERIVADAS")
            logger.info("=" * 70)

            if not self.feature_selector:
                logger.error("❌ Seleção de features não executada!")
                return False

            from modules.analisador_temporal import ClinicalTimelineAnalyzer

            # Inicializar analisador com informações de substituição
            analyzer = ClinicalTimelineAnalyzer(feature_selector=self.feature_selector)

            # Executa análise temporal base
            timeline_results = analyzer.analyze_clinical_timeline()
            feature_categories = analyzer.create_clinical_feature_categories()

            # Análise específica das features selecionadas por cenário
            selected_features_analysis = {}
            processed_scenarios = self.feature_selector.list_processed_scenarios()

            logger.info(f"🔄 Analisando features derivadas criadas:")
            for derived_feature in self.feature_selector.derived_features:
                logger.info(f"   ✅ {derived_feature} (considerada temporalmente segura)")

            for scenario in processed_scenarios:
                data, features = self.feature_selector.get_data_for_scenario(scenario)

                logger.info(f"🎯 Analisando features do cenário {scenario}")

                # Avalia cada feature selecionada (incluindo derivadas)
                features_risk = {}
                for feature in features:
                    risk_assessment = analyzer.get_feature_risk_assessment(feature)
                    features_risk[feature] = risk_assessment

                # Validação das features nos dados
                validation = analyzer.validate_features_in_data(list(data.columns))

                # Análise específica de substituições
                derived_in_scenario = [f for f in features if f in self.feature_selector.derived_features]
                replaced_in_scenario = [f for f in features if f in self.feature_selector.replaced_features]

                selected_features_analysis[scenario] = {
                    'features_used': features,
                    'features_risk': features_risk,
                    'data_validation': validation,
                    'total_features': len(features),
                    'safe_features': [f for f, r in features_risk.items() if r['risk_level'] == 'BAIXO'],
                    'risky_features': [f for f, r in features_risk.items() if r['risk_level'] in ['ALTO', 'CRÍTICO']],
                    'derived_features_used': derived_in_scenario,
                    'replaced_features_still_used': replaced_in_scenario,
                    'replacement_effectiveness': len(replaced_in_scenario) == 0  # True se nenhuma substituída está sendo usada
                }

                safe_count = len(selected_features_analysis[scenario]['safe_features'])
                derived_count = len(derived_in_scenario)
                replaced_count = len(replaced_in_scenario)

                logger.info(f"   ✅ Features seguras: {safe_count}/{len(features)}")
                logger.info(f"   🔄 Features derivadas: {derived_count}/{len(features)}")
                if replaced_count > 0:
                    logger.warning(f"   ⚠️ Features substituídas ainda em uso: {replaced_count}")
                else:
                    logger.info(f"   ✅ Substituições efetivas: originais não estão sendo usadas")

            # Salva resultados
            self.results['clinical_analysis'] = {
                'timeline_results': timeline_results,
                'feature_categories': feature_categories,
                'temporal_features': analyzer.get_temporal_suspicious_features(),
                'safe_features': analyzer.get_safe_features(),
                'derived_features': analyzer.get_derived_features(),
                'replaced_features': analyzer.get_replaced_features(),
                'selected_features_analysis': selected_features_analysis,
                'scenarios_analyzed': processed_scenarios,
                'replacement_analysis': analyzer.replacement_analysis if hasattr(analyzer, 'replacement_analysis') else {},
                'timestamp': datetime.now()
            }

            # Salva relatório da análise temporal
            timeline_report_path = self.output_dir / "reports" / "02_clinical_timeline_analysis.txt"
            analyzer.save_timeline_report(timeline_report_path)

            # Salva análise específica das features selecionadas
            self._save_selected_features_temporal_analysis(selected_features_analysis)

            self.pipeline_status['clinical_analysis'] = True
            self.step_times['clinical_analysis'] = time.time() - step_start

            # Checkpoint
            if self.config.get('pipeline', {}).get('save_checkpoints', True):
                self.save_checkpoint('clinical_analysis')

            logger.info("✅ Análise temporal das features selecionadas E DERIVADAS concluída")
            return True

        except Exception as e:
            logger.error(f"❌ Erro na análise temporal: {e}")
            logger.error(traceback.format_exc())
            if self.config.get('pipeline', {}).get('stop_on_failure', False):
                raise
            return False

    def _save_selected_features_temporal_analysis(self, analysis: Dict[str, Any]):
        """Salva análise temporal específica das features selecionadas COM SUBSTITUIÇÕES"""
        try:
            report_path = self.output_dir / "reports" / "02_selected_features_temporal_analysis.txt"

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("ANÁLISE TEMPORAL DAS FEATURES SELECIONADAS COM SUBSTITUIÇÕES\n")
                f.write("=" * 80 + "\n\n")

                for scenario, data in analysis.items():
                    f.write(f"CENÁRIO: {scenario}\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"Total de features: {data['total_features']}\n")
                    f.write(f"Features seguras: {len(data['safe_features'])}\n")
                    f.write(f"Features arriscadas: {len(data['risky_features'])}\n")

                    # Informações de substituição
                    derived_count = len(data.get('derived_features_used', []))
                    replaced_count = len(data.get('replaced_features_still_used', []))
                    f.write(f"Features derivadas utilizadas: {derived_count}\n")
                    f.write(f"Features substituídas ainda em uso: {replaced_count}\n")
                    f.write(f"Efetividade das substituições: {'✅ SIM' if data.get('replacement_effectiveness', False) else '⚠️ PARCIAL'}\n\n")

                    f.write("FEATURES SEGURAS (Temporalmente):\n")
                    for feature in data['safe_features']:
                        marker = "🔄" if feature in data.get('derived_features_used', []) else "✅"
                        suffix = " (DERIVADA)" if feature in data.get('derived_features_used', []) else ""
                        f.write(f"  {marker} {feature}{suffix}\n")

                    f.write("\nFEATURES ARRISCADAS (Temporalmente):\n")
                    for feature in data['risky_features']:
                        risk_info = data['features_risk'][feature]
                        marker = "⚠️"
                        if feature in data.get('replaced_features_still_used', []):
                            marker = "🚨"
                            suffix = " (SUBSTITUÍDA - REVISAR)"
                        else:
                            suffix = ""
                        f.write(f"  {marker} {feature}{suffix} - Risco: {risk_info['risk_level']}\n")
                        f.write(f"     Categoria: {risk_info['category']}\n")
                        f.write(f"     Momento: {risk_info['moment']}\n")

                    # Análise de substituições por cenário
                    if data.get('derived_features_used'):
                        f.write(f"\nFEATURES DERIVADAS UTILIZADAS:\n")
                        for feature in data['derived_features_used']:
                            f.write(f"  🔄 {feature}\n")

                    if data.get('replaced_features_still_used'):
                        f.write(f"\nFEATURES SUBSTITUÍDAS AINDA EM USO (Revisar):\n")
                        for feature in data['replaced_features_still_used']:
                            f.write(f"  🚨 {feature} - considere remover da seleção\n")

                    f.write("\n" + "=" * 60 + "\n\n")

            logger.info(f"📄 Análise temporal das features COM SUBSTITUIÇÕES salva: {report_path}")

        except Exception as e:
            logger.warning(f"⚠️ Erro ao salvar análise temporal das features: {e}")

    def run_leakage_detection(self) -> bool:
        """
        ETAPA 3: Detecção de Data Leakage nas Features Selecionadas COM DERIVADAS
        Identifica vazamentos óbvios nos dados considerando substituições
        """
        try:
            step_start = time.time()
            logger.info("\n" + "=" * 70)
            logger.info("🚨 ETAPA 3: DETECÇÃO DE DATA LEAKAGE COM FEATURES DERIVADAS")
            logger.info("=" * 70)

            if not self.feature_selector:
                logger.error("❌ Seleção de features não executada!")
                return False

            from modules.detector_dataleak import DataLeakageDetector

            # Log sobre tratamento de features derivadas
            logger.info("🔄 Features derivadas são consideradas temporalmente SEGURAS por construção")
            logger.info(f"   Features derivadas: {self.feature_selector.derived_features}")

            # Análise de leakage por cenário
            leakage_results_by_scenario = {}
            processed_scenarios = self.feature_selector.list_processed_scenarios()

            for scenario in processed_scenarios:
                logger.info(f"\n🎯 Analisando leakage no cenário: {scenario}")

                # Obter dados e features do cenário
                data, features = self.feature_selector.get_data_for_scenario(scenario)

                if len(data) < 50:
                    logger.warning(f"⚠️ Poucos dados para análise de {scenario}")
                    continue

                # Criar detector para este cenário
                detector = DataLeakageDetector()

                # Salvar dados temporários (só com features selecionadas)
                temp_dir = self.output_dir / "temp"
                temp_dir.mkdir(exist_ok=True)
                temp_data_path = temp_dir / f"leakage_data_{scenario}.csv"

                # Manter apenas features selecionadas + target
                columns_to_keep = features + ['SITUA_ENCE'] if 'SITUA_ENCE' in data.columns else features
                columns_available = [col for col in columns_to_keep if col in data.columns]

                data_for_leakage = data[columns_available]
                data_for_leakage.to_csv(temp_data_path, index=False)

                # Carregar dados no detector
                if not detector.load_and_preprocess(str(temp_data_path)):
                    logger.error(f"❌ Falha ao carregar dados para detecção de leakage - {scenario}")
                    continue

                # Executa detecção específica para as features selecionadas
                leakage_results = detector.generate_leakage_report()

                # Análise específica das features selecionadas considerando derivadas
                features_leakage_analysis = {}
                derived_count = 0
                for feature in features:
                    if feature in self.feature_selector.derived_features:
                        # Features derivadas são consideradas seguras
                        features_leakage_analysis[feature] = {
                            'status': 'SEGURA',
                            'reason': 'Feature derivada - construída com dados do momento inicial'
                        }
                        derived_count += 1
                    elif feature in detector.suspicious_features:
                        features_leakage_analysis[feature] = {
                            'status': 'SUSPEITA',
                            'reason': 'Identificada como temporalmente suspeita'
                        }
                    else:
                        features_leakage_analysis[feature] = {
                            'status': 'SEGURA',
                            'reason': 'Não identificada como suspeita'
                        }

                # Contar apenas features originais suspeitas (não derivadas)
                original_suspicious = [f for f in features if f in detector.suspicious_features and f not in self.feature_selector.derived_features]

                leakage_results_by_scenario[scenario] = {
                    'general_results': leakage_results,
                    'features_analysis': features_leakage_analysis,
                    'risk_level': leakage_results.get('risk_level', 'UNKNOWN'),
                    'suspicious_count': len(original_suspicious),  # Só conta originais suspeitas
                    'total_features': len(features),
                    'derived_features_count': derived_count,
                    'derived_features_safe': derived_count  # Todas derivadas são consideradas seguras
                }

                # Limpar arquivo temporário
                temp_data_path.unlink(missing_ok=True)

                suspicious_count = leakage_results_by_scenario[scenario]['suspicious_count']
                total_count = leakage_results_by_scenario[scenario]['total_features']
                risk_level = leakage_results_by_scenario[scenario]['risk_level']

                logger.info(f"   📊 Features suspeitas (originais): {suspicious_count}/{total_count}")
                logger.info(f"   🔄 Features derivadas (seguras): {derived_count}/{total_count}")
                logger.info(f"   🚨 Risco geral: {risk_level}")

            # Determinar risco geral do sistema
            overall_risk = self._determine_overall_leakage_risk(leakage_results_by_scenario)

            # Salva resultados
            self.results['leakage_detection'] = {
                'results_by_scenario': leakage_results_by_scenario,
                'overall_risk_level': overall_risk,
                'scenarios_analyzed': list(leakage_results_by_scenario.keys()),
                'derived_features_treatment': 'considered_safe',
                'timestamp': datetime.now()
            }

            # Salva relatório de leakage
            leakage_report_path = self.output_dir / "reports" / "03_leakage_detection_report.txt"
            self._save_integrated_leakage_report(leakage_report_path, leakage_results_by_scenario, overall_risk)

            self.pipeline_status['leakage_detection'] = True
            self.step_times['leakage_detection'] = time.time() - step_start

            # Checkpoint
            if self.config.get('pipeline', {}).get('save_checkpoints', True):
                self.save_checkpoint('leakage_detection')

            # Verifica se pode continuar
            stop_on_critical = self.config.get('pipeline', {}).get('stop_on_critical_leakage', False)

            if overall_risk == 'CRITICAL' and stop_on_critical:
                logger.error("🚨 RISCO CRÍTICO detectado - Pipeline interrompido conforme configuração")
                return False
            elif overall_risk == 'HIGH':
                logger.warning("⚠️ RISCO ALTO de data leakage detectado - prosseguindo com cautela")
            else:
                logger.info(f"✅ Detecção de leakage concluída - Risco: {overall_risk}")
                logger.info(f"🔄 Features derivadas reduzem risco temporal significativamente")

            return True

        except Exception as e:
            logger.error(f"❌ Erro na detecção de leakage: {e}")
            logger.error(traceback.format_exc())
            if self.config.get('pipeline', {}).get('stop_on_failure', False):
                raise
            return False

    def _determine_overall_leakage_risk(self, results_by_scenario: Dict[str, Any]) -> str:
        """Determina risco geral baseado em todos os cenários"""
        risk_levels = []

        for scenario, results in results_by_scenario.items():
            risk_levels.append(results['risk_level'])

        if 'CRITICAL' in risk_levels:
            return 'CRITICAL'
        elif 'HIGH' in risk_levels:
            return 'HIGH'
        elif 'MODERATE' in risk_levels:
            return 'MODERATE'
        else:
            return 'LOW'

    def _save_integrated_leakage_report(self, report_path: Path, results_by_scenario: Dict[str, Any],
                                        overall_risk: str):
        """Salva relatório integrado de leakage COM INFORMAÇÕES DE SUBSTITUIÇÃO"""
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("RELATÓRIO DE DETECÇÃO DE DATA LEAKAGE COM FEATURES DERIVADAS\n")
                f.write("Análise das Features Selecionadas por Cenário\n")
                f.write("=" * 80 + "\n\n")

                f.write(f"RISCO GERAL DO SISTEMA: {overall_risk}\n")
                f.write(f"TRATAMENTO DE FEATURES DERIVADAS: Consideradas temporalmente seguras\n\n")

                for scenario, results in results_by_scenario.items():
                    f.write(f"CENÁRIO: {scenario}\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"Risco: {results['risk_level']}\n")
                    f.write(f"Features analisadas: {results['total_features']}\n")
                    f.write(f"Features suspeitas (originais): {results['suspicious_count']}\n")
                    f.write(f"Features derivadas (seguras): {results.get('derived_features_count', 0)}\n\n")

                    f.write("ANÁLISE POR FEATURE:\n")
                    for feature, analysis in results['features_analysis'].items():
                        if analysis['status'] == 'SUSPEITA':
                            status_icon = "⚠️"
                        elif 'derivada' in analysis['reason'].lower():
                            status_icon = "🔄"
                        else:
                            status_icon = "✅"

                        f.write(f"  {status_icon} {feature}: {analysis['status']}\n")
                        f.write(f"     Motivo: {analysis['reason']}\n")

                    f.write("\n" + "=" * 60 + "\n\n")

            logger.info(f"📄 Relatório de leakage COM SUBSTITUIÇÕES salvo: {report_path}")

        except Exception as e:
            logger.warning(f"⚠️ Erro ao salvar relatório de leakage: {e}")

    def run_model_training(self) -> bool:
        """ETAPA 4: Treinamento usando Features Selecionadas COM OTIMIZAÇÃO PARA DERIVADAS"""
        try:
            step_start = time.time()
            logger.info("\n" + "=" * 70)
            logger.info("🤖 ETAPA 4: TREINAMENTO OTIMIZADO PARA FEATURES DERIVADAS")
            logger.info("=" * 70)

            if not self.feature_selector:
                logger.error("❌ Seleção de features não executada!")
                return False

            from modules.treino import TuberculosisPredictor

            processed_scenarios = self.feature_selector.list_processed_scenarios()
            training_results = {}

            logger.info("🔄 Treinamento otimizado para features categóricas derivadas")

            # Treinar modelo para cada cenário processado
            for scenario in processed_scenarios:
                logger.info(f"\n🎯 Treinando modelo: {scenario}")

                # Obter dados e features do cenário
                data, features = self.feature_selector.get_data_for_scenario(scenario)

                if len(data) < 100:
                    logger.warning(f"⚠️ Poucos dados para {scenario}: {len(data)}")
                    continue

                if len(features) == 0:
                    logger.warning(f"⚠️ Nenhuma feature para {scenario}")
                    continue

                # Análise das features
                derived_count = len([f for f in features if f in self.feature_selector.derived_features])
                original_count = len(features) - derived_count

                # Verificar distribuição do target
                target_dist = data['SITUA_ENCE'].value_counts()
                logger.info(f"   📊 Dados: {len(data)} registros")
                logger.info(f"   🎯 Target: {dict(target_dist)}")
                logger.info(f"   🔧 Features totais: {len(features)}")
                logger.info(f"   🔄 Features derivadas: {derived_count}")
                logger.info(f"   📊 Features originais: {original_count}")

                # Treinar modelo individual com otimização para derivadas
                model_result = self._train_single_model(data, features, scenario)

                if model_result:
                    training_results[scenario] = model_result
                    logger.info(f"   ✅ Modelo {scenario} treinado com sucesso")

                    # Log da performance
                    if 'predictor' in model_result and hasattr(model_result['predictor'], 'results'):
                        metrics = model_result['predictor'].results
                        acc = metrics.get('balanced_accuracy', 0)
                        f1 = metrics.get('f1_score', 0)
                        logger.info(f"   📈 Acurácia: {acc:.3f}, F1: {f1:.3f}")

                        # Log sobre features derivadas utilizadas
                        derived_used = len([f for f in model_result['features_used'] if f in self.feature_selector.derived_features])
                        logger.info(f"   🔄 Features derivadas no modelo: {derived_used}")
                else:
                    logger.error(f"   ❌ Falha no treinamento de {scenario}")

            if not training_results:
                logger.error("❌ Nenhum modelo treinado com sucesso")
                return False

            # Encontrar melhor modelo
            best_scenario = None
            best_score = 0
            best_predictor = None

            for scenario, result in training_results.items():
                if 'predictor' in result and hasattr(result['predictor'], 'results'):
                    acc = result['predictor'].results.get('balanced_accuracy', 0)
                    if acc > best_score:
                        best_score = acc
                        best_scenario = scenario
                        best_predictor = result['predictor']

            # Armazenar resultados em formato compatível
            self.results['model_training'] = {
                'scenarios': training_results,
                'best_scenario': best_scenario,
                'config_used': self.config.get('model', {}),
                'derived_features_optimization': True,
                'timestamp': datetime.now()
            }

            # Adiciona dados do melhor modelo no formato esperado pelo relatório
            if best_predictor:
                self.results['model_training']['predictor'] = best_predictor
                self.results['model_training']['results'] = best_predictor.results
                self.results['model_training']['best_meta_name'] = getattr(best_predictor, 'best_meta_name', 'N/A')

            self.pipeline_status['model_training'] = True
            self.step_times['model_training'] = time.time() - step_start

            # Salvar modelos
            self._save_trained_models(training_results)

            # Análise comparativa com foco em substituições
            self._analyze_training_results(training_results)

            if best_predictor:
                from parametros import extract_from_trained_predictor
                logger.info("📋 Extraindo parâmetros reais do modelo treinado...")

                # Extrai parâmetros do melhor modelo com informações do treinamento
                params_file = extract_from_trained_predictor(
                    best_predictor,
                    str(self.output_dir / "parametros_treino_executado.txt")
                )

                # Adiciona informações de configuração e resultados do pipeline
                from parametros import extract_from_pipeline_results
                extract_from_pipeline_results(
                    self.results,
                    self.config,
                    training_results,
                    str(self.output_dir / "parametros_pipeline_completo.txt")
                )

                logger.info(f"✅ Parâmetros do treino salvos em: {params_file}")

            if self.config.get('pipeline', {}).get('save_checkpoints', True):
                self.save_checkpoint('model_training')

            logger.info("✅ Treinamento otimizado para features derivadas concluído")
            return True

        except Exception as e:
            logger.error(f"❌ Erro no treinamento: {e}")
            logger.error(traceback.format_exc())
            if self.config.get('pipeline', {}).get('stop_on_failure', False):
                raise
            return False

    def _train_single_model(self, data: Any, features: List[str], scenario: str) -> Optional[Dict[str, Any]]:
        """Treina um modelo individual para um cenário COM FEATURES DERIVADAS"""
        try:
            from modules.treino import TuberculosisPredictor

            # Criar diretório temporário
            temp_dir = self.output_dir / "temp"
            temp_dir.mkdir(exist_ok=True)

            # Salvar dados temporários para o preditor
            temp_data_path = temp_dir / f"data_{scenario}.csv"
            data.to_csv(temp_data_path, index=False)

            # Configurar preditor COM SUPORTE A FEATURES DERIVADAS
            model_config = self.config.get('model', {})
            predictor = TuberculosisPredictor(
                use_tabpfn=model_config.get('use_tabpfn', True),
                meta_model_type=model_config.get('meta_model_selection', 'auto'),
                cv_folds=model_config.get('cv_folds', 10),
                feature_selector=self.feature_selector,  # Passa o feature selector
                config=self.config  # Passa toda a configuração
            )

            predictor.feature_selector = ensure_feature_selector_compatibility(predictor.feature_selector)

            # Carregar e processar dados
            if not predictor.load_data(str(temp_data_path)):
                logger.error(f"❌ Falha ao carregar dados temporários para {scenario}")
                return None

            # Preprocessar com features selecionadas
            if not predictor.preprocess_data():
                logger.error(f"❌ Falha no pré-processamento para {scenario}")
                return None

            # Filtrar features selecionadas
            available_features = [f for f in features if f in predictor.X.columns]
            if len(available_features) < 3:
                logger.warning(f"⚠️ Muito poucas features disponíveis para {scenario}: {len(available_features)}")
                return None

            # Ajustar dados para features selecionadas
            predictor.X = predictor.X[available_features]
            predictor.feature_names = available_features
            predictor.selected_features = available_features

            # Treinar modelo
            if not predictor.train_model():
                logger.error(f"❌ Falha no treinamento para {scenario}")
                return None

            # Limpar arquivo temporário
            temp_data_path.unlink(missing_ok=True)

            return {
                'predictor': predictor,
                'scenario': scenario,
                'features_used': available_features,
                'metrics': predictor.results,
                'cv_results': getattr(predictor, 'cv_results', {}),
                'best_meta_name': getattr(predictor, 'best_meta_name', 'N/A'),
                'derived_features_used': [f for f in available_features if f in self.feature_selector.derived_features]
            }

        except Exception as e:
            logger.error(f"❌ Erro no treinamento individual de {scenario}: {e}")
            return None

    def _save_trained_models(self, training_results: Dict[str, Any]):
        """Salva modelos treinados"""
        try:
            models_dir = self.output_dir / "models"
            models_dir.mkdir(exist_ok=True)

            for scenario, result in training_results.items():
                if 'predictor' in result:
                    model_path = models_dir / f"model_{scenario}.pkl"
                    result['predictor'].save_model(str(model_path))
                    logger.info(f"💾 Modelo {scenario} salvo: {model_path.name}")

        except Exception as e:
            logger.warning(f"⚠️ Erro ao salvar modelos: {e}")

    def _analyze_training_results(self, training_results: Dict[str, Any]):
        """Análise comparativa dos resultados COM FOCO EM SUBSTITUIÇÕES"""
        logger.info(f"\n📊 ANÁLISE COMPARATIVA DOS MODELOS COM FEATURES DERIVADAS:")
        logger.info("-" * 50)

        for scenario, result in training_results.items():
            if 'predictor' in result and hasattr(result['predictor'], 'results'):
                metrics = result['predictor'].results
                derived_used = result.get('derived_features_used', [])

                logger.info(f"\n🎯 {scenario}:")
                logger.info(f"   📈 Acurácia Balanceada: {metrics.get('balanced_accuracy', 0):.3f}")
                logger.info(f"   🎯 F1-Score: {metrics.get('f1_score', 0):.3f}")
                logger.info(f"   📊 AUC: {metrics.get('auc_score', 0):.3f}")
                logger.info(f"   🏅 Meta-modelo: {result.get('best_meta_name', 'N/A')}")
                logger.info(f"   🔧 Features totais: {len(result.get('features_used', []))}")
                logger.info(f"   🔄 Features derivadas: {len(derived_used)}")
                if derived_used:
                    logger.info(f"      Derivadas usadas: {derived_used}")

        # Encontrar melhor modelo
        best_scenario = None
        best_score = 0
        for scenario, result in training_results.items():
            if 'predictor' in result and hasattr(result['predictor'], 'results'):
                score = result['predictor'].results.get('balanced_accuracy', 0)
                if score > best_score:
                    best_score = score
                    best_scenario = scenario

        if best_scenario:
            logger.info(f"\n🏆 MELHOR MODELO: {best_scenario} (Acurácia: {best_score:.3f})")
            best_result = training_results[best_scenario]
            best_derived = best_result.get('derived_features_used', [])
            if best_derived:
                logger.info(f"   🔄 Features derivadas no melhor modelo: {best_derived}")

    def run_report_generation(self) -> bool:
        """
        ETAPA 5: Geração de Relatórios Completos COM ANÁLISE DE SUBSTITUIÇÕES
        Consolida todos os resultados em relatórios detalhados
        """
        try:
            step_start = time.time()
            logger.info("\n" + "="*70)
            logger.info("📄 ETAPA 5: GERAÇÃO DE RELATÓRIOS COM ANÁLISE DE SUBSTITUIÇÕES")
            logger.info("="*70)

            from modules.relatorio import ComprehensiveReportGenerator

            generator = ComprehensiveReportGenerator(
                results=self.results,
                output_dir=self.output_dir,
                config=self.config
            )

            reports_config = self.config.get('reports', {})
            generated_files = []

            # Log sobre relatórios com substituições
            logger.info("📋 Relatórios incluirão análise detalhada das substituições de features")

            # Gera relatório Word completo
            if reports_config.get('generate_word', True):
                word_report_path = generator.generate_word_report()
                if word_report_path:
                    generated_files.append(word_report_path)
                    logger.info(f"📄 Relatório Word: {Path(word_report_path).name}")

            # Gera relatório técnico
            if reports_config.get('generate_technical', True):
                technical_report_path = generator.generate_technical_report()
                if technical_report_path:
                    generated_files.append(technical_report_path)
                    logger.info(f"🔧 Relatório Técnico: {Path(technical_report_path).name}")

            # Gera visualizações
            if reports_config.get('create_visualizations', True):
                viz_paths = generator.generate_visualizations()
                generated_files.extend(viz_paths)
                logger.info(f"📊 Visualizações: {len(viz_paths)} arquivos gerados")

            # Gera resumo
            if reports_config.get('generate_executive', True):
                executive_summary_path = generator.generate_executive_summary()
                if executive_summary_path:
                    generated_files.append(executive_summary_path)
                    logger.info(f"📋 Resumo : {Path(executive_summary_path).name}")

            # Gera export JSON
            if reports_config.get('generate_json_export', True):
                json_export_path = generator.export_summary_json()
                if json_export_path:
                    generated_files.append(json_export_path)
                    logger.info(f"📊 Export JSON: {Path(json_export_path).name}")

            self.results['report_generation'] = {
                'generated_files': generated_files,
                'reports_config': reports_config,
                'includes_substitution_analysis': True,
                'timestamp': datetime.now()
            }

            self.pipeline_status['report_generation'] = True
            self.step_times['report_generation'] = time.time() - step_start

            # Checkpoint final
            if self.config.get('pipeline', {}).get('save_checkpoints', True):
                self.save_checkpoint('report_generation')

            logger.info("✅ Geração de relatórios com análise de substituições concluída")

            return True

        except Exception as e:
            logger.error(f"❌ Erro na geração de relatórios: {e}")
            logger.error(traceback.format_exc())
            if self.config.get('pipeline', {}).get('stop_on_failure', False):
                raise
            return False

    def run_complete_pipeline(self, data_path: str = None, output_dir: str = None) -> bool:
        """
        Executa o pipeline completo do sistema de IA COM VALIDAÇÃO ESTATÍSTICA

        Args:
            data_path: Caminho para arquivo de dados
            output_dir: Diretório de saída
        """
        try:
            self.start_time = time.time()

            logger.info("\n" + "=" * 80)
            logger.info("🚀 INICIANDO PIPELINE COMPLETO - SISTEMA IA TUBERCULOSE")
            logger.info("🔄 COM SUBSTITUIÇÃO INTELIGENTE + VALIDAÇÃO ESTATÍSTICA")
            logger.info("=" * 80)

            # Configura o ambiente
            if not self.setup_environment(data_path, output_dir):
                return False

            # NOVA ORDEM: Adiciona validação estatística
            pipeline_steps = [
                ("🔧 Seleção de Features com Substituição Inteligente", self.run_feature_selection),
                ("🕐 Análise Temporal das Features Derivadas", self.run_clinical_timeline_analysis),
                ("🚨 Detecção de Data Leakage com Features Derivadas", self.run_leakage_detection),
                ("🤖 Treinamento Otimizado para Features Derivadas", self.run_model_training),
                ("🧪 Validação Estatística e Testes de Hipótese", self.run_statistical_validation),  # NOVO
                ("📄 Geração de Relatórios com Análise Completa", self.run_report_generation)
            ]

            for step_name, step_function in pipeline_steps:
                logger.info(f"\n▶️ Executando: {step_name}")

                step_start = time.time()
                success = step_function()
                step_time = time.time() - step_start

                if success:
                    logger.info(f"✅ {step_name} concluída em {step_time:.2f}s")
                else:
                    logger.error(f"❌ Falha em: {step_name}")

                    # Verifica se deve continuar
                    if self.config.get('pipeline', {}).get('stop_on_failure', False):
                        return False
                    else:
                        logger.warning("⚠️ Continuando pipeline apesar da falha...")

            total_time = time.time() - self.start_time

            # Relatório final
            self._print_pipeline_summary(total_time)

            # Notificações
            self._send_notifications(success=True)

            return True

        except Exception as e:
            logger.error(f"❌ Erro no pipeline: {e}")
            logger.error(traceback.format_exc())

            # Notificações de erro
            self._send_notifications(success=False, error=str(e))

            return False

    def _print_pipeline_summary(self, total_time: float):
        """Imprime resumo final do pipeline COM SUBSTITUIÇÕES"""
        logger.info("\n" + "=" * 80)
        logger.info("🎉 PIPELINE CONCLUÍDO!")
        logger.info("=" * 80)

        logger.info(f"⏱️ Tempo total: {total_time:.2f}s ({total_time / 60:.1f}min)")
        logger.info(f"📁 Resultados salvos em: {self.output_dir}")

        # Status das etapas
        logger.info("\n📋 STATUS DAS ETAPAS:")
        status_map = {
            'feature_selection': '🔧 Seleção de Features com Substituição',
            'clinical_analysis': '🕐 Análise Temporal com Derivadas',
            'leakage_detection': '🚨 Detecção de Data Leakage',
            'model_training': '🤖 Treinamento Otimizado',
            'statistical_validation': '🧪 Validação Estatística',
            'report_generation': '📄 Geração de Relatórios'
        }

        for key, name in status_map.items():
            status = "✅ CONCLUÍDA" if self.pipeline_status[key] else "❌ FALHOU"
            step_time = self.step_times.get(key, 0)
            logger.info(f"   {name}: {status} ({step_time:.1f}s)")

        # Resumo da seleção de features COM SUBSTITUIÇÕES
        if 'feature_selection' in self.results and self.feature_selector:
            processed_scenarios = self.feature_selector.list_processed_scenarios()
            logger.info(f"\n🔧 RESUMO DA SELEÇÃO DE FEATURES COM SUBSTITUIÇÕES:")
            logger.info(f"   Cenários processados: {len(processed_scenarios)}")
            logger.info(f"   Features derivadas criadas: {len(self.feature_selector.derived_features)}")
            logger.info(f"   Features originais substituídas: {len(self.feature_selector.replaced_features)}")

            # Resumo de efetividade das substituições
            total_derived_used = 0
            total_replaced_still_used = 0

            for scenario in processed_scenarios:
                data, features = self.feature_selector.get_data_for_scenario(scenario)
                derived_count = len([f for f in features if f in self.feature_selector.derived_features])
                replaced_count = len([f for f in features if f in self.feature_selector.replaced_features])

                total_derived_used += derived_count
                total_replaced_still_used += replaced_count

                logger.info(f"   🎯 {scenario}: {len(features)} features ({derived_count} derivadas), {len(data)} registros")
                if replaced_count > 0:
                    logger.warning(f"      ⚠️ {replaced_count} features substituídas ainda em uso")
                else:
                    logger.info(f"      ✅ Substituições efetivas")

            # Análise geral de efetividade
            if total_replaced_still_used == 0:
                logger.info(f"   ✅ SUBSTITUIÇÕES 100% EFETIVAS - features originais não estão sendo usadas")
            else:
                logger.warning(f"   ⚠️ {total_replaced_still_used} features substituídas ainda em uso - revisar configuração")

        # Resumo da análise temporal
        if 'clinical_analysis' in self.results:
            clinical_data = self.results['clinical_analysis']
            if 'selected_features_analysis' in clinical_data:
                logger.info(f"\n🕐 RESUMO DA ANÁLISE TEMPORAL:")
                for scenario, analysis in clinical_data['selected_features_analysis'].items():
                    safe_count = len(analysis['safe_features'])
                    total_count = analysis['total_features']
                    derived_count = len(analysis.get('derived_features_used', []))
                    logger.info(f"   🎯 {scenario}: {safe_count}/{total_count} features seguras ({derived_count} derivadas)")

        # Resumo da detecção de leakage
        if 'leakage_detection' in self.results:
            leakage_data = self.results['leakage_detection']
            overall_risk = leakage_data.get('overall_risk_level', 'UNKNOWN')
            logger.info(f"\n🚨 RESUMO DA DETECÇÃO DE LEAKAGE:")
            logger.info(f"   Risco geral: {overall_risk}")
            logger.info(f"   🔄 Features derivadas tratadas como seguras")

            if 'results_by_scenario' in leakage_data:
                for scenario, results in leakage_data['results_by_scenario'].items():
                    suspicious_count = results['suspicious_count']
                    total_count = results['total_features']
                    derived_safe = results.get('derived_features_safe', 0)
                    logger.info(f"   🎯 {scenario}: {suspicious_count}/{total_count} features suspeitas (originais), {derived_safe} derivadas seguras")

        if 'statistical_validation' in self.results:
            validation_data = self.results['statistical_validation']
            overall_assessment = validation_data.get('overall_assessment', {})

            logger.info(f"\n🧪 RESUMO DA VALIDAÇÃO ESTATÍSTICA:")
            logger.info(f"   Status: {overall_assessment.get('overall_status', 'N/A')}")
            logger.info(f"   Taxa de significância: {overall_assessment.get('significance_rate', 0):.1%}")
            logger.info(f"   Recomendação: {overall_assessment.get('recommendation', 'N/A')}")

            # Mostrar testes individuais
            for assessment in overall_assessment.get('test_assessments', []):
                status_icon = "✅" if assessment['significant'] else "⚠️"
                logger.info(f"      {status_icon} {assessment['test']}")

        # Resultados do modelo por cenário
        if 'model_training' in self.results:
            training_data = self.results['model_training']

            # Verificar estrutura correta
            if 'scenarios' in training_data:
                logger.info(f"\n🏆 RESULTADOS DOS MODELOS POR CENÁRIO:")

                best_scenario = None
                best_score = 0

                for scenario, result in training_data['scenarios'].items():
                    # Acessar métricas corretamente
                    if 'predictor' in result and hasattr(result['predictor'], 'results'):
                        metrics = result['predictor'].results
                        acc = metrics.get('balanced_accuracy', 0)
                        f1 = metrics.get('f1_score', 0)
                        auc = metrics.get('auc_score', 0)
                        meta_model = result.get('best_meta_name', 'N/A')
                        derived_used = result.get('derived_features_used', [])

                        logger.info(f"   🎯 {scenario}:")
                        logger.info(f"      📊 Acurácia: {acc:.3f}")
                        logger.info(f"      🎯 F1-Score: {f1:.3f}")
                        logger.info(f"      📈 AUC: {auc:.3f}")
                        logger.info(f"      🏅 Meta-modelo: {meta_model}")
                        logger.info(f"      🔄 Features derivadas: {len(derived_used)}")

                        if acc > best_score:
                            best_score = acc
                            best_scenario = scenario

                if best_scenario:
                    logger.info(f"\n🏆 MELHOR MODELO: {best_scenario} (Acurácia: {best_score:.3f})")

        # Arquivos gerados
        logger.info(f"\n💾 ARQUIVOS GERADOS:")
        if 'report_generation' in self.results:
            generated_files = self.results['report_generation'].get('generated_files', [])
            for file_path in generated_files[:10]:
                logger.info(f"   📄 {Path(file_path).name}")

            if len(generated_files) > 10:
                logger.info(f"   ... e mais {len(generated_files) - 10} arquivos")

        # Lista de relatórios específicos
        reports_dir = self.output_dir / "reports"
        if reports_dir.exists():
            logger.info(f"\n📋 RELATÓRIOS ESPECÍFICOS GERADOS:")
            specific_reports = [
                "01_feature_selection_summary.txt",
                "02_clinical_timeline_analysis.txt",
                "02_selected_features_temporal_analysis.txt",
                "03_leakage_detection_report.txt"
            ]

            for report in specific_reports:
                report_path = reports_dir / report
                if report_path.exists():
                    logger.info(f"   📄 {report}")

        logger.info(f"\n✅ SISTEMA IA TUBERCULOSE INFANTIL - CONCLUÍDO!")
        logger.info(f"🔄 SUBSTITUIÇÃO INTELIGENTE DE FEATURES APLICADA")
        logger.info(f"📊 Configuração usada salva em: {self.output_dir}/logs/config_used.yaml")
        logger.info("=" * 80)

    def save_checkpoint(self, step: str):
        """Salva checkpoint para permitir resumo"""
        try:
            if not self.config.get('pipeline', {}).get('save_checkpoints', True):
                return

            checkpoint_dir = self.output_dir / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)

            checkpoint_path = checkpoint_dir / f"{step}.json"

            checkpoint_data = {
                'step': step,
                'timestamp': datetime.now().isoformat(),
                'status': self.pipeline_status,
                'step_times': self.step_times,
                'config_snapshot': self.config,
                'results_keys': list(self.results.keys()),
                'substitution_enabled': self.config.get('feature_selection', {}).get('feature_replacement', {}).get('enable_smart_replacement', True)
            }

            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, default=str)

            logger.debug(f"💾 Checkpoint salvo: {checkpoint_path}")

        except Exception as e:
            logger.warning(f"⚠️ Erro ao salvar checkpoint: {e}")

    def _send_notifications(self, success: bool, error: str = None):
        """Envia notificações se configuradas"""
        try:
            notifications_config = self.config.get('notifications', {})

            # Prepara mensagem
            if success:
                message = f"✅ Pipeline IA Tuberculose com Substituição Inteligente concluído com sucesso!\n"
                message += f"📊 Tempo total: {time.time() - self.start_time:.1f}s\n"
                if self.feature_selector:
                    message += f"🔄 Features derivadas: {len(self.feature_selector.derived_features)}\n"
                    message += f"🔄 Features substituídas: {len(self.feature_selector.replaced_features)}\n"
                message += f"📁 Resultados: {self.output_dir}"
            else:
                message = f"❌ Pipeline IA Tuberculose falhou!\n"
                if error:
                    message += f"🔥 Erro: {error[:200]}...\n"
                message += f"📁 Logs: {self.output_dir}/logs/"

        except Exception as e:
            logger.warning(f"⚠️ Erro ao enviar notificações: {e}")

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Retorna status detalhado do pipeline COM INFORMAÇÕES DE SUBSTITUIÇÃO"""
        status = {
            'status': self.pipeline_status,
            'step_times': self.step_times,
            'total_time': time.time() - self.start_time if self.start_time else 0,
            'results_summary': {
                key: {
                    'completed': key in self.results,
                    'timestamp': self.results[key].get('timestamp') if key in self.results else None
                } for key in self.pipeline_status.keys()
            },
            'config_summary': {
                'sections': list(self.config.keys()),
                'use_tabpfn': self.config.get('model', {}).get('use_tabpfn', True),
                'max_features': self.config.get('model', {}).get('max_features', 15),
                'cv_folds': self.config.get('model', {}).get('cv_folds', 10),
                'substitution_enabled': self.config.get('feature_selection', {}).get('feature_replacement', {}).get('enable_smart_replacement', True)
            },
            'output_dir': str(self.output_dir) if self.output_dir else None
        }

        # Adiciona informações de substituição se disponível
        if self.feature_selector:
            status['substitution_summary'] = {
                'derived_features_created': len(self.feature_selector.derived_features),
                'original_features_replaced': len(self.feature_selector.replaced_features),
                'derived_features_list': self.feature_selector.derived_features,
                'replaced_features_list': list(self.feature_selector.replaced_features),
                'replacement_strategy': self.config.get('feature_selection', {}).get('feature_replacement', {}).get('replacement_strategy', 'unknown')
            }

        return status

    def run_statistical_validation(self) -> bool:
        """
        ETAPA 5: Validação Estatística e Testes de Hipótese
        Executa bateria completa de testes estatísticos para validação médica
        """
        try:
            step_start = time.time()
            logger.info("\n" + "=" * 70)
            logger.info("🧪 ETAPA 5: VALIDAÇÃO ESTATÍSTICA E TESTES DE HIPÓTESE")
            logger.info("=" * 70)

            # Verificar se módulo está disponível
            if not STATISTICAL_VALIDATION_AVAILABLE:
                logger.error("❌ Módulo de validação estatística não disponível!")
                return False

            if 'model_training' not in self.results:
                logger.error("❌ Treinamento de modelos não executado!")
                return False

            logger.info("🧪 Executando bateria completa de testes de hipótese médicos...")

            # Configurações da validação estatística
            validation_config = self.config.get('statistical_validation', {
                'confidence_level': 0.95,
                'non_inferiority_margin': 0.05,
                'baseline_performance': 0.70
            })

            # Chama a função importada do módulo de validação
            validation_results = statistical_validation_func(
                training_results=self.results['model_training'],
                feature_selector=self.feature_selector,
                output_dir=self.output_dir,
                config=self.config
            )

            if validation_results.get('status') == 'ERROR':
                logger.error(f"❌ Erro na validação estatística: {validation_results.get('error')}")
                return False

            # Armazenar resultados
            self.results['statistical_validation'] = {
                'validation_results': validation_results['validation_results'],
                'overall_assessment': validation_results['overall_assessment'],
                'report_path': validation_results['report_path'],
                'config_used': validation_results['config_used'],
                'timestamp': datetime.now()
            }

            self.pipeline_status['statistical_validation'] = True
            self.step_times['statistical_validation'] = time.time() - step_start

            # Análise dos resultados
            overall_assessment = validation_results['overall_assessment']
            overall_status = overall_assessment['overall_status']
            significance_rate = overall_assessment['significance_rate']

            logger.info(f"\n📊 RESULTADOS DA VALIDAÇÃO ESTATÍSTICA:")
            logger.info(f"   Status geral: {overall_status}")
            logger.info(f"   Taxa de significância: {significance_rate:.1%}")
            logger.info(
                f"   Testes significativos: {overall_assessment['significant_tests']}/{overall_assessment['total_tests']}")
            logger.info(f"   Recomendação: {overall_assessment['recommendation']}")

            # Log dos testes individuais
            for assessment in overall_assessment['test_assessments']:
                status_icon = "✅" if assessment['significant'] else "⚠️"
                logger.info(
                    f"   {status_icon} {assessment['test']}: {'Significativo' if assessment['significant'] else 'Não significativo'}")

            # Verificar se sistema foi aprovado
            if overall_status == 'APROVADO':
                logger.info("🎉 SISTEMA VALIDADO ESTATISTICAMENTE PARA USO CLÍNICO!")
            elif overall_status == 'APROVADO_COM_RESSALVAS':
                logger.warning("⚠️ SISTEMA APROVADO COM RESSALVAS - Monitoramento adicional necessário")
            else:
                logger.error("❌ SISTEMA REPROVADO NA VALIDAÇÃO ESTATÍSTICA")

                # Configuração para parar ou continuar
                stop_on_statistical_failure = self.config.get('pipeline', {}).get('stop_on_statistical_failure', False)
                if stop_on_statistical_failure:
                    logger.error("🛑 Pipeline interrompido devido à reprovação estatística")
                    return False
                else:
                    logger.warning("⚠️ Continuando pipeline apesar da reprovação (configuração)")

            # Checkpoint
            if self.config.get('pipeline', {}).get('save_checkpoints', True):
                self.save_checkpoint('statistical_validation')

            logger.info("✅ Validação estatística concluída")
            return True

        except Exception as e:
            logger.error(f"❌ Erro na validação estatística: {e}")
            logger.error(traceback.format_exc())
            if self.config.get('pipeline', {}).get('stop_on_failure', False):
                raise
            return False


def main():
    """Função principal para executar o sistema completo COM SUBSTITUIÇÃO INTELIGENTE"""

    try:
        # Configuração padrão (será sobrescrita por args/config)
        DATA_PATH = None
        OUTPUT_DIR = None

        # Inicializa sistema com configuração
        ai_system = TuberculosisAISystem()

        # Executa pipeline completo
        success = ai_system.run_complete_pipeline(
            data_path=DATA_PATH,
            output_dir=OUTPUT_DIR
        )

        if success:
            logger.info("🎉 Sistema executado com sucesso!")

            # Mostra status final COM INFORMAÇÕES DE SUBSTITUIÇÃO
            status = ai_system.get_pipeline_status()
            logger.info("📊 Status final do pipeline:")
            for step, completed in status['status'].items():
                time_taken = status['step_times'].get(step, 0)
                logger.info(f"   {step}: {'✅' if completed else '❌'} ({time_taken:.1f}s)")

            # Mostra resumo das substituições
            if 'substitution_summary' in status:
                subst = status['substitution_summary']
                logger.info(f"\n🔄 RESUMO DAS SUBSTITUIÇÕES:")
                logger.info(f"   Features derivadas criadas: {subst['derived_features_created']}")
                logger.info(f"   Features originais substituídas: {subst['original_features_replaced']}")
                logger.info(f"   Estratégia utilizada: {subst['replacement_strategy']}")

            return 0
        else:
            logger.error("❌ Falha na execução do sistema")
            return 1

    except KeyboardInterrupt:
        logger.warning("⚠️ Execução interrompida pelo usuário")
        return 130

    except Exception as e:
        logger.error(f"❌ Erro crítico: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())