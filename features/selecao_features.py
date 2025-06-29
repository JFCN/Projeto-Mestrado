"""
Sistema Unificado de Seleção de Features para IA Tuberculose
Substituição Inteligente de Features Derivadas

Responsabilidades:
- Cria features derivadas categóricas
- Substitui features originais por versões derivadas quando apropriado
- Executa seleção considerando substituições
- Evita redundâncias entre features originais e derivadas
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Tuple, Any, Optional
import warnings
import logging
import traceback
from pathlib import Path

warnings.filterwarnings('ignore')
logger = logging.getLogger('FeatureSelector')

class TuberculosisFeatureSelector:
    """
    Seletor de Features Unificado com Substituição Inteligente
    """

    def __init__(self, config: Dict[str, Any] = None):

        self.config = config or {}
        self.df = None
        self.feature_categories = self._define_feature_categories()
        self.derived_features = []
        self.scenarios_data = {}
        self.feature_importance = {}
        self.replacement_mappings = {}
        self.replaced_features = set()
        self.replacement_config = self._load_replacement_config()
        self.replacement_logs = []
        enhanced_config = self.config.get('enhanced_features', {})
        self.max_features = enhanced_config.get('max_features', 20)
        self.use_temporal_critical = enhanced_config.get('use_temporal_critical', False)
        self.available_scenarios = {
            'GERAL': 'População total',
            'MASCULINO': 'Apenas homens',
            'FEMININO': 'Apenas mulheres',
            'NEGROS_PARDOS': 'Negros e pardos',
            'OUTROS_RACA': 'Outras raças'
        }

    def _load_replacement_config(self) -> Dict[str, Any]:
        """Carrega configurações de substituição do config"""

        feature_selection_config = self.config.get('feature_selection', {})
        return {
            'enabled': feature_selection_config.get('feature_replacement', {}).get('enable_smart_replacement', True),
            'strategy': feature_selection_config.get('feature_replacement', {}).get('replacement_strategy', 'replace_originals'),
            'prioritize_derived': feature_selection_config.get('feature_replacement', {}).get('prioritize_derived_features', True),
            'log_replacements': feature_selection_config.get('feature_replacement', {}).get('log_replacements', True),
            'validate_replacements': feature_selection_config.get('feature_replacement', {}).get('validate_replacements', True),
            'fallback_to_originals': feature_selection_config.get('feature_replacement', {}).get('fallback_to_originals', True),
            'derived_configs': feature_selection_config.get('derived_features', {}),
            'replacement_rules': feature_selection_config.get('replacement_rules', {}),
            'min_correlation_threshold': feature_selection_config.get('replacement_rules', {}).get('min_correlation_threshold', 0.05),
            'always_keep': feature_selection_config.get('replacement_rules', {}).get('always_keep', []),
            'never_replace': feature_selection_config.get('replacement_rules', {}).get('never_replace', []),
            'max_derived_features': feature_selection_config.get('replacement_rules', {}).get('max_derived_features', 8)
        }

    def _define_feature_categories(self) -> Dict[str, Dict]:
        """Define categorias de features baseadas na análise temporal"""

        return {
            'DEMOGRAFICAS': {
                'features': ['IDADE', 'FAIXA_ETARIA', 'CS_ESCOL_N', 'REGIAO', 'SG_UF', 'ANO_NASC'],
                'priority': 'HIGH',
                'temporal_risk': 'NONE'
            },
            'CLINICAS_INICIAIS': {
                'features': ['FORMA', 'RAIOX_TORA', 'TESTE_TUBE', 'EXTRAPU1_N', 'EXTRAPU2_N',
                             'BACILOSC_E', 'CULTURA_ES', 'TEST_MOLEC', 'CRITERIO_LABORATORIAL',
                             'HISTOPATOL', 'BACILOS_E2', 'CULTURA_OU'],
                'priority': 'HIGH',
                'temporal_risk': 'LOW'
            },
            'COMORBIDADES': {
                'features': ['HIV', 'AGRAVAIDS', 'AGRAVALCOO', 'AGRAVDIABE', 'AGRAVDOENC',
                             'AGRAVDROGA', 'AGRAVTABAC', 'AGRAVOUTRA', 'ANT_RETRO'],
                'priority': 'HIGH',
                'temporal_risk': 'NONE'
            },
            'POPULACOES_ESPECIAIS': {
                'features': ['POP_LIBER', 'POP_RUA', 'POP_IMIG', 'BENEF_GOV'],
                'priority': 'MEDIUM',
                'temporal_risk': 'NONE'
            },
            'TRATAMENTO': {
                'features': ['TRATAMENTO'],
                'priority': 'HIGH',
                'temporal_risk': 'LOW'
            },
            'TEMPORAIS_SEGUROS': {
                'features': ['DIAS_INIC_TRAT'],
                'priority': 'HIGH',
                'temporal_risk': 'LOW'
            },
            'TEMPORAIS_CRITICOS': {
                'features': ['DIAS'],
                'priority': 'EXPERIMENTAL',
                'temporal_risk': 'CRITICAL'
            }
        }

    def load_data(self, data_path: str) -> bool:
        """Carrega dados do caminho especificado com validação"""

        try:
            if data_path.endswith('.csv'):
                self.df = pd.read_csv(data_path)
            elif data_path.endswith(('.xlsx', '.xls')):
                self.df = pd.read_excel(data_path)
            else:
                raise ValueError("Formato não suportado")

            logger.info(f"✅ Dados carregados: {self.df.shape}")

            if 'SITUA_ENCE' not in self.df.columns:
                logger.error("❌ Coluna 'SITUA_ENCE' não encontrada!")
                return False

            return True

        except Exception as e:
            logger.error(f"❌ Erro ao carregar dados: {e}")
            return False

    def create_derived_features(self) -> bool:
        """Cria features derivadas e configura substituições"""

        if self.df is None:
            return False

        logger.info("🔧 Criando features derivadas com estratégia de substituição...")

        if self._should_create_derived_feature('score_comorbidades'):
            self._create_score_comorbidades()

        if self._should_create_derived_feature('caso_complexo'):
            self._create_caso_complexo()

        if self._should_create_derived_feature('risco_social'):
            self._create_risco_social()

        if self._should_create_derived_feature('tempo_inicio_cat'):
            self._create_tempo_inicio_categorizado()

        if self._should_create_derived_feature('perfil_gravidade'):
            self._create_perfil_gravidade()

        if self._should_create_derived_feature('acesso_servicos'):
            self._create_acesso_servicos()

        if self._should_create_derived_feature('duracao_prevista_cat'):
            self._create_duracao_prevista_categorica()

        if self.replacement_config['enabled']:
            self._apply_feature_replacements()

        self._log_replacement_summary()

        return True

    def _should_create_derived_feature(self, feature_name: str) -> bool:
        """Verifica se uma feature derivada deve ser criada"""

        derived_configs = self.replacement_config.get('derived_configs', {})
        feature_config = derived_configs.get(feature_name, {})
        return feature_config.get('enabled', True)

    def _create_score_comorbidades(self) -> bool:
        """Cria SCORE_COMORBIDADES e configura substituição"""

        try:
            comorbidades = ['AGRAVAIDS', 'AGRAVALCOO', 'AGRAVDIABE', 'AGRAVDOENC',
                            'AGRAVDROGA', 'AGRAVTABAC', 'AGRAVOUTRA']

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
            conversion_log = {}

            for col in comorbidades:
                if col not in self.df.columns:
                    continue

                col_data = self.df[col].copy()
                unique_values = col_data.unique()

                col_mapped = col_data.map(mapping_values)

                unmapped_mask = col_mapped.isnull() & col_data.notnull()
                unmapped_values = col_data[unmapped_mask].unique() if unmapped_mask.any() else []

                if len(unmapped_values) > 0:
                    logger.warning(f"   ⚠️ {col}: valores não mapeados {list(unmapped_values)} → assumindo 0")

                col_mapped = col_mapped.fillna(0).astype(int)

                positive_cases = (col_mapped == 1).sum()
                score += col_mapped
                valid_comorbidades.append(col)

                conversion_log[col] = {
                    'unique_original': list(unique_values),
                    'positive_cases': positive_cases,
                    'unmapped_values': list(unmapped_values) if unmapped_values is not None else []
                }

            if not valid_comorbidades:
                logger.error("❌ Nenhuma coluna de comorbidade válida encontrada")
                return False

            if score.sum() == 0:
                logger.warning("⚠️ SCORE_COMORBIDADES resulta em todos zeros - verificar mapeamento")
            self.df['SCORE_COMORBIDADES'] = score
            self.derived_features.append('SCORE_COMORBIDADES')
            self._register_replacement('SCORE_COMORBIDADES', valid_comorbidades, 'high', 'full')
            score_stats = score.value_counts().sort_index()
            total_cases = len(self.df)

            logger.info("✅ SCORE_COMORBIDADES criado com sucesso!")

            if self.replacement_config['log_replacements']:
                logger.info(f"   🔄 Substituirá: {valid_comorbidades}")

            return True

        except Exception as e:
            logger.error(f"❌ Erro ao criar SCORE_COMORBIDADES: {e}")
            logger.error(traceback.format_exc())
            return False

    def _create_caso_complexo(self) -> bool:
        """Cria CASO_COMPLEXO com substituição parcial"""

        try:
            complexidade = []
            replaced_features = []

            if 'FORMA' in self.df.columns:
                forma_normalizada = self.df['FORMA'].astype(str).str.strip().str.lower()
                formas_complexas = forma_normalizada.isin([
                    'extrapulmonar', 'extra-pulmonar', 'extra pulmonar',
                    'pulmonar + extrapulmonar', 'pulmonar+extrapulmonar',
                    'pulmonar e extrapulmonar'
                ])
                complexidade.append(formas_complexas)

            if 'HIV' in self.df.columns:
                hiv_numeric = pd.to_numeric(self.df['HIV'], errors='coerce').fillna(0)
                complexidade.append(hiv_numeric == 1)

            if 'SCORE_COMORBIDADES' in self.df.columns:
                complexidade.append(self.df['SCORE_COMORBIDADES'] >= 2)

            if complexidade:
                self.df['CASO_COMPLEXO'] = np.any(complexidade, axis=0).astype(int)
                self.derived_features.append('CASO_COMPLEXO')
                self._register_replacement('CASO_COMPLEXO', [], 'medium', 'partial')

                return True
            return False

        except Exception as e:
            logger.warning(f"   ⚠️ Erro ao criar CASO_COMPLEXO: {e}")
            return False

    def _create_risco_social(self) -> bool:
        """Cria RISCO_SOCIAL e substitui populações especiais"""

        try:
            pop_especiais = ['POP_LIBER', 'POP_RUA', 'POP_IMIG']
            risco = 0
            valid_pops = []

            for col in pop_especiais:
                if col in self.df.columns:
                    col_numeric = pd.to_numeric(self.df[col], errors='coerce').fillna(0)
                    risco += (col_numeric == 1).astype(int)
                    valid_pops.append(col)

            if risco.sum() > 0:
                self.df['RISCO_SOCIAL'] = risco
                self.derived_features.append('RISCO_SOCIAL')
                self._register_replacement('RISCO_SOCIAL', valid_pops, 'high', 'full')

                if self.replacement_config['log_replacements']:
                    logger.info(f"      🔄 Substituirá: {valid_pops}")

                return True
            return False

        except Exception as e:
            logger.warning(f"   ⚠️ Erro ao criar RISCO_SOCIAL: {e}")
            return False

    def _create_tempo_inicio_categorizado(self) -> bool:
        """Cria TEMPO_INICIO_CAT e substitui versão contínua"""

        try:
            if 'DIAS_INIC_TRAT' not in self.df.columns:
                return False

            dias_numeric = pd.to_numeric(self.df['DIAS_INIC_TRAT'], errors='coerce')
            dias_clean = dias_numeric[(dias_numeric >= 0) & (dias_numeric <= 365)]

            if len(dias_clean) > 0:
                tempo_cat = pd.cut(
                    dias_clean,
                    bins=[-1, 0, 7, 30, 60, float('inf')],
                    labels=[0, 1, 2, 3, 4],
                    include_lowest=True
                )

                tempo_cat_full = pd.Series(np.nan, index=self.df.index)
                tempo_cat_full.loc[dias_clean.index] = tempo_cat.astype(float)

                self.df['TEMPO_INICIO_CAT'] = tempo_cat_full
                self.derived_features.append('TEMPO_INICIO_CAT')
                self._register_replacement('TEMPO_INICIO_CAT', ['DIAS_INIC_TRAT'], 'medium', 'full')

                if self.replacement_config['log_replacements']:
                    logger.info(f"      🔄 Substituirá: DIAS_INIC_TRAT (contínua → categórica)")

                return True
            return False

        except Exception as e:
            logger.warning(f"   ⚠️ Erro ao criar TEMPO_INICIO_CAT: {e}")
            return False

    def _create_perfil_gravidade(self) -> bool:
        """Cria PERFIL_GRAVIDADE (apenas adiciona, não substitui)"""

        try:
            gravidade = []

            if 'IDADE' in self.df.columns:
                idade_numeric = pd.to_numeric(self.df['IDADE'], errors='coerce').fillna(0)
                gravidade.append((idade_numeric <= 2).astype(int))

            if 'HIV' in self.df.columns:
                hiv_numeric = pd.to_numeric(self.df['HIV'], errors='coerce').fillna(0)
                gravidade.append((hiv_numeric == 1).astype(int))

            if 'SCORE_COMORBIDADES' in self.df.columns:
                gravidade.append((self.df['SCORE_COMORBIDADES'] >= 1).astype(int))

            if gravidade:
                self.df['PERFIL_GRAVIDADE'] = np.sum(gravidade, axis=0)
                self.derived_features.append('PERFIL_GRAVIDADE')
                self._register_replacement('PERFIL_GRAVIDADE', [], 'low', 'additive')

                logger.info("   ✅ PERFIL_GRAVIDADE criada")
                return True
            return False

        except Exception as e:
            logger.warning(f"   ⚠️ Erro ao criar PERFIL_GRAVIDADE: {e}")
            return False

    def _create_acesso_servicos(self) -> bool:
        """Cria ACESSO_SERVICOS com substituição parcial"""

        try:
            components = []
            replaced_features = []

            if 'REGIAO' in self.df.columns:
                regiao_numeric = pd.to_numeric(self.df['REGIAO'], errors='coerce').fillna(0)
                acesso_regiao = regiao_numeric.isin([3, 5]).astype(int)
                components.append(acesso_regiao)
                replaced_features.append('REGIAO')

            if 'CS_ESCOL_N' in self.df.columns:
                escol_numeric = pd.to_numeric(self.df['CS_ESCOL_N'], errors='coerce').fillna(0)
                acesso_educacao = (escol_numeric >= 3).astype(int)
                components.append(acesso_educacao)
                replaced_features.append('CS_ESCOL_N')

            if components:
                self.df['ACESSO_SERVICOS'] = np.sum(components, axis=0)
                self.derived_features.append('ACESSO_SERVICOS')
                self._register_replacement('ACESSO_SERVICOS', replaced_features, 'low', 'partial')

                logger.info("   ✅ ACESSO_SERVICOS criada")
                return True
            return False

        except Exception as e:
            logger.warning(f"   ⚠️ Erro ao criar ACESSO_SERVICOS: {e}")
            return False

    def _create_duracao_prevista_categorica(self) -> bool:
        """Cria DURACAO_PREVISTA_CAT (feature completamente nova)"""

        try:
            new_features = self.create_duracao_prevista_categorica(self.df)
            self.derived_features.extend(new_features)
            self._register_replacement('DURACAO_PREVISTA_CAT', [], 'high', 'additive')

            logger.info("   ✅ DURACAO_PREVISTA_CAT criada")
            return True

        except Exception as e:
            logger.warning(f"   ⚠️ Erro ao criar DURACAO_PREVISTA_CAT: {e}")
            return False

    def _register_replacement(self, derived_feature: str, original_features: List[str],
                             priority: str, mode: str):
        """Registra uma substituição para aplicação posterior"""

        if derived_feature not in self.replacement_mappings:
            self.replacement_mappings[derived_feature] = {
                'replaces': original_features,
                'priority': priority,
                'mode': mode,
                'created': True,
                'validated': False
            }

    def _apply_feature_replacements(self):
        """Aplica as substituições configuradas"""

        if not self.replacement_config['enabled']:
            return

        logger.info("🔄 Aplicando substituições de features...")

        replacements_applied = 0

        for derived_feature, replacement_info in self.replacement_mappings.items():
            if not replacement_info['created']:
                continue

            mode = replacement_info['mode']
            original_features = replacement_info['replaces']

            if mode == 'full':
                for original in original_features:
                    if original in self.df.columns and original not in self.replacement_config['never_replace']:
                        self.replaced_features.add(original)
                        replacements_applied += 1

                        if self.replacement_config['log_replacements']:
                            logger.info(f"   🔄 {original} → {derived_feature} (substituição completa)")

            elif mode == 'partial':
                for original in original_features:
                    if original in self.df.columns:
                        if self.replacement_config['log_replacements']:
                            logger.info(f"   ⚡ {original} ↔ {derived_feature} (substituição parcial)")

            elif mode == 'additive':
                if self.replacement_config['log_replacements']:
                    logger.info(f"   ➕ {derived_feature} (feature adicional)")

        logger.info(f"✅ {replacements_applied} substituições aplicadas")

    def _log_replacement_summary(self):
        """Log do resumo das substituições"""

        if not self.replacement_config['log_replacements']:
            return

        logger.info(f"\n📊 RESUMO DAS SUBSTITUIÇÕES:")
        logger.info(f"✅ Features derivadas criadas: {len(self.derived_features)}")
        logger.info(f"🔄 Features originais substituídas: {len(self.replaced_features)}")

        if self.derived_features:
            logger.info(f"📋 Features derivadas:")
            for feature in self.derived_features:
                logger.info(f"   • {feature}")

        if self.replaced_features:
            logger.info(f"🔄 Features substituídas:")
            for feature in self.replaced_features:
                logger.info(f"   • {feature}")

    def get_available_features_for_selection(self, all_features: List[str]) -> List[str]:
        """Retorna features disponíveis para seleção, considerando substituições"""

        if not self.replacement_config['enabled']:
            return all_features

        available_features = []

        for feature in all_features:
            if feature in self.replaced_features:
                continue

            if feature in self.replacement_config.get('never_replace', []):
                available_features.append(feature)
                continue

            available_features.append(feature)

        if self.replacement_config['prioritize_derived']:
            derived_to_add = [f for f in self.derived_features if f not in available_features]
            available_features = derived_to_add + available_features

        return available_features

    def prepare_scenarios(self, scenarios_to_run: List[str] = None) -> Dict[str, pd.DataFrame]:
        """Prepara cenários específicos para processamento"""

        if scenarios_to_run is None:
            scenarios_to_run = list(self.available_scenarios.keys())

        logger.info(f"📋 Preparando cenários: {scenarios_to_run}")

        datasets = {}

        for scenario in scenarios_to_run:
            if scenario not in self.available_scenarios:
                logger.warning(f"⚠️ Cenário desconhecido: {scenario}")
                continue

            logger.info(f"🎯 Processando cenário: {scenario}")

            if scenario == 'GERAL':
                subset = self.df.copy()
                datasets[scenario] = subset
                logger.info(f"   📊 GERAL: {len(subset)} registros")

            elif scenario == 'MASCULINO':
                if 'CS_SEXO' in self.df.columns:
                    valores_sexo = self.df['CS_SEXO'].value_counts()
                    logger.info(f"   🔍 Valores únicos CS_SEXO: {dict(valores_sexo)}")

                    masculino_values = ['M', '1', 1, 'Masculino', 'MASCULINO', 'masculino']
                    subset = self.df[self.df['CS_SEXO'].isin(masculino_values)].copy()

                    if len(subset) > 0:
                        datasets[scenario] = subset
                        logger.info(f"   👨 MASCULINO: {len(subset)} registros")
                    else:
                        logger.warning(f"⚠️ Nenhum registro masculino encontrado")

            elif scenario == 'FEMININO':
                if 'CS_SEXO' in self.df.columns:
                    feminino_values = ['F', '2', 2, 'Feminino', 'FEMININO', 'feminino']
                    subset = self.df[self.df['CS_SEXO'].isin(feminino_values)].copy()

                    if len(subset) > 0:
                        datasets[scenario] = subset
                        logger.info(f"   👩 FEMININO: {len(subset)} registros")

            elif scenario == 'NEGROS_PARDOS':
                if 'CS_RACA' in self.df.columns:
                    negros_pardos_values = [
                        2, 4, '2', '4',
                        'Preta', 'Parda', 'PRETA', 'PARDA',
                        'preta', 'parda'
                    ]
                    subset = self.df[self.df['CS_RACA'].isin(negros_pardos_values)].copy()

                    if len(subset) > 0:
                        datasets[scenario] = subset
                        logger.info(f"   🌍 NEGROS_PARDOS: {len(subset)} registros")

            elif scenario == 'OUTROS_RACA':
                if 'CS_RACA' in self.df.columns:
                    excluir_values = [
                        2, 4, '2', '4',
                        'Preta', 'Parda', 'PRETA', 'PARDA',
                        'preta', 'parda'
                    ]
                    subset = self.df[~self.df['CS_RACA'].isin(excluir_values)].copy()

                    if len(subset) > 0:
                        datasets[scenario] = subset
                        logger.info(f"   🌍 OUTROS_RACA: {len(subset)} registros")

        return datasets

    def select_features_for_scenario(self, df: pd.DataFrame, scenario: str) -> List[str]:
        """Seleciona features otimizadas para um cenário específico considerando substituições"""

        logger.info(f"🔍 Selecionando features para {scenario} (com substituições)")

        essential_features = self._get_essential_features(scenario)
        all_features = [col for col in df.columns if col != 'SITUA_ENCE']
        available_features = self.get_available_features_for_selection(all_features)
        safe_features = self._get_safe_features(available_features)
        selected_features = self._hierarchical_selection(df, safe_features, essential_features, scenario)

        if self.replacement_config['log_replacements']:
            replaced_in_selection = [f for f in selected_features if f in self.derived_features]
            if replaced_in_selection:
                logger.info(f"   🔄 Features derivadas selecionadas: {replaced_in_selection}")

        logger.info(f"✅ {len(selected_features)} features selecionadas para {scenario}")

        return selected_features

    def _get_essential_features(self, scenario: str) -> List[str]:
        """Features essenciais por cenário considerando substituições"""

        base_essential = ['IDADE', 'FORMA', 'HIV', 'TRATAMENTO']

        if 'SCORE_COMORBIDADES' in self.derived_features:
            base_essential.append('SCORE_COMORBIDADES')
        if 'DURACAO_PREVISTA_CAT' in self.derived_features:
            base_essential.append('DURACAO_PREVISTA_CAT')

        if scenario == 'GERAL':
            essential = base_essential + ['CS_SEXO', 'CS_RACA', 'REGIAO']
        elif scenario in ['MASCULINO', 'FEMININO']:
            essential = base_essential + ['CS_RACA', 'REGIAO']
        elif scenario in ['NEGROS_PARDOS', 'OUTROS_RACA']:
            essential = base_essential + ['CS_SEXO', 'REGIAO']
        else:
            essential = base_essential

        available_essential = [f for f in essential if f in self.df.columns and f not in self.replaced_features]

        return available_essential

    def _get_safe_features(self, available_features: List[str]) -> List[str]:
        """Filtra features por segurança temporal"""

        safe_features = []

        for feature in available_features:
            category = self._get_feature_category(feature)

            if category:
                temporal_risk = self.feature_categories[category]['temporal_risk']

                if temporal_risk in ['NONE', 'LOW']:
                    safe_features.append(feature)
                elif temporal_risk == 'CRITICAL' and self.use_temporal_critical:
                    safe_features.append(feature)
                    logger.warning(f"⚠️ Incluindo feature temporal crítica: {feature}")
            else:
                if feature in self.derived_features:
                    safe_features.append(feature)
                    if self.replacement_config['log_replacements']:
                        logger.debug(f"   ✅ Feature derivada considerada segura: {feature}")
                else:
                    safe_features.append(feature)

        return safe_features

    def _get_feature_category(self, feature: str) -> str:
        """Identifica categoria de uma feature"""

        for category, info in self.feature_categories.items():
            if feature in info['features']:
                return category
        return None

    def _hierarchical_selection(self, df: pd.DataFrame, available_features: List[str],
                               essential_features: List[str], scenario: str) -> List[str]:
        """Seleção hierárquica de features com priorização de derivadas"""

        selected = [f for f in essential_features if f in available_features]
        remaining_slots = self.max_features - len(selected)

        if remaining_slots <= 0:
            return selected[:self.max_features]

        if self.replacement_config['prioritize_derived']:
            derived_to_add = []
            for feature in self.derived_features:
                if feature in available_features and feature not in selected:
                    derived_to_add.append(feature)

            max_derived = self.replacement_config.get('max_derived_features', 8)
            derived_to_add = derived_to_add[:min(remaining_slots, max_derived)]

            selected.extend(derived_to_add)
            remaining_slots -= len(derived_to_add)

            if self.replacement_config['log_replacements'] and derived_to_add:
                logger.info(f"   🎯 Features derivadas priorizadas: {derived_to_add}")

        if remaining_slots <= 0:
            return selected

        candidate_features = [f for f in available_features if f not in selected]

        if len(candidate_features) > 0:
            selected_auto = self._statistical_selection(df, candidate_features, remaining_slots)
            selected.extend(selected_auto)

        return selected[:self.max_features]

    def _statistical_selection(self, df: pd.DataFrame, candidates: List[str], n_features: int) -> List[str]:
        """Seleção estatística usando F-score com tratamento robusto de tipos"""

        try:
            X = df[candidates].copy()
            y = df['SITUA_ENCE'].copy()

            for col in X.columns:
                if X[col].dtype == 'object':
                    numeric_col = pd.to_numeric(X[col], errors='coerce')

                    if numeric_col.notna().sum() > len(X) * 0.5:
                        X[col] = numeric_col.fillna(X[col].mode().iloc[0] if not X[col].mode().empty else 0)
                    else:
                        le = LabelEncoder()
                        X[col] = le.fit_transform(X[col].astype(str))

                if X[col].dtype in ['int64', 'float64']:
                    X[col] = X[col].fillna(X[col].median())
                else:
                    X[col] = X[col].fillna(0)

                X[col] = X[col].astype(float)

            if X.isna().any().any():
                logger.warning("⚠️ Ainda há valores NaN após pré-processamento")
                X = X.fillna(0)

            selector = SelectKBest(score_func=f_classif, k=min(n_features, len(candidates)))

            if self.replacement_config['prioritize_derived']:
                adjusted_scores = []
                temp_selector = SelectKBest(score_func=f_classif, k='all')
                temp_selector.fit(X, y)

                original_scores = temp_selector.scores_
                weight_multiplier = self.config.get('model', {}).get('derived_features_handling', {}).get('derived_feature_weight_multiplier', 1.2)

                for i, feature in enumerate(candidates):
                    if feature in self.derived_features:
                        adjusted_scores.append(original_scores[i] * weight_multiplier)
                    else:
                        adjusted_scores.append(original_scores[i])

                feature_scores = list(zip(candidates, adjusted_scores))
                feature_scores.sort(key=lambda x: x[1], reverse=True)
                selected_features = [f[0] for f in feature_scores[:n_features]]

            else:
                selector.fit(X, y)
                selected_mask = selector.get_support()
                selected_features = [candidates[i] for i, selected in enumerate(selected_mask) if selected]

            if 'temp_selector' in locals():
                feature_scores = dict(zip(candidates, temp_selector.scores_))
            else:
                feature_scores = dict(zip(candidates, selector.scores_))
            self.feature_importance.update(feature_scores)

            logger.info(f"   📊 Seleção estatística: {len(selected_features)} features")

            return selected_features

        except Exception as e:
            logger.warning(f"⚠️ Erro na seleção estatística: {e}")
            logger.warning(f"   Usando seleção simples dos primeiros {n_features} features")
            return candidates[:n_features]

    def _prepare_data_for_training(self, df: pd.DataFrame, selected_features: List[str]) -> pd.DataFrame:
        """Prepara dados limpos para treinamento considerando substituições"""

        columns_to_keep = selected_features + ['SITUA_ENCE']
        columns_available = [col for col in columns_to_keep if col in df.columns]

        df_clean = df[columns_available].copy()

        for col in selected_features:
            if col in df_clean.columns:
                if df_clean[col].dtype == 'object':
                    numeric_col = pd.to_numeric(df_clean[col], errors='coerce')

                    if numeric_col.notna().sum() > len(df_clean) * 0.5:
                        df_clean[col] = numeric_col.fillna(0)
                    else:
                        df_clean[col] = df_clean[col].fillna('Desconhecido')
                else:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

        if 'SITUA_ENCE' in df_clean.columns:
            df_clean['SITUA_ENCE'] = pd.to_numeric(df_clean['SITUA_ENCE'], errors='coerce')
            before_len = len(df_clean)
            df_clean = df_clean.dropna(subset=['SITUA_ENCE'])
            after_len = len(df_clean)

            if before_len != after_len:
                logger.info(f"   🧹 Removidas {before_len - after_len} linhas com target inválido")

        return df_clean

    def _validate_temporal_safety(self, features: List[str]) -> Dict[str, Any]:
        """Valida segurança temporal considerando substituições"""

        validation = {
            'safe_features': [],
            'risky_features': [],
            'critical_features': [],
            'derived_features': [],
            'replaced_features': [],
            'risk_level': 'LOW'
        }

        for feature in features:
            if feature in self.derived_features:
                validation['derived_features'].append(feature)
                validation['safe_features'].append(feature)
            elif feature in self.replaced_features:
                validation['replaced_features'].append(feature)
                validation['safe_features'].append(feature)
            else:
                category = self._get_feature_category(feature)
                if category:
                    risk = self.feature_categories[category]['temporal_risk']
                    if risk in ['NONE', 'LOW']:
                        validation['safe_features'].append(feature)
                    elif risk == 'CRITICAL':
                        validation['critical_features'].append(feature)
                else:
                    validation['safe_features'].append(feature)

        if validation['critical_features']:
            validation['risk_level'] = 'CRITICAL'
        elif len(validation['derived_features']) > len(validation['safe_features']) * 0.8:
            validation['risk_level'] = 'LOW'
        return validation

    def run_feature_selection(self, data_path: str, scenarios_to_run: List[str] = None) -> Dict[str, Any]:
        """Executa seleção de features completa com estratégia de substituição"""

        logger.info("🚀 INICIANDO SELEÇÃO DE FEATURES COM SUBSTITUIÇÃO INTELIGENTE")
        logger.info("="*60)

        if not self.load_data(data_path):
            return {}

        if not self.create_derived_features():
            return {}

        datasets = self.prepare_scenarios(scenarios_to_run)

        if not datasets:
            logger.error("❌ Nenhum cenário válido encontrado")
            return {}

        results = {}

        for scenario, dataset in datasets.items():
            logger.info(f"\n🎯 PROCESSANDO CENÁRIO: {scenario}")
            logger.info(f"   📊 Registros: {len(dataset)}")

            if len(dataset) < 50:
                logger.warning(f"⚠️ Poucos dados para {scenario}")
                continue

            target_dist = dataset['SITUA_ENCE'].value_counts()
            logger.info(f"   🎯 Target: {dict(target_dist)}")
            selected_features = self.select_features_for_scenario(dataset, scenario)
            validation = self._validate_temporal_safety(selected_features)
            clean_data = self._prepare_data_for_training(dataset, selected_features)
            replacement_analysis = self._analyze_replacements_for_scenario(selected_features)

            results[scenario] = {
                'data': clean_data,
                'selected_features': selected_features,
                'validation': validation,
                'replacement_analysis': replacement_analysis,
                'description': self.available_scenarios[scenario],
                'risk_level': validation['risk_level'],
                'record_count': len(dataset)
            }

            logger.info(f"   ✅ Features: {len(selected_features)}")
            logger.info(f"   🔄 Features derivadas: {len(validation['derived_features'])}")
            logger.info(f"   🔍 Risco: {validation['risk_level']}")

        self.scenarios_data = results
        self._print_final_replacement_summary(results)

        logger.info("✅ Seleção de features com substituição inteligente concluída!")

        return results

    def _analyze_replacements_for_scenario(self, selected_features: List[str]) -> Dict[str, Any]:
        """Analisa substituições aplicadas em um cenário específico"""

        analysis = {
            'derived_features_used': [],
            'original_features_used': [],
            'replacement_mappings_applied': {},
            'replacement_efficiency': 0.0
        }

        for feature in selected_features:
            if feature in self.derived_features:
                analysis['derived_features_used'].append(feature)
                if feature in self.replacement_mappings:
                    mapping = self.replacement_mappings[feature]
                    analysis['replacement_mappings_applied'][feature] = {
                        'replaces': mapping['replaces'],
                        'mode': mapping['mode'],
                        'priority': mapping['priority']
                    }
            else:
                analysis['original_features_used'].append(feature)

        total_features = len(selected_features)
        derived_count = len(analysis['derived_features_used'])

        if total_features > 0:
            analysis['replacement_efficiency'] = derived_count / total_features

        return analysis

    def _print_final_replacement_summary(self, results: Dict[str, Any]):
        """Imprime resumo final das substituições por cenário"""

        logger.info(f"\n📊 RESUMO FINAL DAS SUBSTITUIÇÕES:")
        logger.info(f"   🔧 Features derivadas criadas: {len(self.derived_features)}")
        logger.info(f"   🔄 Features originais substituídas: {len(self.replaced_features)}")

        for scenario, data in results.items():
            replacement_analysis = data['replacement_analysis']
            derived_count = len(replacement_analysis['derived_features_used'])
            total_count = len(data['selected_features'])
            efficiency = replacement_analysis['replacement_efficiency']

            logger.info(f"\n   🎯 {scenario}:")
            logger.info(f"      Features derivadas usadas: {derived_count}/{total_count} ({efficiency:.1%})")

            if replacement_analysis['derived_features_used']:
                logger.info(f"      Derivadas: {replacement_analysis['derived_features_used']}")

    def create_duracao_prevista_categorica(self, df):
        """Cria feature derivada DURACAO_PREVISTA_CAT baseada no protocolo médico"""

        logger.info("🔧 Criando DURACAO_PREVISTA_CAT...")

        duracao_categoria = pd.Series(0, index=df.index, name='DURACAO_PREVISTA_CAT')

        if 'FORMA' in df.columns:
            forma_normalizada = df['FORMA'].astype(str).str.strip().str.lower()

            extrapulmonar_mask = forma_normalizada.isin([
                'extrapulmonar',
                'extra-pulmonar',
                'extra pulmonar'
            ])
            duracao_categoria.loc[extrapulmonar_mask] = 1
            extrapulmonar_count = extrapulmonar_mask.sum()

            mista_mask = forma_normalizada.isin([
                'pulmonar + extrapulmonar',
                'pulmonar+extrapulmonar',
                'pulmonar e extrapulmonar'
            ])
            duracao_categoria.loc[mista_mask] = 2
            mista_count = mista_mask.sum()

            pulmonar_count = len(df) - extrapulmonar_count - mista_count

            logger.info(f"   📊 Distribuição aplicada:")
            logger.info(f"      Pulmonar + Ignorado (cat 0): {pulmonar_count} casos")
            logger.info(f"      Extrapulmonar (cat 1): {extrapulmonar_count} casos")
            logger.info(f"      Mista (cat 2): {mista_count} casos")

        condicoes_criticas = []

        if 'HIV' in df.columns:
            hiv_numeric = pd.to_numeric(df['HIV'], errors='coerce').fillna(0)
            hiv_positivo = (hiv_numeric == 1)
            condicoes_criticas.append(hiv_positivo)

        if 'SCORE_COMORBIDADES' in df.columns:
            comorbidades_multiplas = (df['SCORE_COMORBIDADES'] >= 3)
            condicoes_criticas.append(comorbidades_multiplas)

        if 'CASO_COMPLEXO' in df.columns:
            caso_complexo = (df['CASO_COMPLEXO'] == 1)
            condicoes_criticas.append(caso_complexo)

        if 'IDADE' in df.columns:
            idade_numeric = pd.to_numeric(df['IDADE'], errors='coerce').fillna(0)
            bebes = (idade_numeric <= 2)
            condicoes_criticas.append(bebes)

        if condicoes_criticas:
            casos_criticos = np.any(condicoes_criticas, axis=0)

            casos_para_estender = casos_criticos & (duracao_categoria == 0)
            duracao_categoria.loc[casos_para_estender] = 1

            if 'HIV' in df.columns and 'FORMA' in df.columns:
                hiv_numeric = pd.to_numeric(df['HIV'], errors='coerce').fillna(0)
                forma_numeric = pd.to_numeric(df['FORMA'], errors='coerce').fillna(9)
                casos_maximos = (hiv_numeric == 1) & (forma_numeric == 3) & casos_criticos
                duracao_categoria.loc[casos_maximos] = 2

        df['DURACAO_PREVISTA_CAT'] = duracao_categoria

        categorias = df['DURACAO_PREVISTA_CAT'].value_counts().sort_index()
        total_casos = len(df)

        logger.info(f"   ✅ DURACAO_PREVISTA_CAT criada com sucesso:")
        logger.info(f"      📊 Padrão (6m): {categorias.get(0, 0)} casos ({categorias.get(0, 0) / total_casos * 100:.1f}%)")
        logger.info(f"      📊 Intermediário (9m): {categorias.get(1, 0)} casos ({categorias.get(1, 0) / total_casos * 100:.1f}%)")
        logger.info(f"      📊 Estendido (12m): {categorias.get(2, 0)} casos ({categorias.get(2, 0) / total_casos * 100:.1f}%)")

        return ['DURACAO_PREVISTA_CAT']

    def get_data_for_scenario(self, scenario: str) -> Tuple[pd.DataFrame, List[str]]:
        """Retorna dados e features para um cenário específico"""

        if scenario not in self.scenarios_data:
            logger.error(f"❌ Cenário {scenario} não encontrado")
            return pd.DataFrame(), []

        scenario_data = self.scenarios_data[scenario]
        return scenario_data['data'], scenario_data['selected_features']

    def list_available_scenarios(self) -> Dict[str, str]:
        """Lista cenários disponíveis"""

        return self.available_scenarios.copy()

    def list_processed_scenarios(self) -> List[str]:
        """Lista cenários já processados"""

        return list(self.scenarios_data.keys())

    def get_replacement_summary(self) -> Dict[str, Any]:
        """Retorna resumo das substituições realizadas"""

        return {
            'replacement_enabled': self.replacement_config['enabled'],
            'replacement_strategy': self.replacement_config['strategy'],
            'derived_features_created': self.derived_features.copy(),
            'replaced_features': list(self.replaced_features),
            'replacement_mappings': self.replacement_mappings.copy(),
            'replacement_logs': self.replacement_logs.copy()
        }

    def save_results_summary(self, output_path: str):
        """Salva resumo dos resultados incluindo substituições"""

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("RESUMO DE SELEÇÃO DE FEATURES COM SUBSTITUIÇÃO INTELIGENTE\n")
                f.write("="*80 + "\n\n")

                f.write("1. ESTRATÉGIA DE SUBSTITUIÇÃO\n")
                f.write("-"*40 + "\n")
                f.write(f"Substituição habilitada: {self.replacement_config['enabled']}\n")
                f.write(f"Estratégia: {self.replacement_config['strategy']}\n")
                f.write(f"Priorizar derivadas: {self.replacement_config['prioritize_derived']}\n\n")

                f.write("2. FEATURES DERIVADAS CRIADAS\n")
                f.write("-"*40 + "\n")
                for feature in self.derived_features:
                    f.write(f"• {feature}\n")
                    if feature in self.replacement_mappings:
                        mapping = self.replacement_mappings[feature]
                        f.write(f"  Substitui: {mapping['replaces']}\n")
                        f.write(f"  Modo: {mapping['mode']}\n")
                        f.write(f"  Prioridade: {mapping['priority']}\n")

                f.write(f"\n3. FEATURES ORIGINAIS SUBSTITUÍDAS\n")
                f.write("-"*40 + "\n")
                for feature in self.replaced_features:
                    f.write(f"• {feature}\n")

                f.write(f"\n4. CENÁRIOS PROCESSADOS\n")
                f.write("-"*40 + "\n")

                for scenario, data in self.scenarios_data.items():
                    f.write(f"\n{scenario} ({data['description']}):\n")
                    f.write(f"  Registros: {data['record_count']}\n")
                    f.write(f"  Features totais: {len(data['selected_features'])}\n")
                    f.write(f"  Features derivadas: {len(data['replacement_analysis']['derived_features_used'])}\n")
                    f.write(f"  Eficiência substituição: {data['replacement_analysis']['replacement_efficiency']:.1%}\n")
                    f.write(f"  Risco: {data['risk_level']}\n")
                    f.write(f"  Features selecionadas:\n")
                    for feature in data['selected_features']:
                        marker = "🔄" if feature in self.derived_features else "📊"
                        f.write(f"    {marker} {feature}\n")

            logger.info(f"✅ Resumo com substituições salvo: {output_path}")

        except Exception as e:
            logger.error(f"❌ Erro ao salvar resumo: {e}")

    def fit(self, X, y=None):
        """
        Compatibilidade com interface sklearn
        """

        if hasattr(self, 'selected_features') and self.selected_features:
            self.feature_names_ = self.selected_features
        else:
            self.feature_names_ = list(X.columns) if hasattr(X, 'columns') else [f'feature_{i}' for i in
                                                                                 range(X.shape[1])]

        logger.info(f"✅ FeatureSelector fit com {len(self.feature_names_)} features")
        return self

    def transform(self, X):
        """
        Aplica a seleção de features aos dados
        """
        try:
            if not hasattr(self, 'feature_names_'):
                logger.warning("⚠️ FeatureSelector não foi fitted. Usando todas as features.")
                return X

            if hasattr(X, 'columns'):
                available_features = [f for f in self.feature_names_ if f in X.columns]
                if len(available_features) != len(self.feature_names_):
                    missing = set(self.feature_names_) - set(X.columns)
                    logger.warning(f"⚠️ Features ausentes: {missing}")

                X_selected = X[available_features]
                logger.info(f"✅ Transform aplicado: {X.shape[1]} → {X_selected.shape[1]} features")
                return X_selected

            else:
                if hasattr(self, 'feature_indices_'):
                    X_selected = X[:, self.feature_indices_]
                    logger.info(f"✅ Transform aplicado: {X.shape[1]} → {X_selected.shape[1]} features")
                    return X_selected
                else:
                    logger.warning("⚠️ Índices de features não disponíveis para array numpy")
                    return X

        except Exception as e:
            logger.error(f"❌ Erro no transform: {e}")
            return X

    def fit_transform(self, X, y=None):
        """
        Fit e transform em uma única operação
        """
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        """
        Retorna nomes das features selecionadas (compatibilidade sklearn)
        """

        if hasattr(self, 'feature_names_'):
            return self.feature_names_
        elif input_features is not None:
            return list(input_features)
        else:
            return []


def run_feature_selection_for_main(data_path: str, config: Dict[str, Any],
                                  scenarios: List[str] = None) -> TuberculosisFeatureSelector:
    """Função principal para integração com main.py"""

    logger.info("🔧 INTEGRAÇÃO COM SISTEMA PRINCIPAL - SUBSTITUIÇÃO INTELIGENTE")
    selector = TuberculosisFeatureSelector(config)
    results = selector.run_feature_selection(data_path, scenarios)

    if results:
        logger.info("✅ Seleção com substituição inteligente concluída!")

        replacement_summary = selector.get_replacement_summary()
        if replacement_summary['replacement_enabled']:
            logger.info(f"🔄 Substituições: {len(replacement_summary['derived_features_created'])} derivadas criadas")
            logger.info(f"🔄 Substituições: {len(replacement_summary['replaced_features'])} originais substituídas")

        return selector
    else:
        logger.error("❌ Falha na seleção de features")
        return None


def integrate_enhanced_features_with_main(data_path: str, config: Dict[str, Any] = None,
                                        scenarios: List[str] = None) -> TuberculosisFeatureSelector:
    """Função de compatibilidade"""

    return run_feature_selection_for_main(data_path, config or {}, scenarios)