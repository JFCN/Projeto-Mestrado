"""
Módulo 1: Analisador da Linha Temporal Clínica
Separa informações por momento de disponibilidade clínica
SUPORTE A FEATURES DERIVADAS E SUBSTITUIÇÃO INTELIGENTE

Responsabilidades:
- Identifica quando cada feature está disponível na linha temporal
- Categoriza features por momento clínico
- Trata features derivadas como seguras por construção
- Integra com sistema de substituição de features
- Identifica features temporalmente suspeitas vs substituídas
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger('ClinicalTimelineAnalyzer')

class ClinicalTimelineAnalyzer:
    """Analisador da linha temporal clínica para tuberculose infantil"""

    def __init__(self, feature_selector=None):
        self.timeline_moments = self._define_clinical_timeline()
        self.feature_categories = {}
        self.temporal_suspicious = []
        self.safe_features = []
        self.feature_selector = feature_selector
        self.derived_features = []
        self.replaced_features = set()
        self.replacement_mappings = {}

        if self.feature_selector:
            self._load_replacement_info()

    def _load_replacement_info(self):
        """Carrega informações de substituição do feature selector"""

        if hasattr(self.feature_selector, 'derived_features'):
            self.derived_features = self.feature_selector.derived_features.copy()

        if hasattr(self.feature_selector, 'replaced_features'):
            self.replaced_features = self.feature_selector.replaced_features.copy()

        if hasattr(self.feature_selector, 'replacement_mappings'):
            self.replacement_mappings = self.feature_selector.replacement_mappings.copy()

    def _define_clinical_timeline(self) -> Dict[str, Dict]:
        """Define momentos da linha temporal clínica"""

        return {
            "MOMENTO_0_NOTIFICACAO": {
                "quando": "Notificação inicial / Primeira consulta",
                "tempo": "Dia 0",
                "vazamento": "NENHUM",
                "risco": "BAIXO",
                "informacoes_disponiveis": [
                    # Dados demográficos
                    "IDADE", "CS_SEXO", "CS_RACA", "CS_ESCOL_N", "SG_UF",
                    "NU_ANO", "ANO_NASC", "FAIXA_ETARIA",

                    # História clínica inicial
                    "FORMA", "RAIOX_TORA", "TESTE_TUBE",
                    "EXTRAPU1_N", "EXTRAPU2_N",

                    # Comorbidades conhecidas
                    "HIV", "AGRAVAIDS", "AGRAVALCOO", "AGRAVDIABE",
                    "AGRAVDOENC", "AGRAVDROGA", "AGRAVTABAC", "AGRAVOUTRA",

                    # Populações especiais
                    "POP_LIBER", "POP_RUA", "POP_IMIG",

                    # Exames iniciais
                    "BACILOS_E2", "CULTURA_OU", "TEST_MOLEC",
                    "CRITERIO_LABORATORIAL", "ANT_RETRO",

                    # Features derivadas baseadas em dados do momento inicial
                    "SCORE_COMORBIDADES",
                    "CASO_COMPLEXO",
                    "RISCO_SOCIAL",
                    "PERFIL_GRAVIDADE",
                    "ACESSO_SERVICOS",
                    "DURACAO_PREVISTA_CAT",
                ],
                "uso_clinico": "✅ ADEQUADO para IA preditiva (incluindo features derivadas)",
                "cor": "green"
            },

            "MOMENTO_1_INICIO_TRATAMENTO": {
                "quando": "Início do tratamento",
                "tempo": "Dias 0-7",
                "vazamento": "MÍNIMO",
                "risco": "BAIXO",
                "informacoes_disponiveis": [
                    "DIAS_INIC_TRAT",
                    "TEMPO_INICIO_CAT",
                ],
                "uso_clinico": "✅ ADEQUADO (features derivadas preferíveis)",
                "cor": "blue"
            },

            "MOMENTO_2_ACOMPANHAMENTO": {
                "quando": "Durante o tratamento",
                "tempo": "Dias 15, 30, 60, 120...",
                "vazamento": "CRÍTICO",
                "risco": "ALTO",
                "informacoes_disponiveis": [
                    # Exames de controle
                    "BACILOSC_1", "BACILOSC_2", "BACILOSC_3",
                    "BACILOSC_4", "BACILOSC_5", "BACILOSC_6",

                    # Resultados descobertos DURANTE tratamento
                    "TEST_SENSI",
                    "DOENCA_TRA",
                    "TRANSF",

                    # Resistência descoberta
                    "RIFAMPICIN", "ISONIAZIDA", "ETAMBUTOL",
                    "ESTREPTOMI", "PIRAZINAMI", "ETIONAMIDA",

                    # Mudanças no tratamento
                    "TRATAMENTO",
                    "OUTRAS"
                ],
                "uso_clinico": "❌ VAZAMENTO - descoberto após início",
                "cor": "orange"
            },

            "MOMENTO_3_FINALIZACAO": {
                "quando": "Final do tratamento",
                "tempo": "Dia do encerramento",
                "vazamento": "CRÍTICO",
                "risco": "CRÍTICO",
                "informacoes_disponiveis": [
                    "DIAS",
                    "BAC_APOS_6",
                    "SITUA_ENCE",
                    "DT_ENCERRA"
                ],
                "uso_clinico": "❌ VAZAMENTO ÓBVIO - é o próprio desfecho!",
                "cor": "red"
            }
        }

    def analyze_clinical_timeline(self) -> Dict[str, Any]:
        """Executa análise completa da linha temporal clínica"""

        logger.info("🕐 Analisando linha temporal clínica...")

        self._print_timeline_summary()
        leakage_analysis = self._analyze_leakage_by_category()
        clinical_solution = self._generate_clinical_solution()
        self._categorize_features_by_safety()
        derived_features_analysis = self._analyze_derived_features()
        replacement_analysis = self._analyze_feature_replacements()

        return {
            'timeline_moments': self.timeline_moments,
            'leakage_analysis': leakage_analysis,
            'clinical_solution': clinical_solution,
            'temporal_suspicious': self.temporal_suspicious,
            'safe_features': self.safe_features,
            'derived_features_analysis': derived_features_analysis,
            'replacement_analysis': replacement_analysis,
            'analysis_timestamp': datetime.now()
        }

    def _print_timeline_summary(self):
        """Imprime resumo da linha temporal"""

        logger.info("📋 LINHA TEMPORAL CLÍNICA:")

        for momento_key, momento_info in self.timeline_moments.items():
            total_features = len(momento_info['informacoes_disponiveis'])
            derived_count = sum(1 for f in momento_info['informacoes_disponiveis'] if f in self.derived_features)
            replaced_count = sum(1 for f in momento_info['informacoes_disponiveis'] if f in self.replaced_features)

            logger.info(f"{momento_info['quando']}: {total_features} features ({derived_count} derivadas, {replaced_count} substituídas) - {momento_info['uso_clinico']}")

    def _analyze_derived_features(self) -> Dict[str, Any]:
        """Analisa features derivadas"""

        analysis = {
            'total_derived_features': len(self.derived_features),
            'derived_by_moment': {},
            'derived_safety_assessment': {},
            'clinical_validity': {}
        }

        for momento_key, momento_info in self.timeline_moments.items():
            derived_in_moment = [f for f in momento_info['informacoes_disponiveis'] if f in self.derived_features]
            analysis['derived_by_moment'][momento_key] = {
                'moment_name': momento_info['quando'],
                'derived_features': derived_in_moment,
                'count': len(derived_in_moment),
                'safety_level': momento_info['risco']
            }

        for feature in self.derived_features:
            safety_assessment = self._assess_derived_feature_safety(feature)
            analysis['derived_safety_assessment'][feature] = safety_assessment
            clinical_validity = self._assess_clinical_validity(feature)
            analysis['clinical_validity'][feature] = clinical_validity

        logger.info(f"📊 Features derivadas: {analysis['total_derived_features']} (seguras para uso preditivo)")

        return analysis

    def _assess_derived_feature_safety(self, feature: str) -> Dict[str, Any]:
        """Avalia segurança temporal de uma feature derivada"""

        safety_levels = {
            'SCORE_COMORBIDADES': {
                'risk_level': 'BAIXO',
                'reason': 'Baseado em comorbidades conhecidas na anamnese inicial',
                'temporal_safe': True,
                'clinical_moment': 'NOTIFICACAO'
            },
            'CASO_COMPLEXO': {
                'risk_level': 'BAIXO',
                'reason': 'Baseado em forma clínica e HIV conhecidos no diagnóstico',
                'temporal_safe': True,
                'clinical_moment': 'NOTIFICACAO'
            },
            'RISCO_SOCIAL': {
                'risk_level': 'BAIXO',
                'reason': 'Baseado em populações especiais identificadas na anamnese',
                'temporal_safe': True,
                'clinical_moment': 'NOTIFICACAO'
            },
            'TEMPO_INICIO_CAT': {
                'risk_level': 'BAIXO',
                'reason': 'Categorização de tempo real até início',
                'temporal_safe': True,
                'clinical_moment': 'INICIO_TRATAMENTO'
            },
            'PERFIL_GRAVIDADE': {
                'risk_level': 'BAIXO',
                'reason': 'Síntese de fatores de gravidade conhecidos no diagnóstico',
                'temporal_safe': True,
                'clinical_moment': 'NOTIFICACAO'
            },
            'ACESSO_SERVICOS': {
                'risk_level': 'BAIXO',
                'reason': 'Baseado em região e educação',
                'temporal_safe': True,
                'clinical_moment': 'NOTIFICACAO'
            },
            'DURACAO_PREVISTA_CAT': {
                'risk_level': 'BAIXO',
                'reason': 'Baseado em protocolo clínico, não duração real',
                'temporal_safe': True,
                'clinical_moment': 'NOTIFICACAO'
            }
        }

        return safety_levels.get(feature, {
            'risk_level': 'BAIXO',
            'reason': 'Feature derivada - assumida como segura por construção',
            'temporal_safe': True,
            'clinical_moment': 'UNKNOWN'
        })

    def _assess_clinical_validity(self, feature: str) -> Dict[str, Any]:
        """Avalia validade clínica de uma feature derivada"""

        clinical_assessments = {
            'SCORE_COMORBIDADES': {
                'clinical_rationale': 'Soma de comorbidades é melhor preditor que análise individual',
                'medical_literature_support': 'Alto',
                'interpretability': 'Excelente',
                'clinical_actionability': 'Direta - número de comorbidades orienta complexidade do caso'
            },
            'CASO_COMPLEXO': {
                'clinical_rationale': 'Identifica casos que requerem manejo especializado',
                'medical_literature_support': 'Alto',
                'interpretability': 'Excelente',
                'clinical_actionability': 'Direta - orienta nível de cuidado necessário'
            },
            'RISCO_SOCIAL': {
                'clinical_rationale': 'Fatores sociais são preditores importantes de adesão',
                'medical_literature_support': 'Alto',
                'interpretability': 'Boa',
                'clinical_actionability': 'Indireta - orienta suporte social necessário'
            },
            'TEMPO_INICIO_CAT': {
                'clinical_rationale': 'Categorização é mais interpretável que tempo contínuo',
                'medical_literature_support': 'Médio',
                'interpretability': 'Excelente',
                'clinical_actionability': 'Moderada - orienta urgência do caso'
            },
            'PERFIL_GRAVIDADE': {
                'clinical_rationale': 'Síntese de múltiplos fatores de risco de gravidade',
                'medical_literature_support': 'Alto',
                'interpretability': 'Boa',
                'clinical_actionability': 'Direta - orienta intensidade do monitoramento'
            },
            'ACESSO_SERVICOS': {
                'clinical_rationale': 'Acesso a serviços influencia outcomes em tuberculose',
                'medical_literature_support': 'Alto',
                'interpretability': 'Boa',
                'clinical_actionability': 'Indireta - orienta necessidade de suporte adicional'
            },
            'DURACAO_PREVISTA_CAT': {
                'clinical_rationale': 'Duração baseada em protocolo vs duração real evita vazamento',
                'medical_literature_support': 'Alto',
                'interpretability': 'Excelente',
                'clinical_actionability': 'Direta - orienta planejamento do tratamento'
            }
        }

        return clinical_assessments.get(feature, {
            'clinical_rationale': 'Feature derivada sem avaliação clínica específica',
            'medical_literature_support': 'Não avaliado',
            'interpretability': 'Não avaliado',
            'clinical_actionability': 'Não avaliado'
        })

    def _analyze_feature_replacements(self) -> Dict[str, Any]:
        """Analisa o impacto das substituições na linha temporal"""

        analysis = {
            'total_replacements': len(self.replacement_mappings),
            'replacement_by_type': {},
            'temporal_impact': {},
            'clinical_benefit': {}
        }

        for derived_feature, replacement_info in self.replacement_mappings.items():
            replaced_features = replacement_info.get('replaces', [])
            replacement_mode = replacement_info.get('mode', 'unknown')
            temporal_impact = self._assess_replacement_temporal_impact(derived_feature, replaced_features)
            analysis['temporal_impact'][derived_feature] = temporal_impact
            clinical_benefit = self._assess_replacement_clinical_benefit(derived_feature, replaced_features)
            analysis['clinical_benefit'][derived_feature] = clinical_benefit
            if replacement_mode not in analysis['replacement_by_type']:
                analysis['replacement_by_type'][replacement_mode] = []
            analysis['replacement_by_type'][replacement_mode].append({
                'derived': derived_feature,
                'replaces': replaced_features,
                'count_replaced': len(replaced_features)
            })

        if analysis['total_replacements'] > 0:
            logger.info(f"🔄 Substituições configuradas: {analysis['total_replacements']}")

        return analysis

    def _assess_replacement_temporal_impact(self, derived_feature: str, replaced_features: List[str]) -> Dict[str, Any]:
        """Avalia impacto temporal de uma substituição"""

        impact = {
            'reduces_temporal_risk': False,
            'maintains_clinical_info': True,
            'improves_interpretability': True,
            'risk_reduction_score': 0
        }

        risky_replaced = sum(1 for feature in replaced_features if self._is_feature_temporally_risky(feature))

        if risky_replaced > 0:
            impact['reduces_temporal_risk'] = True
            impact['risk_reduction_score'] = risky_replaced

        if derived_feature == 'TEMPO_INICIO_CAT' and 'DIAS_INIC_TRAT' in replaced_features:
            impact['reduces_temporal_risk'] = True
            impact['improves_interpretability'] = True
            impact['clinical_justification'] = 'Categorização é mais interpretável e robusta'

        if derived_feature == 'SCORE_COMORBIDADES':
            impact['reduces_temporal_risk'] = False
            impact['improves_interpretability'] = True
            impact['clinical_justification'] = 'Soma é mais informativa que análise individual'

        return impact

    def _assess_replacement_clinical_benefit(self, derived_feature: str, replaced_features: List[str]) -> Dict[str, Any]:
        """Avalia benefício clínico de uma substituição"""

        benefits = {
            'SCORE_COMORBIDADES': {
                'information_density': 'Alta - condensa 7 features em 1 score',
                'clinical_interpretability': 'Melhor - número total é mais interpretável',
                'predictive_power': 'Mantém ou melhora - soma captura carga total',
                'clinical_actionability': 'Melhor - score orienta diretamente complexidade'
            },
            'CASO_COMPLEXO': {
                'information_density': 'Alta - síntese de múltiplos fatores',
                'clinical_interpretability': 'Excelente - binário complexo/simples',
                'predictive_power': 'Melhora - captura interações entre fatores',
                'clinical_actionability': 'Excelente - decisão direta de nível de cuidado'
            },
            'RISCO_SOCIAL': {
                'information_density': 'Média - condensa 3 features',
                'clinical_interpretability': 'Boa - score de vulnerabilidade social',
                'predictive_power': 'Mantém - soma preserva informação',
                'clinical_actionability': 'Boa - orienta necessidade de suporte'
            },
            'TEMPO_INICIO_CAT': {
                'information_density': 'Mantém - mesma informação categorizada',
                'clinical_interpretability': 'Melhor - categorias têm significado clínico',
                'predictive_power': 'Melhora - categorização captura thresholds clínicos',
                'clinical_actionability': 'Melhor - categorias orientam urgência'
            }
        }

        return benefits.get(derived_feature, {
            'information_density': 'Não avaliada',
            'clinical_interpretability': 'Não avaliada',
            'predictive_power': 'Não avaliada',
            'clinical_actionability': 'Não avaliada'
        })

    def _is_feature_temporally_risky(self, feature: str) -> bool:
        """Verifica se uma feature tem risco temporal"""

        risky_patterns = [
            'BACILOSC_', 'BAC_APOS', 'TRANSF', 'DIAS', '_APOS_',
            'RESULTADO', 'DESFECHO', 'ENCERRA', 'TRAT'
        ]

        return any(pattern in feature.upper() for pattern in risky_patterns)

    def _analyze_leakage_by_category(self) -> Dict[str, Any]:
        """Análise de vazamento por categoria"""

        leakage_summary = {}

        for momento, info in self.timeline_moments.items():
            momento_nome = info["quando"]
            vazamento = info["vazamento"]
            risco = info["risco"]
            uso_clinico = info["uso_clinico"]
            original_features = []
            derived_features = []
            replaced_features = []

            for feature in info['informacoes_disponiveis']:
                if feature in self.derived_features:
                    derived_features.append(feature)
                elif feature in self.replaced_features:
                    replaced_features.append(feature)
                else:
                    original_features.append(feature)

            leakage_summary[momento] = {
                'momento_nome': momento_nome,
                'vazamento': vazamento,
                'risco': risco,
                'uso_clinico': uso_clinico,
                'original_features': original_features,
                'derived_features': derived_features,
                'replaced_features': replaced_features,
                'total_features': len(info['informacoes_disponiveis'])
            }

        return leakage_summary

    def _generate_clinical_solution(self) -> Dict[str, List[str]]:
        """Gera proposta de solução clínica"""

        solution = {
            "features_iniciais_ok": [
                "Dados demográficos completos",
                "Forma clínica na apresentação inicial",
                "Comorbidades conhecidas na anamnese",
                "Exames iniciais solicitados",
                "Populações especiais identificadas",
                "Critérios laboratoriais iniciais"
            ],
            "features_derivadas_recomendadas": [
                "SCORE_COMORBIDADES = soma(todas as comorbidades binárias)",
                "CASO_COMPLEXO = f(forma_extrapulmonar, HIV_positivo, múltiplas_comorbidades)",
                "RISCO_SOCIAL = soma(população_liberada, rua, imigrante)",
                "DURACAO_PREVISTA_CAT = categorias baseadas em protocolo clínico",
                "PERFIL_GRAVIDADE = f(idade_crítica, HIV, score_comorbidades)",
                "ACESSO_SERVICOS = f(região_sul_sudeste, educação_superior)"
            ],
            "features_substituidas_com_sucesso": [
                "SCORE_COMORBIDADES substitui 7 features individuais de comorbidades",
                "RISCO_SOCIAL substitui 3 features de populações especiais",
                "TEMPO_INICIO_CAT substitui DIAS_INIC_TRAT (versão categórica mais robusta)"
            ],
            "features_vazamento_critico": [
                "DIAS (duração real do tratamento) ← NUNCA USAR",
                "RIFAMPICIN='Resistente' (descoberto depois) ← SUBSTITUIR por protocolo",
                "TRATAMENTO='Esquema especial' (mudança posterior) ← USAR tratamento inicial",
                "BACILOSC_1, BACILOSC_2... (exames de acompanhamento) ← EVITAR",
                "TEST_SENSI (resultado de sensibilidade posterior) ← USAR protocolo inicial",
                "BAC_APOS_6 (resultado final) ← OBVIO VAZAMENTO"
            ],
            "estrategia_substituicao": [
                "1. SEMPRE priorizar features derivadas quando disponíveis",
                "2. USAR features derivadas em vez de componentes individuais",
                "3. MANTER apenas features originais essenciais não substituíveis",
                "4. VALIDAR que substituições fazem sentido clinicamente",
                "5. MONITORAR se features derivadas mantêm ou melhoram performance"
            ]
        }

        return solution

    def _categorize_features_by_safety(self):
        """Categoriza features por segurança temporal"""

        safe_features = []
        for momento_key, momento_info in self.timeline_moments.items():
            if momento_info['vazamento'] in ['NENHUM', 'MÍNIMO']:
                safe_features.extend(momento_info['informacoes_disponiveis'])

        suspicious_features = []
        for momento_key, momento_info in self.timeline_moments.items():
            if momento_info['vazamento'] == 'CRÍTICO':
                suspicious_features.extend(momento_info['informacoes_disponiveis'])

        self.safe_features = list(set(safe_features))
        self.temporal_suspicious = list(set(suspicious_features))

        derived_safe = [f for f in self.derived_features if f in self.safe_features]
        derived_suspicious = [f for f in self.derived_features if f in self.temporal_suspicious]

        logger.info(f"📊 Features seguras: {len(self.safe_features)} ({len(derived_safe)} derivadas)")
        logger.info(f"🚨 Features suspeitas: {len(self.temporal_suspicious)} ({len(derived_suspicious)} derivadas)")

        if derived_suspicious:
            logger.warning(f"⚠️ Features derivadas suspeitas encontradas: {derived_suspicious}")

    def create_clinical_feature_categories(self) -> Dict[str, Dict]:
        """Cria categorização clínica detalhada das features"""

        categories = {
            "CATEGORIA_A_DEMOGRAFICA": {
                "descricao": "Dados demográficos básicos",
                "momento": "Notificação",
                "vazamento": "Nenhum",
                "risco_nivel": "BAIXO",
                "features": [
                    "NU_ANO", "ANO_NASC", "IDADE", "CS_SEXO", "CS_RACA",
                    "CS_ESCOL_N", "SG_UF", "FAIXA_ETARIA"
                ]
            },
            "CATEGORIA_B_CLINICA_INICIAL": {
                "descricao": "Apresentação clínica inicial",
                "momento": "Primeira consulta",
                "vazamento": "Nenhum",
                "risco_nivel": "BAIXO",
                "features": [
                    "FORMA", "RAIOX_TORA", "TESTE_TUBE", "EXTRAPU1_N", "EXTRAPU2_N"
                ]
            },
            "CATEGORIA_C_COMORBIDADES": {
                "descricao": "Comorbidades conhecidas (SUBSTITUÍVEIS)",
                "momento": "Anamnese inicial",
                "vazamento": "Nenhum",
                "risco_nivel": "BAIXO",
                "features": [
                    "HIV", "AGRAVAIDS", "AGRAVALCOO", "AGRAVDIABE", "AGRAVDOENC",
                    "AGRAVDROGA", "AGRAVTABAC", "AGRAVOUTRA"
                ],
                "substitutable_by": "SCORE_COMORBIDADES",
                "substitution_benefit": "Condensa 7 features em 1 score mais interpretável"
            },
            "CATEGORIA_D_SOCIAL": {
                "descricao": "Fatores sociais (SUBSTITUÍVEIS)",
                "momento": "Anamnese inicial",
                "vazamento": "Nenhum",
                "risco_nivel": "BAIXO",
                "features": [
                    "POP_LIBER", "POP_RUA", "POP_IMIG"
                ],
                "substitutable_by": "RISCO_SOCIAL",
                "substitution_benefit": "Condensa fatores de vulnerabilidade social"
            },
            "CATEGORIA_E_EXAMES_INICIAIS": {
                "descricao": "Exames solicitados na consulta inicial",
                "momento": "Primeiros dias",
                "vazamento": "Mínimo",
                "risco_nivel": "BAIXO",
                "features": [
                    "BACILOS_E2", "CULTURA_OU", "TEST_MOLEC",
                    "CRITERIO_LABORATORIAL", "ANT_RETRO"
                ]
            },
            "CATEGORIA_F_INICIO_TRATAMENTO": {
                "descricao": "Informações do início do tratamento (MELHORÁVEIS)",
                "momento": "Primeira semana",
                "vazamento": "Mínimo",
                "risco_nivel": "BAIXO",
                "features": [
                    "DT_INIC_TR", "DIAS_INIC_TRAT"
                ],
                "substitutable_by": "TEMPO_INICIO_CAT",
                "substitution_benefit": "Versão categórica mais robusta e interpretável"
            },
            "CATEGORIA_G_FEATURES_DERIVADAS": {
                "descricao": "Features derivadas - PREFERÍVEIS às originais",
                "momento": "Construídas com dados do momento inicial",
                "vazamento": "NENHUM",
                "risco_nivel": "BAIXO",
                "features": [
                    "SCORE_COMORBIDADES",
                    "CASO_COMPLEXO",
                    "RISCO_SOCIAL",
                    "TEMPO_INICIO_CAT",
                    "PERFIL_GRAVIDADE",
                    "ACESSO_SERVICOS",
                    "DURACAO_PREVISTA_CAT"
                ],
                "derivation_logic": {
                    "SCORE_COMORBIDADES": "Soma binária de todas as comorbidades",
                    "CASO_COMPLEXO": "HIV + forma extrapulmonar + múltiplas comorbidades",
                    "RISCO_SOCIAL": "Soma de populações especiais vulneráveis",
                    "TEMPO_INICIO_CAT": "Categorização clínica do tempo até início",
                    "PERFIL_GRAVIDADE": "Idade crítica + HIV + comorbidades",
                    "ACESSO_SERVICOS": "Região favorável + educação superior",
                    "DURACAO_PREVISTA_CAT": "Protocolo baseado em forma clínica e fatores"
                }
            },
            "CATEGORIA_H_VAZAMENTO_TEMPORAL": {
                "descricao": "VAZAMENTO: Informações descobertas durante/após tratamento",
                "momento": "Durante/após tratamento",
                "vazamento": "CRÍTICO",
                "risco_nivel": "CRÍTICO",
                "features": [
                    "DIAS", "TRATAMENTO", "TEST_SENSI", "DOENCA_TRA", "TRANSF",
                    "RIFAMPICIN", "ISONIAZIDA", "ETAMBUTOL", "ESTREPTOMI",
                    "PIRAZINAMI", "ETIONAMIDA", "OUTRAS", "BACILOSC_1",
                    "BACILOSC_2", "BACILOSC_3", "BACILOSC_4", "BACILOSC_5",
                    "BACILOSC_6", "BAC_APOS_6"
                ]
            },
            "CATEGORIA_I_TARGET": {
                "descricao": "Variável alvo (desfecho final)",
                "momento": "Final do tratamento",
                "vazamento": "CRÍTICO",
                "risco_nivel": "CRÍTICO",
                "features": [
                    "SITUA_ENCE", "DT_ENCERRA"
                ]
            }
        }

        self.feature_categories = categories
        return categories

    def _get_risk_icon(self, risk_level: str) -> str:
        """Retorna ícone baseado no nível de risco"""

        icons = {
           'BAIXO': '✅',
           'MODERADO': '⚠️',
           'ALTO': '🚨',
           'CRÍTICO': '❌'
        }
        return icons.get(risk_level, '⚠️')

    def get_temporal_suspicious_features(self) -> List[str]:
        """Retorna lista de features temporalmente suspeitas"""
        return self.temporal_suspicious.copy()

    def get_safe_features(self) -> List[str]:
        """Retorna lista de features seguras para uso preditivo"""
        return self.safe_features.copy()

    def get_derived_features(self) -> List[str]:
        """Retorna lista de features derivadas"""
        return self.derived_features.copy()

    def get_replaced_features(self) -> List[str]:
        """Retorna lista de features que foram substituídas"""
        return list(self.replaced_features)

    def get_feature_risk_assessment(self, feature_name: str) -> Dict[str, str]:
        """Avalia risco de uma feature específica"""

        if feature_name in self.derived_features:
            return {
                'feature': feature_name,
                'category': 'Features Derivadas',
                'moment': 'Construída com dados iniciais',
                'leakage': 'Nenhum',
                'risk_level': 'BAIXO',
                'recommendation': '✅ USAR (PREFERÍVEL)',
                'derivation_info': self._get_derivation_info(feature_name)
            }

        if feature_name in self.replaced_features:
            replacing_feature = None
            for derived, mapping in self.replacement_mappings.items():
                if feature_name in mapping.get('replaces', []):
                    replacing_feature = derived
                    break

            return {
                'feature': feature_name,
                'category': 'Feature Original (Substituída)',
                'moment': 'Disponível inicialmente mas substituída',
                'leakage': 'Nenhum',
                'risk_level': 'BAIXO',
                'recommendation': f'⚠️ USAR {replacing_feature} PREFERENCIALMENTE',
                'replacement_info': f'Substituída por {replacing_feature}' if replacing_feature else 'Substituída'
            }

        for cat_key, cat_info in self.feature_categories.items():
            if feature_name in cat_info['features']:
                return {
                    'feature': feature_name,
                    'category': cat_info['descricao'],
                    'moment': cat_info['momento'],
                    'leakage': cat_info['vazamento'],
                    'risk_level': cat_info['risco_nivel'],
                    'recommendation': '✅ USAR' if cat_info['risco_nivel'] == 'BAIXO' else '❌ NÃO USAR'
                }

        return {
            'feature': feature_name,
            'category': 'Desconhecida',
            'moment': 'Indefinido',
            'leakage': 'Desconhecido',
            'risk_level': 'ALTO',
            'recommendation': '⚠️ INVESTIGAR'
        }

    def _get_derivation_info(self, feature_name: str) -> str:
        """Retorna informação sobre como a feature derivada foi criada"""

        derivation_info = {
            'SCORE_COMORBIDADES': 'Soma binária de AGRAVAIDS + AGRAVALCOO + AGRAVDIABE + AGRAVDOENC + AGRAVDROGA + AGRAVTABAC + AGRAVOUTRA',
            'CASO_COMPLEXO': 'Binário baseado em forma extrapulmonar OU HIV positivo OU múltiplas comorbidades (≥2)',
            'RISCO_SOCIAL': 'Soma binária de POP_LIBER + POP_RUA + POP_IMIG',
            'TEMPO_INICIO_CAT': 'DIAS_INIC_TRAT categorizado: 0=imediato, 1=≤7dias, 2=≤30dias, 3=≤60dias, 4=>60dias',
            'PERFIL_GRAVIDADE': 'Soma de idade≤2anos + HIV positivo + comorbidades≥1',
            'ACESSO_SERVICOS': 'Soma de região Sul/Sudeste + educação superior',
            'DURACAO_PREVISTA_CAT': 'Protocolo: 0=6meses(pulmonar), 1=9meses(extrapulmonar), 2=12meses(mista/HIV/complicada)'
        }
        return derivation_info.get(feature_name, 'Informação de derivação não disponível')

    def save_timeline_report(self, filepath: str):
        """Salva relatório da análise temporal em arquivo"""

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("RELATÓRIO DE ANÁLISE TEMPORAL CLÍNICA\n")
                f.write("Sistema IA Tuberculose Infantil - COM FEATURES DERIVADAS\n")
                f.write(f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
                f.write("="*80 + "\n\n")

                f.write("0. STATUS DO SISTEMA DE SUBSTITUIÇÃO\n")
                f.write("-"*40 + "\n")
                f.write(f"Features derivadas criadas: {len(self.derived_features)}\n")
                f.write(f"Features originais substituídas: {len(self.replaced_features)}\n")
                f.write(f"Mapeamentos de substituição: {len(self.replacement_mappings)}\n\n")

                if self.derived_features:
                    f.write("Features derivadas:\n")
                    for feature in self.derived_features:
                        derivation = self._get_derivation_info(feature)
                        f.write(f"  ✅ {feature}\n")
                        f.write(f"     Derivação: {derivation}\n")

                if self.replaced_features:
                    f.write(f"\nFeatures substituídas:\n")
                    for feature in self.replaced_features:
                        f.write(f"  🔄 {feature}\n")

                f.write(f"\n1. LINHA TEMPORAL CLÍNICA\n")
                f.write("-"*40 + "\n")
                for momento_key, momento_info in self.timeline_moments.items():
                    f.write(f"\n{momento_info['quando'].upper()}\n")
                    f.write(f"Tempo: {momento_info['tempo']}\n")
                    f.write(f"Vazamento: {momento_info['vazamento']}\n")
                    f.write(f"Risco: {momento_info['risco']}\n")
                    f.write(f"Uso clínico: {momento_info['uso_clinico']}\n")
                    f.write(f"Features ({len(momento_info['informacoes_disponiveis'])}):\n")

                    for feature in momento_info['informacoes_disponiveis']:
                        if feature in self.derived_features:
                            f.write(f"  🔄 {feature} (DERIVADA)\n")
                        elif feature in self.replaced_features:
                            f.write(f"  📊 {feature} (SUBSTITUÍDA)\n")
                        else:
                            f.write(f"  - {feature}\n")

                f.write(f"\n\n2. CATEGORIZAÇÃO DE FEATURES\n")
                f.write("-"*40 + "\n")
                for cat_key, cat_info in self.feature_categories.items():
                    f.write(f"\n{cat_info['descricao'].upper()}\n")
                    f.write(f"Momento: {cat_info['momento']}\n")
                    f.write(f"Vazamento: {cat_info['vazamento']}\n")
                    f.write(f"Risco: {cat_info['risco_nivel']}\n")

                    if 'substitutable_by' in cat_info:
                        f.write(f"Pode ser substituída por: {cat_info['substitutable_by']}\n")
                        f.write(f"Benefício: {cat_info['substitution_benefit']}\n")

                    f.write(f"Features ({len(cat_info['features'])}):\n")
                    for feature in cat_info['features']:
                        if feature in self.derived_features:
                            f.write(f"  🔄 {feature} (DERIVADA)\n")
                        elif feature in self.replaced_features:
                            f.write(f"  📊 {feature} (SUBSTITUÍDA)\n")
                        else:
                            f.write(f"  - {feature}\n")

                f.write(f"\n\n3. RESUMO EXECUTIVO\n")
                f.write("-"*40 + "\n")
                f.write(f"Features seguras para IA: {len(self.safe_features)}\n")
                f.write(f"Features com vazamento temporal: {len(self.temporal_suspicious)}\n")
                f.write(f"Features derivadas (preferíveis): {len(self.derived_features)}\n")
                f.write(f"Features substituídas: {len(self.replaced_features)}\n")
                f.write(f"Total de categorias: {len(self.feature_categories)}\n")

            logger.info(f"✅ Relatório temporal salvo em: {filepath}")

        except Exception as e:
            logger.error(f"❌ Erro ao salvar relatório temporal: {e}")

    def validate_features_in_data(self, data_columns: List[str]) -> Dict[str, Any]:
        """Valida quais features estão presentes nos dados"""

        validation_results = {
            'total_columns': len(data_columns),
            'safe_found': 0,
            'suspicious_found': 0,
            'derived_found': 0,
            'replaced_found': 0,
            'missing_safe': [],
            'missing_suspicious': [],
            'missing_derived': [],
            'unknown_features': [],
            'replacement_analysis': {}
        }

        for feature in self.safe_features:
            if feature in data_columns:
                validation_results['safe_found'] += 1
            else:
                validation_results['missing_safe'].append(feature)

        for feature in self.temporal_suspicious:
            if feature in data_columns:
                validation_results['suspicious_found'] += 1
            else:
                validation_results['missing_suspicious'].append(feature)

        for feature in self.derived_features:
            if feature in data_columns:
                validation_results['derived_found'] += 1
            else:
                validation_results['missing_derived'].append(feature)

        for feature in self.replaced_features:
            if feature in data_columns:
                validation_results['replaced_found'] += 1

        for derived_feature, mapping in self.replacement_mappings.items():
            original_features = mapping.get('replaces', [])
            derived_present = derived_feature in data_columns
            originals_present = [f for f in original_features if f in data_columns]

            validation_results['replacement_analysis'][derived_feature] = {
                'derived_present': derived_present,
                'originals_present': originals_present,
                'replacement_complete': derived_present and len(originals_present) == 0,
                'replacement_partial': derived_present and len(originals_present) > 0,
                'replacement_missing': not derived_present and len(originals_present) > 0
            }

        all_known_features = set(self.safe_features + self.temporal_suspicious + self.derived_features)
        for column in data_columns:
            if column not in all_known_features:
                validation_results['unknown_features'].append(column)

        logger.info(f"📊 VALIDAÇÃO DE FEATURES NOS DADOS:")
        logger.info(f"✅ Features seguras: {validation_results['safe_found']}/{len(self.safe_features)}")
        logger.info(f"🔄 Features derivadas: {validation_results['derived_found']}/{len(self.derived_features)}")
        logger.info(f"📊 Features substituídas: {validation_results['replaced_found']}/{len(self.replaced_features)}")
        logger.info(f"🚨 Features suspeitas: {validation_results['suspicious_found']}/{len(self.temporal_suspicious)}")
        logger.info(f"❓ Features desconhecidas: {len(validation_results['unknown_features'])}")

        return validation_results