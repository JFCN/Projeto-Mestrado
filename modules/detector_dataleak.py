"""
Módulo 2: Detector Básico de Data Leakage
Detecção inicial e triagem de vazamentos óbvios

Responsabilidades:
- Identifica features temporalmente suspeitas por padrões
- Análise de poder preditivo individual (F-score)
- Teste com modelo simples (com vs sem features suspeitas)
- Análise de padrões de missing values vs desfecho
- Relatório básico de triagem para validação avançada

Complementa o detector avançado (analisador_final.py)
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from datetime import datetime
from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger('DataLeakageDetector')

class DataLeakageDetector:
    """
    Detector de data leakage
    """

    def __init__(self):
        self.data = None
        self.suspicious_features = []
        self.feature_importance = {}
        self.temporal_analysis = {}

        self.suspicious_patterns = [
            'BACILOSC_',
            'BAC_APOS',
            'TRANSF',
            'DIAS',
            '_APOS_',
            'RESULTADO',
            'DESFECHO',
            'ENCERRA',
            'TRAT',
            'TEST_SENSI',
            'DOENCA_TRA',
            'RIFAMPICIN',
            'ISONIAZIDA',
            'ETAMBUTOL',
            'ESTREPTOMI',
            'PIRAZINAMI',
            'ETIONAMIDA'
        ]

        self.explicitly_safe_patterns = [
            'DURACAO_PREVISTA_CAT',
            'SCORE_COMORBIDADES',
            'CASO_COMPLEXO',
            'RISCO_SOCIAL',
            'PERFIL_GRAVIDADE',
            'ACESSO_SERVICOS'
        ]
        
    def load_and_preprocess(self, file_path: str) -> bool:
        """Carrega e processa dados básicos"""

        try:
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path)
            else:
                self.data = pd.read_excel(file_path)

            if 'SITUA_ENCE' not in self.data.columns:
                logger.error("❌ Coluna 'SITUA_ENCE' não encontrada")
                return False
            
            target_mapping = {1: 1, 2: 0}  # Cura = 1, Óbito = 0
            self.data['TARGET'] = self.data['SITUA_ENCE'].map(target_mapping)
            self.data = self.data.dropna(subset=['TARGET'])
            
            logger.info(f"✅ Target criado: {self.data['TARGET'].value_counts().to_dict()}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar dados: {e}")
            return False

    def identify_suspicious_temporal_features(self) -> List[str]:
        """Identifica features temporalmente suspeitas"""

        logger.info("\n" + "=" * 60)
        logger.info("🚨 IDENTIFICAÇÃO DE FEATURES TEMPORALMENTE SUSPEITAS")
        logger.info("=" * 60)

        temporal_features = []

        for col in self.data.columns:
            is_safe = any(safe_pattern in col.upper()
                          for safe_pattern in self.explicitly_safe_patterns)

            if is_safe:
                continue

            for pattern in self.suspicious_patterns:
                if pattern.upper() in col.upper():
                    temporal_features.append(col)
                    break

        temporal_features = list(set(temporal_features))
        system_cols = ['TARGET', 'SITUA_ENCE', 'DT_NOTI_AT']
        temporal_features = [f for f in temporal_features if f not in system_cols]

        if 'DIAS' in temporal_features:
            logger.warning("🚨 'DIAS' (duração real) identificada como suspeita")

        if 'DURACAO_PREVISTA_CAT' not in temporal_features:
            logger.info("✅ 'DURACAO_PREVISTA_CAT' (duração prevista) é SEGURA")

        logger.info(f"🚨 FEATURES TEMPORALMENTE SUSPEITAS ({len(temporal_features)}):")
        for feature in sorted(temporal_features):
            missing_pct = self.data[feature].isnull().mean() * 100
            logger.info(f"  📋 {feature}: {missing_pct:.1f}% missing")

        self.suspicious_features = temporal_features
        return temporal_features
    
    def analyze_feature_predictive_power(self) -> pd.DataFrame:
        """Analisa poder preditivo individual de cada feature"""
        
        logger.info("\n" + "="*60)
        logger.info("📈 ANÁLISE DE PODER PREDITIVO DAS FEATURES")
        logger.info("="*60)
        
        exclude_cols = ['TARGET', 'SITUA_ENCE']
        X = self.data.drop(exclude_cols, axis=1, errors='ignore')
        y = self.data['TARGET']
        
        X_processed = X.copy()
        for col in X_processed.columns:
            if X_processed[col].dtype == 'object':
                X_processed[col] = X_processed[col].fillna('Missing')
                le = LabelEncoder()
                X_processed[col] = le.fit_transform(X_processed[col].astype(str))
            else:
                X_processed[col] = X_processed[col].fillna(X_processed[col].median())
        
        selector = SelectKBest(score_func=f_classif, k='all')
        selector.fit(X_processed, y)
        
        feature_scores = pd.DataFrame({
            'feature': X_processed.columns,
            'f_score': selector.scores_,
            'p_value': selector.pvalues_
        }).sort_values('f_score', ascending=False)
        
        logger.info(f"📈 TOP 15 FEATURES MAIS PREDITIVAS:")
        for _, row in feature_scores.head(15).iterrows():
            suspicious_marker = "🚨" if row['feature'] in self.suspicious_features else "✅"
            logger.info(f"  {suspicious_marker} {row['feature']}: F-score = {row['f_score']:.1f}")
        
        extremely_predictive = feature_scores[feature_scores['f_score'] > 1000]['feature'].tolist()
        
        if extremely_predictive:
            logger.warning(f"\n🚨 FEATURES EXCESSIVAMENTE PREDITIVAS (F-score > 1000):")
            for feature in extremely_predictive:
                score = feature_scores[feature_scores['feature'] == feature]['f_score'].iloc[0]
                logger.warning(f"  🔥 {feature}: F-score = {score:.1f}")
        
        self.feature_importance = feature_scores
        return feature_scores
    
    def test_leakage_with_simple_model(self) -> Dict[str, Any]:
        """Testa vazamento com modelo simples"""
        
        logger.info(f"\n" + "="*60)
        logger.info(f"🧪 TESTE COM MODELO SIMPLES")
        logger.info(f"="*60)
        
        exclude_cols = ['TARGET', 'SITUA_ENCE']
        safe_features = [col for col in self.data.columns 
                        if col not in self.suspicious_features + exclude_cols]
        
        if not safe_features:
            logger.warning("⚠️ Nenhuma feature segura encontrada!")
            return {'error': 'No safe features'}
        
        X_safe = self.data[safe_features].copy()
        X_all = self.data.drop(exclude_cols, axis=1, errors='ignore')
        y = self.data['TARGET']
        
        datasets = {'SAFE': X_safe, 'ALL': X_all}
        processed_datasets = {}
        
        for name, X in datasets.items():
            X_processed = X.copy()
            for col in X_processed.columns:
                if X_processed[col].dtype == 'object':
                    X_processed[col] = X_processed[col].fillna('Missing')
                    le = LabelEncoder()
                    X_processed[col] = le.fit_transform(X_processed[col].astype(str))
                else:
                    X_processed[col] = X_processed[col].fillna(X_processed[col].median())
            processed_datasets[name] = X_processed
        
        results = {}
        
        for name, X in processed_datasets.items():
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42, stratify=y
                )
                
                model = RandomForestClassifier(
                    n_estimators=50,
                    max_depth=5,
                    random_state=42
                )
                
                model.fit(X_train, y_train)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_pred_proba)
                
                results[name] = {
                    'auc': auc,
                    'n_features': X.shape[1],
                    'features': list(X.columns)
                }
                
                dataset_desc = "APENAS FEATURES SEGURAS" if name == 'SAFE' else "TODAS AS FEATURES"
                logger.info(f"📊 {dataset_desc}:")
                logger.info(f"  Features: {X.shape[1]}")
                logger.info(f"  AUC: {auc:.3f}")
                
            except Exception as e:
                logger.error(f"❌ Erro ao testar {name}: {e}")
                results[name] = {'auc': 0.5, 'n_features': 0, 'features': []}
        
        if 'SAFE' in results and 'ALL' in results:
            auc_safe = results['SAFE']['auc']
            auc_all = results['ALL']['auc']
            auc_diff = auc_all - auc_safe
            
            logger.info(f"\n📊 COMPARAÇÃO DE PERFORMANCE:")
            logger.info(f"  AUC com features seguras: {auc_safe:.3f}")
            logger.info(f"  AUC com todas as features: {auc_all:.3f}")
            logger.info(f"  Diferença: {auc_diff:.3f}")
            
            if auc_diff > 0.15:
                risk_level = "CRITICAL"
                logger.error(f"🚨 VAZAMENTO CRÍTICO! Diferença de AUC > 0.15")
            elif auc_diff > 0.10:
                risk_level = "HIGH"
                logger.warning(f"⚠️ VAZAMENTO ALTO! Diferença de AUC > 0.10")
            elif auc_diff > 0.05:
                risk_level = "MODERATE"
                logger.warning(f"⚡ VAZAMENTO MODERADO! Diferença de AUC > 0.05")
            else:
                risk_level = "LOW"
                logger.info(f"✅ Baixo risco de vazamento")
            
            results['comparison'] = {
                'auc_difference': auc_diff,
                'risk_level': risk_level,
                'interpretation': self._interpret_auc_difference(auc_diff)
            }
        
        return results

    def analyze_missing_patterns(self) -> pd.DataFrame:
        """Analisa padrões de dados ausentes que podem indicar leakage"""

        logger.info(f"\n" + "=" * 60)
        logger.info(f"🕳️ ANÁLISE DE PADRÕES DE MISSING VALUES")
        logger.info(f"=" * 60)

        missing_analysis = []

        for col in self.data.columns:
            if col in ['TARGET', 'SITUA_ENCE']:
                continue

            missing_pct = self.data[col].isnull().mean() * 100

            if missing_pct > 0:
                missing_mask = self.data[col].isnull()

                if missing_mask.sum() > 0:
                    survival_rate_missing = self.data[missing_mask]['TARGET'].mean()
                else:
                    survival_rate_missing = 0.0

                if (~missing_mask).sum() > 0:
                    survival_rate_present = self.data[~missing_mask]['TARGET'].mean()
                else:
                    survival_rate_present = 0.0

                difference = abs(survival_rate_missing - survival_rate_present)

                missing_analysis.append({
                    'feature': col,
                    'missing_pct': missing_pct,
                    'survival_rate_missing': survival_rate_missing,
                    'survival_rate_present': survival_rate_present,
                    'difference': difference,
                    'suspicious': col in self.suspicious_features
                })

        if not missing_analysis:
            logger.info("📊 Nenhum padrão de missing significativo encontrado")
            return pd.DataFrame()

        missing_df = pd.DataFrame(missing_analysis)

        if missing_df.empty:
            logger.info("📊 DataFrame de análise vazio")
            return missing_df

        missing_df = missing_df.sort_values('difference', ascending=False)

        if not missing_df.empty:
            logger.info(f"🔍 TOP 10 FEATURES COM MAIOR DIFERENÇA DE DESFECHO:")
            for _, row in missing_df.head(10).iterrows():
                suspicious_marker = "🚨" if row['suspicious'] else "📊"
                logger.info(
                    f"  {suspicious_marker} {row['feature']}: diferença = {row['difference']:.3f}, missing = {row['missing_pct']:.1f}%")

            highly_differential = missing_df[missing_df['difference'] > 0.15]['feature'].tolist()

            if highly_differential:
                logger.warning(f"\n🚨 FEATURES COM PADRÃO CRÍTICO DE AUSÊNCIA:")
                for feature in highly_differential:
                    row = missing_df[missing_df['feature'] == feature].iloc[0]
                    logger.warning(f"  🔥 {feature}: {row['difference']:.3f} diferença de sobrevivência")

        return missing_df
    
    def _interpret_auc_difference(self, auc_diff: float) -> str:
        """Interpreta a diferença de AUC"""

        if auc_diff > 0.15:
            return "Vazamento crítico - features suspeitas dominam completamente a predição"
        elif auc_diff > 0.10:
            return "Vazamento alto - features suspeitas contribuem significativamente"
        elif auc_diff > 0.05:
            return "Vazamento moderado - alguma influência de features suspeitas"
        else:
            return "Baixo risco - diferença dentro do esperado"
    
    def generate_leakage_report(self) -> Dict[str, Any]:
        """Gera relatório completo de vazamento básico"""
        
        logger.info(f"\n" + "="*70)
        logger.info(f"📋 RELATÓRIO BÁSICO DE DATA LEAKAGE")
        logger.info("="*70)
        
        temporal_features = self.identify_suspicious_temporal_features()
        
        feature_scores = self.analyze_feature_predictive_power()
        
        model_results = self.test_leakage_with_simple_model()
        
        missing_patterns = self.analyze_missing_patterns()
        
        overall_risk = self._determine_overall_risk(model_results, feature_scores, missing_patterns)
        
        recommendations = self._generate_recommendations(overall_risk, temporal_features, model_results)
        
        results = {
            'risk_level': overall_risk,
            'temporal_features': temporal_features,
            'feature_scores': feature_scores,
            'model_results': model_results,
            'missing_patterns': missing_patterns,
            'recommendations': recommendations,
            'analysis_timestamp': datetime.now()
        }
        
        self._print_summary_report(results)
        
        return results
    
    def _determine_overall_risk(self, model_results: Dict, feature_scores: pd.DataFrame, 
                               missing_patterns: pd.DataFrame) -> str:
        """Determina o risco geral de vazamento"""
        
        risk_factors = []
        
        if 'comparison' in model_results:
            auc_diff = model_results['comparison']['auc_difference']
            if auc_diff > 0.15:
                risk_factors.append('CRITICAL')
            elif auc_diff > 0.10:
                risk_factors.append('HIGH')
            elif auc_diff > 0.05:
                risk_factors.append('MODERATE')
        
        if not feature_scores.empty:
            extremely_predictive = len(feature_scores[feature_scores['f_score'] > 1000])
            if extremely_predictive > 10:
                risk_factors.append('CRITICAL')
            elif extremely_predictive > 5:
                risk_factors.append('HIGH')
            elif extremely_predictive > 2:
                risk_factors.append('MODERATE')
        
        if not missing_patterns.empty:
            critical_missing = len(missing_patterns[missing_patterns['difference'] > 0.15])
            if critical_missing > 5:
                risk_factors.append('HIGH')
            elif critical_missing > 2:
                risk_factors.append('MODERATE')
        
        temporal_count = len(self.suspicious_features)
        if temporal_count > 20:
            risk_factors.append('HIGH')
        elif temporal_count > 10:
            risk_factors.append('MODERATE')
        
        if 'CRITICAL' in risk_factors:
            return 'CRITICAL'
        elif 'HIGH' in risk_factors:
            return 'HIGH'
        elif 'MODERATE' in risk_factors:
            return 'MODERATE'
        else:
            return 'LOW'
    
    def _generate_recommendations(self, risk_level: str, temporal_features: List[str], 
                                 model_results: Dict) -> List[str]:
        """Gera recomendações baseadas no nível de risco"""
        
        recommendations = []
        
        if risk_level == 'CRITICAL':
            recommendations.extend([
                "🚨 VAZAMENTO CRÍTICO DETECTADO na triagem básica!",
                "❌ Recomenda-se NÃO prosseguir para validação avançada sem correções",
                "🔧 Remover TODAS as features temporalmente suspeitas",
                "📊 Revisar processo de coleta de dados",
                "⚠️ Considerar redesenho do dataset"
            ])
        elif risk_level == 'HIGH':
            recommendations.extend([
                "⚠️ RISCO ALTO de vazamento detectado na triagem",
                "🔍 PROCEDER com validação avançada para confirmar",
                "🚨 Remover features mais suspeitas antes da validação avançada",
                "📊 Monitorar resultados da validação temporal rigorosa"
            ])
        elif risk_level == 'MODERATE':
            recommendations.extend([
                "⚡ RISCO MODERADO detectado na triagem",
                "✅ PROSSEGUIR para validação avançada",
                "🔍 Focar validação avançada nas features suspeitas identificadas",
                "📊 Acompanhar testes de degradação temporal"
            ])
        else:
            recommendations.extend([
                "✅ BAIXO RISCO na triagem básica",
                "🚀 PROSSEGUIR confiante para validação avançada",
                "📊 Validação avançada deve confirmar baixo risco",
                "🎉 Dados aparentam estar bem estruturados"
            ])

        if temporal_features:
            top_suspicious = temporal_features[:5]  # Top 5
            recommendations.append(f"🎯 Features prioritárias para investigação: {', '.join(top_suspicious)}")
        
        if 'comparison' in model_results:
            auc_diff = model_results['comparison']['auc_difference']
            if auc_diff > 0.05:
                recommendations.append(f"📈 Diferença de AUC detectada: {auc_diff:.3f} - investigar na validação avançada")
        
        return recommendations
    
    def _print_summary_report(self, results: Dict[str, Any]):
        """Imprime resumo final do relatório básico"""
        
        logger.info(f"\n🎯 RESUMO DA TRIAGEM BÁSICA:")
        logger.info(f"🚨 Nível de Risco: {results['risk_level']}")
        logger.info(f"📊 Features temporalmente suspeitas: {len(results['temporal_features'])}")
        
        if 'comparison' in results['model_results']:
            auc_diff = results['model_results']['comparison']['auc_difference']
            logger.info(f"📈 Diferença de AUC: {auc_diff:.3f}")
        
        logger.info(f"\n📋 RECOMENDAÇÕES PARA PRÓXIMA ETAPA:")
        for rec in results['recommendations']:
            logger.info(f"  {rec}")
        
        if results['risk_level'] in ['LOW', 'MODERATE']:
            logger.info(f"\n✅ PRONTO PARA VALIDAÇÃO AVANÇADA")

        elif results['risk_level'] == 'HIGH':
            logger.warning(f"\n⚠️ PROSSEGUIR COM CAUTELA PARA VALIDAÇÃO AVANÇADA")
        else:
            logger.error(f"\n🚨 RECOMENDA-SE CORREÇÃO ANTES DA VALIDAÇÃO AVANÇADA")

    def save_detailed_report(self, filepath: str, results: Dict[str, Any]):
        """Salva relatório detalhado em arquivo"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("RELATÓRIO BÁSICO DE DETECÇÃO DE DATA LEAKAGE\n")
                f.write("Sistema IA Tuberculose Infantil - Triagem Inicial\n")
                f.write(f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
                f.write("="*80 + "\n\n")
                
                f.write("1. RESUMO EXECUTIVO\n")
                f.write("-"*40 + "\n")
                f.write(f"Nível de risco detectado: {results['risk_level']}\n")
                f.write(f"Features temporalmente suspeitas: {len(results['temporal_features'])}\n")
                f.write(f"Data da análise: {results['analysis_timestamp'].strftime('%d/%m/%Y %H:%M')}\n\n")
                
                f.write("2. FEATURES TEMPORALMENTE SUSPEITAS\n")
                f.write("-"*40 + "\n")
                if results['temporal_features']:
                    for feature in results['temporal_features']:
                        f.write(f"  - {feature}\n")
                else:
                    f.write("  Nenhuma feature suspeita identificada por padrões.\n")
                
                f.write(f"\n3. ANÁLISE DE PODER PREDITIVO\n")
                f.write("-"*40 + "\n")
                if not results['feature_scores'].empty:
                    f.write("Top 10 features mais preditivas:\n")
                    for _, row in results['feature_scores'].head(10).iterrows():
                        f.write(f"  {row['feature']}: F-score = {row['f_score']:.1f}\n")
                
                f.write(f"\n4. TESTE COM MODELO SIMPLES\n")
                f.write("-"*40 + "\n")
                if 'comparison' in results['model_results']:
                    comp = results['model_results']['comparison']
                    f.write(f"AUC com features seguras: {results['model_results']['SAFE']['auc']:.3f}\n")
                    f.write(f"AUC com todas as features: {results['model_results']['ALL']['auc']:.3f}\n")
                    f.write(f"Diferença de AUC: {comp['auc_difference']:.3f}\n")
                    f.write(f"Interpretação: {comp['interpretation']}\n")
                
                f.write(f"\n5. RECOMENDAÇÕES\n")
                f.write("-"*40 + "\n")
                for rec in results['recommendations']:
                    f.write(f"  {rec}\n")
                
                f.write(f"\n6. PRÓXIMOS PASSOS\n")
                f.write("-"*40 + "\n")
                if results['risk_level'] in ['LOW', 'MODERATE']:
                    f.write("✅ PROSSEGUIR para validação avançada (analisador_final.py)\n")
                    f.write("🔍 A validação avançada executará 5 testes rigorosos\n")
                elif results['risk_level'] == 'HIGH':
                    f.write("⚠️ PROSSEGUIR COM CAUTELA para validação avançada\n")
                    f.write("🔍 Monitorar resultados da validação temporal rigorosa\n")
                else:
                    f.write("🚨 RECOMENDA-SE correção dos dados antes de prosseguir\n")
                    f.write("💡 Remover features mais suspeitas e reexecutar triagem\n")
            
            logger.info(f"✅ Relatório básico salvo em: {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Erro ao salvar relatório básico: {e}")
    
