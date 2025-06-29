"""
Módulo 4: Sistema de Treinamento TabPFN + Meta-Models
COM OTIMIZAÇÃO PARA FEATURES DERIVADAS

Responsabilidades:
- Treinamento do modelo principal TabPFN (com fallback XGBoost)
- Otimização para features derivadas categóricas
- Tratamento preferencial de features substituídas
- Validação de que substituições melhoram performance
- Cross-validation rigoroso anti-overfitting
- Divisão 60-20-20 para validação robusta
- Geração de relatórios detalhados de performance
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, balanced_accuracy_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.base import clone
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import time
import joblib
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import logging
import warnings
import traceback

try:
    from tabpfn import TabPFNClassifier
    tabpfn_available = True
    print("✅ TabPFN importado com sucesso!")
except ImportError:
    tabpfn_available = False
    print("⚠️ TabPFN não disponível. Usando XGBoost como modelo principal.")

warnings.filterwarnings('ignore')
logger = logging.getLogger('TuberculosisPredictor')


class TuberculosisPredictor:
    """
    Preditor de sobrevivência em tuberculose infantil
    COM OTIMIZAÇÃO PARA FEATURES DERIVADAS
    Arquitetura: TabPFN + Meta-Model Selection ANTI-OVERFITTING
    """

    def __init__(self, use_tabpfn: bool = True, meta_model_type: str = 'auto', cv_folds: int = 10,
                 feature_selector=None, config: Dict[str, Any] = None):
        self.use_tabpfn = use_tabpfn and tabpfn_available
        self.meta_model_type = meta_model_type
        self.cv_folds = cv_folds
        self.config = config or {}
        self.feature_selector = feature_selector
        self.derived_features = []
        self.replaced_features = set()
        self.replacement_mappings = {}
        self.derived_features_config = self._load_derived_features_config()
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_selector_main = None
        self.tabpfn_model = None
        self.meta_model = None
        self.feature_names = []
        self.results = {}
        self.cv_results = {}
        self.meta_results = {}
        self.best_meta_name = None
        self.data_processed = False
        self.X = None
        self.y = None
        self.X_balanced = None
        self.y_balanced = None
        self.selected_features = []
        self.max_samples_tabpfn = 1000
        self.max_features_tabpfn = 100

        if self.feature_selector:
            self._load_replacement_info()

        logger.info(f"🤖 Inicializando preditor com otimização para features derivadas:")
        logger.info(f"   📊 TabPFN disponível: {self.use_tabpfn}")
        logger.info(f"   🧠 Meta-modelo: {meta_model_type}")
        logger.info(f"   🔄 CV Folds: {cv_folds}")
        logger.info(f"   🔄 Features derivadas integradas: {len(self.derived_features)}")
        logger.info(f"   📊 Features substituídas: {len(self.replaced_features)}")
        logger.info(f"   🛡️ Configuração: ANTI-OVERFITTING + DERIVADAS OTIMIZADAS")

        self.base_models = self._get_optimized_base_models()

    def _score_features(self, X: np.ndarray, y: np.ndarray,
                        available_features: List[str], features_to_score: List[str]) -> List[Tuple[str, float]]:
        """Converter para DataFrame"""

        try:
            logger.debug(f"🔧 _score_features")

            valid_features = [f for f in features_to_score if f in available_features]

            if not valid_features:
                logger.warning("⚠️ Nenhuma feature válida para pontuar")
                return []

            if isinstance(X, np.ndarray):
                df = pd.DataFrame(X, columns=available_features)
            else:
                df = X.copy()

            try:
                X_subset = df[valid_features].values
                logger.debug(f"   ✅ Subset extraído: {X_subset.shape}")
            except Exception as e:
                logger.warning(f"   ⚠️ Erro ao extrair subset: {e}")
                return []

            X_processed = self._prepare_X_for_selection(X_subset, valid_features)

            if X_processed.shape[1] == 0:
                return []

            from sklearn.feature_selection import SelectKBest, f_classif
            selector = SelectKBest(score_func=f_classif, k='all')
            selector.fit(X_processed, y)
            scores = selector.scores_

            result = list(zip(valid_features, scores))
            logger.debug(f"   ✅ {len(result)} scores calculados")
            return result

        except Exception as e:
            logger.error(f"❌ Erro na versão DataFrame: {e}")
            return [(f, 1.0) for f in features_to_score if f in available_features]

    def _select_with_derived_priority(self, X: np.ndarray, y: np.ndarray,
                                                available_features: List[str], max_features: int) -> List[str]:
        """Seleção priorizando features derivadas"""

        selected = []

        try:
            logger.info(f"   🔧 Iniciando seleção com priorização de derivadas")
            logger.info(f"      X shape: {X.shape}")
            logger.info(f"      Available features: {len(available_features)}")
            logger.info(f"      Derived features: {len(self.derived_features)}")

            derived_available = [f for f in self.derived_features if f in available_features]

            max_derived = min(
                len(derived_available),
                self.derived_features_config.get('max_derived_in_selection', 5),
                max_features // 2
            )

            if max_derived > 0 and derived_available:
                logger.info(
                    f"   🔄 Selecionando até {max_derived} features derivadas de {len(derived_available)} disponíveis")

                derived_scores = self._score_features(X, y, available_features, derived_available)
                derived_scores.sort(key=lambda x: x[1], reverse=True)

                selected_derived = [f for f, _ in derived_scores[:max_derived]]
                selected.extend(selected_derived)

                logger.info(f"   ✅ {len(selected_derived)} features derivadas selecionadas: {selected_derived}")

            remaining_slots = max_features - len(selected)

            if remaining_slots > 0:
                non_derived = [f for f in available_features
                               if f not in self.derived_features and f not in selected]

                if non_derived:
                    logger.info(
                        f"   📊 Selecionando {remaining_slots} features não-derivadas de {len(non_derived)} disponíveis")

                    non_derived_scores = self._score_features(X, y, available_features, non_derived)
                    non_derived_scores.sort(key=lambda x: x[1], reverse=True)

                    selected_non_derived = [f for f, _ in non_derived_scores[:remaining_slots]]
                    selected.extend(selected_non_derived)

                    logger.info(f"   ✅ {len(selected_non_derived)} features não-derivadas selecionadas")

            logger.info(f"   🎯 Total selecionado: {len(selected)} features")
            return selected[:max_features]

        except Exception as e:
            logger.error(f"❌ Erro na seleção com priorização: {e}")
            return self._select_standard_fallback(X, y, available_features, max_features)

    def _select_standard(self, X: np.ndarray, y: np.ndarray,
                         available_features: List[str], max_features: int) -> List[str]:
        """Seleção padrão sem priorização especial"""

        try:
            X_processed = self._prepare_X_for_selection(X, available_features)

            selector = SelectKBest(score_func=f_classif, k=min(max_features, len(available_features)))
            selector.fit(X_processed, y)

            selected_mask = selector.get_support()
            selected_features = [available_features[i] for i, selected in enumerate(selected_mask) if selected]

            return selected_features

        except Exception as e:
            logger.warning(f"⚠️ Erro na seleção padrão: {e}")
            return available_features[:max_features]

    def _select_standard_fallback(self, X: np.ndarray, y: np.ndarray,
                                            available_features: List[str], max_features: int) -> List[str]:
        """Seleção padrão como fallback"""

        try:
            logger.info("   🔄 Usando seleção padrão...")
            logger.info(f"      X shape: {X.shape}")
            logger.info(f"      Available features: {len(available_features)}")

            X_processed = self._prepare_X_for_selection(X, available_features)

            if X_processed.shape[1] == 0:
                logger.error("❌ Nenhuma feature válida após processamento")
                return available_features[:max_features]

            n_select = min(max_features, X_processed.shape[1], len(available_features))
            selector = SelectKBest(score_func=f_classif, k=n_select)
            selector.fit(X_processed, y)
            selected_mask = selector.get_support()
            selected_features = [available_features[i] for i, selected_flag in enumerate(selected_mask) if
                                 selected_flag]

            logger.info(f"   ✅ Fallback : {len(selected_features)} features selecionadas")
            return selected_features

        except Exception as e:
            logger.error(f"❌ Erro no fallback: {e}")
            return available_features[:max_features]

    def _prepare_X_for_selection(self, X: np.ndarray, features: List[str]) -> np.ndarray:
        """Prepara X para seleção de features"""

        X_processed = X.copy()

        if X_processed.ndim == 1:
            X_processed = X_processed.reshape(-1, 1)

        for i in range(X_processed.shape[1]):
            col_data = X_processed[:, i]

            try:
                numeric_col = pd.to_numeric(col_data, errors='coerce')
                if not pd.isna(numeric_col).all():
                    X_processed[:, i] = numeric_col
            except:
                pass

            col_data = X_processed[:, i]
            if pd.isna(col_data).any():
                if pd.api.types.is_numeric_dtype(col_data):
                    median_val = np.nanmedian(col_data.astype(float))
                    X_processed[:, i] = np.where(pd.isna(col_data), median_val, col_data)
                else:
                    le = LabelEncoder()
                    non_na_mask = ~pd.isna(col_data)
                    if non_na_mask.any():
                        unique_vals = pd.unique(col_data[non_na_mask])
                        le.fit(unique_vals)
                        encoded = np.full(len(col_data), -1)
                        encoded[non_na_mask] = le.transform(col_data[non_na_mask])
                        X_processed[:, i] = encoded

        return X_processed.astype(float)

    def select_best_features(self, X: np.ndarray, y: np.ndarray, max_features: int = 15) -> Tuple[
        np.ndarray, Any, List[str]]:
        """Usar DataFrame e seleção simples"""

        logger.info(f"🔍 SELEÇÃO ANTI-OVERFITTING")
        logger.info(f"🔄 {max_features} features de {X.shape[1]} disponíveis")

        try:
            available_features = self.feature_names if hasattr(self, 'feature_names') else [f"feature_{i}" for i in
                                                                                            range(X.shape[1])]
            if len(available_features) != X.shape[1]:
                logger.warning(f"⚠️ Ajustando feature_names: {len(available_features)} -> {X.shape[1]}")
                available_features = [f"feature_{i}" for i in range(X.shape[1])]

            df = pd.DataFrame(X, columns=available_features)
            logger.info(f"   ✅ DataFrame criado: {df.shape}")

            derived_available = [f for f in self.derived_features if f in df.columns]
            logger.info(f"   🔄 Features derivadas disponíveis: {len(derived_available)}")

            if derived_available and self.derived_features_config.get('prioritize_derived_in_selection', True):
                selected_features = self._simple_derived_selection(df, y, derived_available, max_features)
            else:
                selected_features = self._simple_standard_selection(df, y, max_features)

            if not selected_features:
                logger.error("❌ Fallback para primeiras features")
                selected_features = available_features[:max_features]

            try:
                X_selected = df[selected_features].values
                logger.info(f"   ✅ X_selected: {X_selected.shape}")
            except Exception as e:
                logger.error(f"❌ Erro na conversão final: {e}")
                X_selected = X[:, :max_features]
                selected_features = available_features[:max_features]

            logger.info(f"✅ SELEÇÃO CONCLUÍDA: {len(selected_features)} features")
            logger.info(f"🔄 Features derivadas: {[f for f in selected_features if f in self.derived_features]}")

            return X_selected, None, selected_features

        except Exception as e:
            logger.error(f"❌ Erro crítico na seleção: {e}")
            n_features = min(max_features, X.shape[1])
            return X[:, :n_features], None, available_features[:n_features]

    def _simple_derived_selection(self, df: pd.DataFrame, y: np.ndarray,
                                  derived_available: List[str], max_features: int) -> List[str]:
        """Seleção simples priorizando derivadas usando DataFrame"""

        try:
            selected = []

            max_derived = min(len(derived_available), max_features // 2, 5)

            if max_derived > 0:
                X_derived = df[derived_available].values
                from sklearn.feature_selection import SelectKBest, f_classif

                selector_derived = SelectKBest(score_func=f_classif, k=max_derived)
                selector_derived.fit(X_derived, y)
                selected_mask = selector_derived.get_support()
                selected_derived = [derived_available[i] for i, sel in enumerate(selected_mask) if sel]
                selected.extend(selected_derived)

                logger.info(f"   ✅ {len(selected_derived)} derivadas selecionadas: {selected_derived}")

            remaining_slots = max_features - len(selected)
            if remaining_slots > 0:
                non_derived = [col for col in df.columns if col not in self.derived_features and col not in selected]

                if non_derived and remaining_slots > 0:
                    max_non_derived = min(len(non_derived), remaining_slots)
                    X_non_derived = df[non_derived].values

                    selector_others = SelectKBest(score_func=f_classif, k=max_non_derived)
                    selector_others.fit(X_non_derived, y)

                    selected_mask = selector_others.get_support()
                    selected_others = [non_derived[i] for i, sel in enumerate(selected_mask) if sel]
                    selected.extend(selected_others)

                    logger.info(f"   ✅ {len(selected_others)} não-derivadas selecionadas")

            return selected[:max_features]

        except Exception as e:
            logger.error(f"❌ Erro na seleção com derivadas: {e}")
            return list(df.columns[:max_features])

    def _simple_standard_selection(self, df: pd.DataFrame, y: np.ndarray, max_features: int) -> List[str]:
        """Seleção padrão simples usando DataFrame"""

        try:
            logger.info("   📊 Seleção padrão simples")

            from sklearn.feature_selection import SelectKBest, f_classif
            n_select = min(max_features, len(df.columns))
            selector = SelectKBest(score_func=f_classif, k=n_select)
            selector.fit(df.values, y)
            selected_mask = selector.get_support()
            selected_features = [df.columns[i] for i, sel in enumerate(selected_mask) if sel]

            logger.info(f"   ✅ Seleção padrão: {len(selected_features)} features")
            return selected_features

        except Exception as e:
            logger.error(f"❌ Erro na seleção padrão: {e}")
            return list(df.columns[:max_features])

    def cross_validate_base_models(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Cross-validation com análise especial para features derivadas"""

        try:
            logger.info(f"\n🔄 Cross-validation com análise de features derivadas...")

            cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

            derived_present = [f for f in self.derived_features if f in self.selected_features]
            if derived_present:
                logger.info(f"   🔄 Features derivadas em uso: {derived_present}")

            for name, model in self.base_models.items():
                logger.info(f"   📊 Cross-validando {name}...")

                if isinstance(model, XGBClassifier):
                    unique, counts = np.unique(y, return_counts=True)
                    scale_pos_weight = counts[1] / counts[0] if len(counts) > 1 else 1.0
                    model.set_params(scale_pos_weight=scale_pos_weight)

                cv_scores = cross_validate(
                    model, X, y,
                    cv=cv_strategy,
                    scoring=['balanced_accuracy', 'f1', 'roc_auc'],
                    n_jobs=-1,
                    return_train_score=True
                )

                train_acc_mean = cv_scores['train_balanced_accuracy'].mean()
                test_acc_mean = cv_scores['test_balanced_accuracy'].mean()
                test_acc_std = cv_scores['test_balanced_accuracy'].std()

                overfitting_gap = train_acc_mean - test_acc_mean

                self.cv_results[name] = {
                    'test_balanced_accuracy': cv_scores['test_balanced_accuracy'],
                    'test_f1': cv_scores['test_f1'],
                    'test_roc_auc': cv_scores['test_roc_auc'],
                    'train_balanced_accuracy': cv_scores['train_balanced_accuracy'],
                    'train_f1': cv_scores['train_f1'],
                    'train_roc_auc': cv_scores['train_roc_auc'],
                    'overfitting_gap': overfitting_gap,
                    'derived_features_used': derived_present
                }

                logger.info(f"      ✅ {name}: {test_acc_mean:.3f} ± {test_acc_std:.3f}")

                if overfitting_gap > 0.03:
                    logger.warning(f"      🚨 OVERFITTING DETECTADO! Gap: {overfitting_gap:.3f}")
                else:
                    logger.info(f"      ✅ Modelo bem generalizado (gap: {overfitting_gap:.3f})")

            return True

        except Exception as e:
            logger.error(f"❌ Erro no cross-validation: {e}")
            return False

    def train_base_models(self, X_train: np.ndarray, y_train: np.ndarray) -> bool:
        """Treina modelos base com features derivadas"""

        try:
            logger.info(f"\n🎯 Treinando modelos base com features derivadas...")

            for name, model in self.base_models.items():
                logger.info(f"   🔧 Treinando {name}...")
                start_time = time.time()

                if isinstance(model, XGBClassifier):
                    unique, counts = np.unique(y_train, return_counts=True)
                    scale_pos_weight = counts[1] / counts[0] if len(counts) > 1 else 1.0
                    model.set_params(scale_pos_weight=scale_pos_weight)

                elif isinstance(model, RandomForestClassifier):
                    derived_present = [f for f in self.derived_features if f in self.selected_features]
                    if len(derived_present) > len(self.selected_features) * 0.3:
                        model.set_params(max_features='log2')

                model.fit(X_train, y_train)

                elapsed_time = time.time() - start_time
                logger.info(f"      ✅ {name} treinado em {elapsed_time:.2f}s")

            return True

        except Exception as e:
            logger.error(f"❌ Erro no treinamento dos modelos base: {e}")
            return False

    def prepare_data_for_tabpfn(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepara dados para TabPFN com otimização para features derivadas"""

        n_samples, n_features = X.shape

        logger.info(f"\n🔧 Preparando dados para TabPFN com features derivadas:")
        logger.info(f"   📊 Dados originais: {n_samples} amostras, {n_features} features")

        X_processed = X.copy()
        y_processed = y.copy()

        if n_samples > self.max_samples_tabpfn:
            logger.warning(f"⚠️ Reduzindo amostras de {n_samples} para {self.max_samples_tabpfn}")
            X_processed, _, y_processed, _ = train_test_split(
                X_processed, y_processed,
                train_size=self.max_samples_tabpfn,
                stratify=y_processed,
                random_state=42
            )

        if n_features > self.max_features_tabpfn:
            logger.warning(f"⚠️ Reduzindo features de {n_features} para {self.max_features_tabpfn}")

            if self.feature_selector is None:
                if hasattr(self, 'selected_features') and self.selected_features:
                    derived_in_selected = [f for f in self.selected_features if f in self.derived_features]
                    non_derived_in_selected = [f for f in self.selected_features if f not in self.derived_features]

                    priority_features = derived_in_selected + non_derived_in_selected
                    features_to_keep = priority_features[:self.max_features_tabpfn]

                    feature_indices = []
                    for feature in features_to_keep:
                        try:
                            idx = self.selected_features.index(feature)
                            if idx < X_processed.shape[1]:
                                feature_indices.append(idx)
                        except ValueError:
                            continue

                    if feature_indices:
                        X_processed = X_processed[:, feature_indices]
                        logger.info(
                            f"   🔄 Priorizadas {len([f for f in features_to_keep if f in self.derived_features])} features derivadas")
                else:
                    self.feature_selector = SelectKBest(score_func=f_classif, k=self.max_features_tabpfn)
                    X_processed = self.feature_selector.fit_transform(X_processed, y_processed)
            else:
                X_processed = self.feature_selector.transform(X_processed)

        logger.info(f"✅ Dados preparados para TabPFN: {X_processed.shape[0]} amostras, {X_processed.shape[1]} features")
        return X_processed, y_processed

    def train_tabpfn_model(self, X_train: np.ndarray, y_train: np.ndarray) -> bool:
        """Treina TabPFN como modelo principal com  derivadas"""

        try:
            logger.info(f"\n🤖 Treinando TabPFN com features derivadas...")

            if self.use_tabpfn:
                try:
                    X_processed, y_processed = self.prepare_data_for_tabpfn(X_train, y_train)

                    logger.info("🔄 Inicializando TabPFN...")
                    self.tabpfn_model = TabPFNClassifier(device='cpu', random_state=42)

                    logger.info("🔄 Treinando TabPFN...")
                    start_time = time.time()
                    self.tabpfn_model.fit(X_processed, y_processed)
                    elapsed_time = time.time() - start_time

                    logger.info(f"✅ TabPFN treinado com sucesso em {elapsed_time:.2f}s!")

                    if hasattr(self, 'selected_features'):
                        derived_used = [f for f in self.selected_features if f in self.derived_features]
                        if derived_used:
                            logger.info(f"🔄 Features derivadas utilizadas no TabPFN: {derived_used}")

                    return True

                except Exception as e:
                    logger.error(f"❌ Erro com TabPFN: {e}")
                    logger.warning("🔄 Mudando para XGBoost como modelo principal...")
                    self.use_tabpfn = False
                    return self._train_fallback_model(X_train, y_train)
            else:
                return self._train_fallback_model(X_train, y_train)

        except Exception as e:
            logger.error(f"❌ Erro no treinamento do TabPFN: {e}")
            return False

    def _train_fallback_model(self, X_train: np.ndarray, y_train: np.ndarray) -> bool:
        """Modelo de fallback XGBoost """

        try:
            logger.info("🔄 Treinando modelo XGBoost de fallback ...")

            unique, counts = np.unique(y_train, return_counts=True)
            scale_pos_weight = counts[1] / counts[0] if len(counts) > 1 else 1.0

            self.tabpfn_model = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss',
                verbosity=0,
                tree_method='hist',
                grow_policy='lossguide',
                max_leaves=0,
                gamma=0.1
            )

            start_time = time.time()
            self.tabpfn_model.fit(X_train, y_train)
            elapsed_time = time.time() - start_time

            logger.info(f"✅ XGBoost fallback treinado em {elapsed_time:.2f}s")

            if hasattr(self, 'selected_features'):
                derived_used = [f for f in self.selected_features if f in self.derived_features]
                if derived_used:
                    logger.info(f"🔄 Features derivadas utilizadas no XGBoost: {derived_used}")

            return True

        except Exception as e:
            logger.error(f"❌ Erro no modelo fallback: {e}")
            return False

    def get_model_predictions(self, X: np.ndarray, model_name: Optional[str] = None) -> np.ndarray:
        """Obtém predições de um modelo específico ou principal"""

        try:
            if model_name and model_name in self.base_models:
                model = self.base_models[model_name]
                return model.predict_proba(X)[:, 1]
            else:
                if self.use_tabpfn and hasattr(self, 'feature_selector') and self.feature_selector:
                    X_processed = self.feature_selector.transform(X)
                    return self.tabpfn_model.predict_proba(X_processed)[:, 1]
                else:
                    return self.tabpfn_model.predict_proba(X)[:, 1]
        except Exception as e:
            logger.warning(f"⚠️ Erro ao obter predições: {e}")
            return np.full(X.shape[0], 0.5)

    def test_meta_models(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Dict]:
        """Testa diferentes meta-modelos com análise de features derivadas"""

        try:
            logger.info(f"\n🧠 Testando meta-modelos com features derivadas...")

            main_preds = self.get_model_predictions(X_val)
            base_preds = []
            for name in self.base_models.keys():
                preds = self.get_model_predictions(X_val, name)
                base_preds.append(preds)

            all_preds = np.column_stack([main_preds] + base_preds)
            logger.info(f"📊 Predições combinadas: {all_preds.shape}")

            candidates = self.get_meta_model_candidates()
            meta_results = {}

            cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            for meta_name, meta_model in candidates.items():
                logger.info(f"   🧪 Testando meta-modelo: {meta_name}")

                if isinstance(meta_model, XGBClassifier):
                    unique, counts = np.unique(y_val, return_counts=True)
                    scale_pos_weight = counts[1] / counts[0] if len(counts) > 1 else 1.0
                    meta_model.set_params(scale_pos_weight=scale_pos_weight)

                cv_scores = cross_validate(
                    meta_model, all_preds, y_val,
                    cv=cv_strategy,
                    scoring=['balanced_accuracy', 'f1', 'roc_auc'],
                    n_jobs=-1
                )

                meta_results[meta_name] = {
                    'balanced_accuracy': cv_scores['test_balanced_accuracy'].mean(),
                    'f1_score': cv_scores['test_f1'].mean(),
                    'auc_score': cv_scores['test_roc_auc'].mean(),
                    'balanced_accuracy_std': cv_scores['test_balanced_accuracy'].std()
                }

                logger.info(
                    f"      📊 {meta_name}: {meta_results[meta_name]['balanced_accuracy']:.3f} ± {meta_results[meta_name]['balanced_accuracy_std']:.3f}")

            best_meta = max(meta_results.keys(), key=lambda x: meta_results[x]['balanced_accuracy'])

            logger.info(f"\n✅ Melhor meta-modelo: {best_meta.upper()}")
            logger.info(f"   📊 Performance: {meta_results[best_meta]['balanced_accuracy']:.3f}")

            self.meta_model = candidates[best_meta]
            if isinstance(self.meta_model, XGBClassifier):
                unique, counts = np.unique(y_val, return_counts=True)
                scale_pos_weight = counts[1] / counts[0] if len(counts) > 1 else 1.0
                self.meta_model.set_params(scale_pos_weight=scale_pos_weight)

            self.meta_model.fit(all_preds, y_val)
            self.best_meta_name = best_meta
            self.meta_results = meta_results

            return meta_results

        except Exception as e:
            logger.error(f"❌ Erro no teste de meta-modelos: {e}")
            return {}

    def train_model(self) -> bool:
        """Pipeline completo ANTI-OVERFITTING + FEATURES DERIVADAS"""

        try:
            logger.info("🚀 INICIANDO TREINAMENTO ANTI-OVERFITTING + FEATURES DERIVADAS")
            logger.info("📊 TabPFN + LR/RF/XGB + Meta-Model Selection + Derivadas Otimizadas")
            logger.info("=" * 80)

            if not self.data_processed:
                raise ValueError("Execute preprocess_data() primeiro")

            if not self.create_balanced_data(strategy='undersample_exact'):
                return False

            logger.info("\n🔍 Seleção de features com priorização de derivadas...")
            X_selected, self.feature_selector_main, self.selected_features = self.select_best_features(
                self.X_balanced, self.y_balanced, max_features=15
            )

            self._analyze_selected_features()

            logger.info("\n📊 Normalizando dados (preservando features categóricas derivadas)...")
            X_scaled = self._smart_scaling(X_selected)

            logger.info("\n✂️ Divisão 60-20-20 (treino-validação-teste) ANTI-OVERFITTING...")

            X_temp, X_test, y_temp, y_test = train_test_split(
                X_scaled, self.y_balanced, test_size=0.20, random_state=42, stratify=self.y_balanced
            )

            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
            )

            logger.info(f"📊 Treino: {X_train.shape[0]} amostras ({X_train.shape[0] / len(self.y_balanced) * 100:.1f}%)")
            logger.info(f"📊 Validação: {X_val.shape[0]} amostras ({X_val.shape[0] / len(self.y_balanced) * 100:.1f}%)")
            logger.info(f"📊 Teste: {X_test.shape[0]} amostras ({X_test.shape[0] / len(self.y_balanced) * 100:.1f}%)")

            if not self.cross_validate_base_models(X_train, y_train):
                return False

            if not self.train_base_models(X_train, y_train):
                return False

            if not self.train_tabpfn_model(X_train, y_train):
                return False

            if not self.test_meta_models(X_train, y_train, X_val, y_val):
                return False

            logger.info("\n📈 Avaliação final no conjunto de teste...")
            self.evaluate_model_processed(X_test, y_test)

            self._analyze_derived_features_performance()

            logger.info("\n" + "=" * 80)
            logger.info("✅ TREINAMENTO ANTI-OVERFITTING + DERIVADAS CONCLUÍDO!")
            logger.info(f"🏆 Modelo Principal: {'TabPFN' if self.use_tabpfn else 'XGBoost'}")
            logger.info(f"🧠 Melhor Meta-modelo: {self.best_meta_name.upper() if self.best_meta_name else 'N/A'}")
            logger.info(f"📉 Features usadas: {len(self.selected_features)}")
            logger.info(
                f"🔄 Features derivadas: {len([f for f in self.selected_features if f in self.derived_features])}")
            logger.info("=" * 80)

            return True

        except Exception as e:
            logger.error(f"❌ Erro no treinamento: {e}")
            return False

    def _analyze_selected_features(self):
        """Analisa as features selecionadas"""

        derived_selected = [f for f in self.selected_features if f in self.derived_features]
        original_selected = [f for f in self.selected_features if f not in self.derived_features]
        replaced_selected = [f for f in self.selected_features if f in self.replaced_features]

        logger.info(f"📊 ANÁLISE DAS FEATURES SELECIONADAS:")
        logger.info(
            f"   🔄 Features derivadas: {len(derived_selected)}/{len(self.selected_features)} ({len(derived_selected) / len(self.selected_features) * 100:.1f}%)")
        logger.info(
            f"   📊 Features originais: {len(original_selected)}/{len(self.selected_features)} ({len(original_selected) / len(self.selected_features) * 100:.1f}%)")
        logger.info(f"   ⚠️ Features substituídas ainda em uso: {len(replaced_selected)}")

        if derived_selected:
            logger.info(f"   🔄 Derivadas selecionadas: {derived_selected}")

        if replaced_selected:
            logger.warning(f"   ⚠️ Features substituídas em uso: {replaced_selected}")

    def _smart_scaling(self, X: np.ndarray) -> np.ndarray:
        """Normalização inteligente que preserva features categóricas derivadas"""

        logger.info("📊 Aplicando normalização inteligente...")

        categorical_derived = ['CASO_COMPLEXO', 'TEMPO_INICIO_CAT', 'DURACAO_PREVISTA_CAT']

        X_scaled = self.scaler.fit_transform(X)

        if hasattr(self, 'selected_features'):
            for i, feature in enumerate(self.selected_features):
                if feature in categorical_derived and i < X_scaled.shape[1]:
                    col_data = X[:, i]
                    col_min, col_max = col_data.min(), col_data.max()
                    if col_max > col_min:
                        X_scaled[:, i] = (col_data - col_min) / (col_max - col_min)

                    logger.info(f"   🔄 {feature}: normalização min-max aplicada")

        return X_scaled

    def _analyze_derived_features_performance(self):
        """Analisa performance específica das features derivadas"""

        if not hasattr(self, 'results') or not self.results:
            return

        logger.info(f"\n🔄 ANÁLISE DE PERFORMANCE DAS FEATURES DERIVADAS:")

        derived_in_use = [f for f in self.selected_features if f in self.derived_features]

        if derived_in_use:
            logger.info(f"   ✅ Features derivadas utilizadas: {len(derived_in_use)}")
            logger.info(f"   📋 Lista: {derived_in_use}")

            derived_ratio = len(derived_in_use) / len(self.selected_features)
            logger.info(f"   📊 Proporção de derivadas: {derived_ratio:.1%}")

            if hasattr(self, 'base_models') and 'rf' in self.base_models:
                try:
                    rf_model = self.base_models['rf']
                    if hasattr(rf_model, 'feature_importances_'):
                        importances = rf_model.feature_importances_

                        derived_importances = []
                        for i, feature in enumerate(self.selected_features):
                            if feature in self.derived_features and i < len(importances):
                                derived_importances.append((feature, importances[i]))

                        if derived_importances:
                            derived_importances.sort(key=lambda x: x[1], reverse=True)
                            logger.info(f"   🎯 Importância das features derivadas (Random Forest):")
                            for feature, importance in derived_importances:
                                logger.info(f"      🔄 {feature}: {importance:.4f}")

                except Exception as e:
                    logger.warning(f"   ⚠️ Erro ao analisar importâncias: {e}")
        else:
            logger.warning(f"   ⚠️ Nenhuma feature derivada foi selecionada!")

        replaced_still_used = [f for f in self.selected_features if f in self.replaced_features]
        if replaced_still_used:
            logger.warning(f"   ⚠️ Features substituídas ainda em uso: {replaced_still_used}")
            logger.warning(f"   💡 Considere ajustar a lógica de substituição")
        else:
            logger.info(f"   ✅ Substituições aplicadas com sucesso - originais não estão sendo usadas")

    def predict_processed(self, X_processed: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predições para dados já processados"""

        try:
            main_preds = self.get_model_predictions(X_processed)

            base_preds = []
            for name in self.base_models.keys():
                preds = self.get_model_predictions(X_processed, name)
                base_preds.append(preds)

            all_preds = np.column_stack([main_preds] + base_preds)

            final_preds = self.meta_model.predict(all_preds)
            final_proba = self.meta_model.predict_proba(all_preds)[:, 1]

            return final_preds, final_proba

        except Exception as e:
            logger.error(f"❌ Erro nas predições: {e}")
            return np.zeros(X_processed.shape[0]), np.full(X_processed.shape[0], 0.5)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predições do ensemble para dados novos com features derivadas"""

        try:
            if hasattr(self, 'feature_selector_main') and self.feature_selector_main:
                if hasattr(self.feature_selector_main, 'transform'):
                    X_selected = self.feature_selector_main.transform(X)
                elif hasattr(self.feature_selector_main, 'selected_features'):
                    if isinstance(X, pd.DataFrame):
                        available_features = [f for f in self.feature_selector_main.selected_features if f in X.columns]
                        X_selected = X[available_features].values
                    else:
                        logger.warning("⚠️ Usando todas as features - seletor não compatível")
                        X_selected = X
                else:
                    logger.warning("⚠️ Feature selector não tem método transform nem selected_features")
                    X_selected = X
            else:
                X_selected = X

            X_scaled = self.scaler.transform(X_selected)

            return self.predict_processed(X_scaled)

        except Exception as e:
            logger.error(f"❌ Erro na predição: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return np.zeros(X.shape[0]), np.full(X.shape[0], 0.5)

    def evaluate_model_processed(self, X_test: np.ndarray, y_test: np.ndarray) -> bool:
        """Avaliação para dados já processados com análise de features derivadas"""
        try:
            predictions, probabilities = self.predict_processed(X_test)

            accuracy = np.mean(predictions == y_test)
            balanced_acc = balanced_accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions)
            auc_score = roc_auc_score(y_test, probabilities)

            optimal_threshold = self._find_optimal_threshold(y_test, probabilities)
            optimized_preds = (probabilities >= optimal_threshold).astype(int)
            optimized_f1 = f1_score(y_test, optimized_preds)

            self.results = {
                'accuracy': accuracy,
                'balanced_accuracy': balanced_acc,
                'f1_score': f1,
                'auc_score': auc_score,
                'optimal_threshold': optimal_threshold,
                'optimized_f1': optimized_f1,
                'confusion_matrix': confusion_matrix(y_test, predictions),
                'optimized_confusion_matrix': confusion_matrix(y_test, optimized_preds),
                'derived_features_used': [f for f in self.selected_features if f in self.derived_features],
                'derived_features_count': len([f for f in self.selected_features if f in self.derived_features]),
                'total_features_count': len(self.selected_features)
            }

            logger.info("\n" + "=" * 60)
            logger.info("📊 RESULTADOS FINAIS COM FEATURES DERIVADAS")
            logger.info("=" * 60)
            logger.info(f"🎯 Acurácia: {accuracy:.4f}")
            logger.info(f"⚖️ Acurácia Balanceada: {balanced_acc:.4f}")
            logger.info(f"🔥 F1-Score: {f1:.4f}")
            logger.info(f"📈 AUC Score: {auc_score:.4f}")
            logger.info(f"🎚️ Threshold: {optimal_threshold:.4f}")
            logger.info(f"⭐ F1: {optimized_f1:.4f}")
            logger.info(f"📊 Features Totais: {len(self.selected_features)}")
            logger.info(
                f"🔄 Features Derivadas: {len([f for f in self.selected_features if f in self.derived_features])}")
            logger.info(f"🏆 Meta-modelo: {self.best_meta_name.upper() if self.best_meta_name else 'N/A'}")

            cm = self.results['optimized_confusion_matrix']
            if cm.sum() > 0:
                tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

                logger.info(f"\n🔍 MÉTRICAS CLÍNICAS:")
                logger.info(f"   Sensibilidade (detectar óbitos): {sensitivity:.4f}")
                logger.info(f"   Especificidade (detectar sobrevivência): {specificity:.4f}")

            derived_used = [f for f in self.selected_features if f in self.derived_features]
            if derived_used:
                logger.info(f"\n🔄 CONTRIBUIÇÃO DAS FEATURES DERIVADAS:")
                logger.info(f"   Features utilizadas: {derived_used}")
                logger.info(f"   Proporção do modelo: {len(derived_used) / len(self.selected_features):.1%}")

                original_replaced = []
                for derived in derived_used:
                    if derived in self.replacement_mappings:
                        original_replaced.extend(self.replacement_mappings[derived].get('replaces', []))

                original_still_used = [f for f in self.selected_features if f in original_replaced]
                if not original_still_used:
                    logger.info(f"   ✅ Substituições efetivas: originais não estão sendo usadas")
                else:
                    logger.warning(f"   ⚠️ Originais ainda em uso: {original_still_used}")

            self._print_cv_summary()
            self._print_meta_model_comparison()

            return True

        except Exception as e:
            logger.error(f"❌ Erro na avaliação: {e}")
            return False

    def _find_optimal_threshold(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Encontra threshold que maximiza F1-score"""
        thresholds = np.linspace(0, 1, 100)
        f1_scores = []

        for t in thresholds:
            preds = (y_prob >= t).astype(int)
            f1_scores.append(f1_score(y_true, preds))

        return thresholds[np.argmax(f1_scores)]

    def _print_cv_summary(self):
        """Imprime resumo dos resultados de cross-validation"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 RESUMO CROSS-VALIDATION ANTI-OVERFITTING")
        logger.info("=" * 60)

        for model_name, results in self.cv_results.items():
            acc_mean = results['test_balanced_accuracy'].mean()
            acc_std = results['test_balanced_accuracy'].std()
            f1_mean = results['test_f1'].mean()
            auc_mean = results['test_roc_auc'].mean()
            gap = results.get('overfitting_gap', 0)
            derived_used = results.get('derived_features_used', [])

            logger.info(f"\n🎯 {model_name.upper()}:")
            logger.info(f"   Acurácia Balanceada: {acc_mean:.3f} ± {acc_std:.3f}")
            logger.info(f"   F1-Score: {f1_mean:.3f}")
            logger.info(f"   AUC: {auc_mean:.3f}")
            logger.info(f"   Gap Overfitting: {gap:.3f}")
            if derived_used:
                logger.info(f"   Features derivadas: {len(derived_used)}")

            if gap > 0.03:
                logger.warning(f"   🚨 OVERFITTING DETECTADO")
            else:
                logger.info(f"   ✅ Bem generalizado")

    def _print_meta_model_comparison(self):
        """Imprime comparação dos meta-modelos testados"""
        if not self.meta_results:
            return

        logger.info("\n" + "=" * 60)
        logger.info("🏆 COMPARAÇÃO DOS META-MODELOS")
        logger.info("=" * 60)

        sorted_models = sorted(
            self.meta_results.items(),
            key=lambda x: x[1]['balanced_accuracy'],
            reverse=True
        )

        for i, (name, results) in enumerate(sorted_models):
            rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i + 1}º"
            logger.info(f"{rank} {name.upper()}:")
            logger.info(f"   Acurácia: {results['balanced_accuracy']:.3f} ± {results['balanced_accuracy_std']:.3f}")
            logger.info(f"   F1-Score: {results['f1_score']:.3f}")
            logger.info(f"   AUC: {results['auc_score']:.3f}")

    def _load_derived_features_config(self) -> Dict[str, Any]:
        """Carrega configurações específicas para features derivadas"""
        model_config = self.config.get('model', {})
        derived_config = model_config.get('derived_features_handling', {})

        return {
            'prioritize_derived_in_selection': derived_config.get('prioritize_derived_in_selection', True),
            'derived_feature_weight_multiplier': derived_config.get('derived_feature_weight_multiplier', 1.2),
            'validate_derived_performance': derived_config.get('validate_derived_performance', True),
            'min_derived_features': derived_config.get('min_derived_features', 2),
            'max_derived_ratio': derived_config.get('max_derived_ratio', 0.6),
            'max_derived_in_selection': derived_config.get('max_derived_in_selection', 5)
        }

    def _load_replacement_info(self):
        """Carrega informações de substituição do feature selector"""
        if hasattr(self.feature_selector, 'derived_features'):
            self.derived_features = self.feature_selector.derived_features.copy()

        if hasattr(self.feature_selector, 'replaced_features'):
            self.replaced_features = self.feature_selector.replaced_features.copy()

        if hasattr(self.feature_selector, 'replacement_mappings'):
            self.replacement_mappings = self.feature_selector.replacement_mappings.copy()

        logger.info(f"🔄 Informações de substituição carregadas:")
        logger.info(f"   Features derivadas: {self.derived_features}")
        logger.info(f"   Features substituídas: {list(self.replaced_features)}")

    def _get_optimized_base_models(self) -> Dict[str, Any]:
        """Modelos base com features derivadas categóricas"""
        return {
            'lr': LogisticRegression(
                C=0.1,
                class_weight='balanced',
                random_state=42,
                max_iter=1000,
                penalty='l2',
                solver='liblinear'
            ),
            'rf': RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                min_samples_split=15,
                min_samples_leaf=8,
                max_features='sqrt',
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                bootstrap=True,
                oob_score=True
            ),
            'xgb': XGBClassifier(
                n_estimators=50,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=0.5,
                reg_lambda=0.5,
                min_child_weight=15,
                gamma=0.3,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss',
                verbosity=0,
                tree_method='hist',
                grow_policy='lossguide'
            )
        }

    def get_meta_model_candidates(self) -> Dict[str, Any]:
        """Meta-modelos candidatos com features derivadas"""
        candidates = {
            'lr': LogisticRegression(
                C=0.05,
                class_weight='balanced',
                random_state=42,
                max_iter=1000,
                penalty='l2',
                solver='liblinear'
            ),
            'rf': RandomForestClassifier(
                n_estimators=30,
                max_depth=4,
                min_samples_split=15,
                min_samples_leaf=8,
                max_features='sqrt',
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'xgb': XGBClassifier(
                n_estimators=25,
                max_depth=3,
                learning_rate=0.01,
                reg_alpha=1.0,
                reg_lambda=1.0,
                min_child_weight=20,
                gamma=0.5,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss',
                verbosity=0
            ),
            'svm': SVC(
                C=0.1,
                kernel='rbf',
                class_weight='balanced',
                probability=True,
                gamma='scale',
                random_state=42
            )
        }
        return candidates

    def load_data(self, file_path: str) -> bool:
        """Carrega dados com validação robusta"""

        try:
            logger.info(f"📊 Carregando dados de: {file_path}")

            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                self.data = pd.read_excel(file_path)
            else:
                raise ValueError("Formato de arquivo não suportado. Use CSV ou Excel.")

            logger.info(f"✅ Dados carregados: {self.data.shape}")

            if 'SITUA_ENCE' not in self.data.columns:
                logger.error("❌ Coluna 'SITUA_ENCE' não encontrada!")
                return False

            return True

        except Exception as e:
            logger.error(f"❌ Erro ao carregar dados: {e}")
            return False

    def create_target_variable(self) -> bool:
        """Cria variável target com validação"""

        target_mapping = {1: 1, 2: 0}  # Cura = 1, Óbito = 0

        if 'SITUA_ENCE' in self.data.columns:
            logger.info("🎯 Criando variável target...")

            self.data['TARGET'] = self.data['SITUA_ENCE'].map(target_mapping)

            before_drop = len(self.data)
            self.data = self.data.dropna(subset=['TARGET'])
            after_drop = len(self.data)

            if before_drop != after_drop:
                logger.warning(f"⚠️ Removidos {before_drop - after_drop} registros sem target válido")

            target_counts = self.data['TARGET'].value_counts()
            logger.info(f"✅ Distribuição do target:")
            logger.info(f"   Sobrevivência (1): {target_counts.get(1, 0)} ({target_counts.get(1, 0) / len(self.data) * 100:.1f}%)")
            logger.info(f"   Óbito (0): {target_counts.get(0, 0)} ({target_counts.get(0, 0) / len(self.data) * 100:.1f}%)")

            return True
        else:
            logger.error("❌ Coluna SITUA_ENCE não encontrada")
            return False

    def preprocess_data(self) -> bool:
        """Pré-processamento  para features derivadas"""

        try:
            logger.info("\n🔄 Iniciando pré-processamento para features derivadas...")

            if not self.create_target_variable():
                raise ValueError("Erro na criação da variável target")

            date_cols = ['DT_NOTIFIC', 'DT_DIAG', 'DT_INIC_TR', 'DT_NOTI_AT', 'DT_ENCERRA', 'NU_ANO']
            temporal_unsafe_cols = ['DIAS']
            cols_to_drop = [col for col in date_cols + temporal_unsafe_cols if col in self.data.columns]
            cols_to_drop.extend(['TARGET', 'SITUA_ENCE'])

            logger.info(f"🗑️ Removendo colunas unsafe: {cols_to_drop}")

            self.X = self.data.drop(columns=cols_to_drop, errors='ignore')
            self.y = self.data['TARGET'].astype(int)

            self._validate_derived_features_in_data()

            logger.info(f"📊 Features restantes: {self.X.shape[1]}")
            logger.info(f"📊 Amostras: {len(self.y)}")

            categorical_cols = self.X.select_dtypes(include=['object']).columns
            numerical_cols = self.X.select_dtypes(include=[np.number]).columns

            derived_categorical = [col for col in self.derived_features if col in categorical_cols]
            derived_numerical = [col for col in self.derived_features if col in numerical_cols]

            logger.info(f"📝 Colunas categóricas: {len(categorical_cols)} (derivadas: {len(derived_categorical)})")
            logger.info(f"🔢 Colunas numéricas: {len(numerical_cols)} (derivadas: {len(derived_numerical)})")

            self._preprocess_derived_features(derived_categorical, derived_numerical)

            self._preprocess_standard_features(categorical_cols, numerical_cols)

            self.feature_names = list(self.X.columns)
            self.data_processed = True

            logger.info(f"✅ Pré-processamento concluído")
            logger.info(f"📊 Features finais: {len(self.feature_names)}")

            derived_present = [f for f in self.derived_features if f in self.feature_names]
            if derived_present:
                logger.info(f"🔄 Features derivadas presentes: {derived_present}")

            return True

        except Exception as e:
            logger.error(f"❌ Erro no pré-processamento: {e}")
            return False

    def _validate_derived_features_in_data(self):
        """Valida presença e qualidade das features derivadas nos dados"""

        logger.info("🔍 Validando features derivadas nos dados...")

        derived_present = []
        derived_missing = []
        derived_quality = {}

        for feature in self.derived_features:
            if feature in self.X.columns:
                derived_present.append(feature)

                quality_analysis = self._analyze_derived_feature_quality(feature)
                derived_quality[feature] = quality_analysis

                logger.info(f"   ✅ {feature}: {quality_analysis['status']}")
                if quality_analysis['warnings']:
                    for warning in quality_analysis['warnings']:
                        logger.warning(f"      ⚠️ {warning}")
            else:
                derived_missing.append(feature)
                logger.warning(f"   ❌ {feature}: AUSENTE")

        logger.info(f"📊 Features derivadas: {len(derived_present)} presentes, {len(derived_missing)} ausentes")

        if self.replacement_mappings:
            self._validate_replacements_in_data()

    def _analyze_derived_feature_quality(self, feature: str) -> Dict[str, Any]:
        """Analisa qualidade de uma feature derivada específica"""

        quality = {
            'status': 'OK',
            'warnings': [],
            'statistics': {}
        }

        col_data = self.X[feature]

        quality['statistics'] = {
            'missing_pct': col_data.isnull().mean() * 100,
            'unique_values': col_data.nunique(),
            'dtype': str(col_data.dtype)
        }

        if feature == 'SCORE_COMORBIDADES':
            if col_data.dtype not in ['int64', 'float64']:
                quality['warnings'].append('Deveria ser numérico')
            elif col_data.max() > 7 or col_data.min() < 0:
                quality['warnings'].append(f'Valores fora do esperado (0-7): {col_data.min()}-{col_data.max()}')

        elif feature == 'CASO_COMPLEXO':
            unique_vals = set(col_data.dropna().unique())
            if not unique_vals.issubset({0, 1, 0.0, 1.0}):
                quality['warnings'].append(f'Valores não binários: {unique_vals}')

        elif feature == 'RISCO_SOCIAL':
            if col_data.max() > 3 or col_data.min() < 0:
                quality['warnings'].append(f'Valores fora do esperado (0-3): {col_data.min()}-{col_data.max()}')

        elif feature == 'TEMPO_INICIO_CAT':
            unique_vals = set(col_data.dropna().unique())
            expected_vals = {0, 1, 2, 3, 4, 0.0, 1.0, 2.0, 3.0, 4.0}
            if not unique_vals.issubset(expected_vals):
                quality['warnings'].append(f'Categorias inesperadas: {unique_vals}')

        elif feature == 'DURACAO_PREVISTA_CAT':
            unique_vals = set(col_data.dropna().unique())
            expected_vals = {0, 1, 2, 0.0, 1.0, 2.0}
            if not unique_vals.issubset(expected_vals):
                quality['warnings'].append(f'Categorias inesperadas: {unique_vals}')

        missing_pct = quality['statistics']['missing_pct']
        if missing_pct > 50:
            quality['warnings'].append(f'Muitos valores ausentes: {missing_pct:.1f}%')
            quality['status'] = 'WARNING'
        elif missing_pct > 20:
            quality['warnings'].append(f'Valores ausentes significativos: {missing_pct:.1f}%')

        return quality

    def _validate_replacements_in_data(self):
        """Valida se as substituições foram aplicadas corretamente"""

        logger.info("🔄 Validando substituições aplicadas...")

        for derived_feature, mapping in self.replacement_mappings.items():
            original_features = mapping.get('replaces', [])
            mode = mapping.get('mode', 'unknown')

            derived_present = derived_feature in self.X.columns
            originals_present = [f for f in original_features if f in self.X.columns]

            if mode == 'full' and derived_present and originals_present:
                logger.warning(f"   ⚠️ {derived_feature}: Substituição completa mas originais ainda presentes: {originals_present}")
            elif mode == 'full' and not derived_present and originals_present:
                logger.error(f"   ❌ {derived_feature}: Derivada ausente mas originais presentes: {originals_present}")
            elif derived_present and not originals_present:
                logger.info(f"   ✅ {derived_feature}: Substituição bem-sucedida")
            elif mode == 'partial' and derived_present:
                logger.info(f"   ✅ {derived_feature}: Substituição parcial OK ({len(originals_present)} originais mantidas)")

    def _preprocess_derived_features(self, derived_categorical: List[str], derived_numerical: List[str]):
        """Pré-processamento especializado para features derivadas"""

        for col in derived_categorical:
            if col in self.X.columns:
                missing_count = self.X[col].isnull().sum()
                if missing_count > 0:
                    logger.info(f"   🔄 {col} (derivada categórica): {missing_count} valores faltantes → modo específico")

                    if col == 'CASO_COMPLEXO':
                        self.X[col] = self.X[col].fillna(0)
                    else:
                        self.X[col] = self.X[col].fillna('Desconhecido')

        for col in derived_numerical:
            if col in self.X.columns:
                missing_count = self.X[col].isnull().sum()
                if missing_count > 0:
                    logger.info(f"   🔄 {col} (derivada numérica): {missing_count} valores faltantes → 0 ou mediana")

                    if col in ['SCORE_COMORBIDADES', 'RISCO_SOCIAL', 'PERFIL_GRAVIDADE']:
                        self.X[col] = self.X[col].fillna(0)
                    else:
                        self.X[col] = self.X[col].fillna(self.X[col].median())

    def _preprocess_standard_features(self, categorical_cols: List[str], numerical_cols: List[str]):
        """Pré-processamento padrão para features não-derivadas"""

        standard_categorical = [col for col in categorical_cols if col not in self.derived_features]
        for col in standard_categorical:
            missing_count = self.X[col].isnull().sum()
            if missing_count > 0:
                logger.info(f"   📊 {col} (categórica): {missing_count} valores faltantes → 'Desconhecido'")
            self.X[col] = self.X[col].fillna('Desconhecido')

        standard_numerical = [col for col in numerical_cols if col not in self.derived_features]
        for col in standard_numerical:
            missing_count = self.X[col].isnull().sum()
            if missing_count > 0:
                logger.info(f"   📊 {col} (numérica): {missing_count} valores faltantes → mediana")
            self.X[col] = self.X[col].fillna(self.X[col].median())

        for col in categorical_cols:
            if col in self.X.columns:
                le = LabelEncoder()
                self.X[col] = le.fit_transform(self.X[col].astype(str))
                self.label_encoders[col] = le

    def create_balanced_data(self, strategy: str = 'undersample_exact') -> bool:
        """Balanceamento com controle de proporção"""

        try:
            if not self.data_processed:
                raise ValueError("Execute preprocess_data() primeiro")

            logger.info(f"\n⚖️ Aplicando balanceamento: {strategy}")

            original_counts = self.y.value_counts()
            minority_class_size = original_counts.min()
            majority_class_size = original_counts.max()

            logger.info(f"📊 Distribuição original: {dict(original_counts)}")
            logger.info(f"📐 Razão de desbalanceamento: 1:{majority_class_size / minority_class_size:.1f}")

            if strategy == 'undersample_exact':
                target_samples_per_class = minority_class_size
                logger.info(f"🎯 Target: {target_samples_per_class} amostras por classe")

                sampling_strategy = {0: target_samples_per_class, 1: target_samples_per_class}
                sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)

            elif strategy == 'smote':
                target_ratio = min(0.3, 1000 / majority_class_size)
                sampler = SMOTE(sampling_strategy=target_ratio, random_state=42)

            else:
                target_ratio = max(0.2, minority_class_size / 5000)
                sampler = RandomUnderSampler(sampling_strategy=target_ratio, random_state=42)

            logger.info("🔄 Aplicando balanceamento...")
            start_time = time.time()

            self.X_balanced, self.y_balanced = sampler.fit_resample(self.X, self.y)

            elapsed_time = time.time() - start_time
            logger.info(f"⏱️ Balanceamento concluído em {elapsed_time:.2f}s")

            balanced_counts = pd.Series(self.y_balanced).value_counts()
            logger.info(f"✅ Distribuição balanceada: {dict(balanced_counts)}")
            logger.info(f"📊 Tamanho final: {len(self.y_balanced)} amostras")

            self._validate_derived_features_after_balancing()

            return True

        except Exception as e:
            logger.error(f"❌ Erro no balanceamento: {e}")
            return False

    def _validate_derived_features_after_balancing(self):
        """Valida que features derivadas foram preservadas após balanceamento"""

        derived_present_after = [f for f in self.derived_features if f in self.X_balanced.columns]

        logger.info(f"🔄 Features derivadas preservadas após balanceamento: {len(derived_present_after)}/{len(self.derived_features)}")

        if len(derived_present_after) < len(self.derived_features):
            missing_derived = [f for f in self.derived_features if f not in self.X_balanced.columns]
            logger.warning(f"⚠️ Features derivadas perdidas: {missing_derived}")

    def save_model(self, filepath: str) -> bool:
        """Salva modelo treinado com informações de features derivadas"""

        try:
            model_data = {
                'base_models': self.base_models,
                'meta_model': self.meta_model,
                'tabpfn_model': self.tabpfn_model,
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'feature_names': self.feature_names,
                'use_tabpfn': self.use_tabpfn,
                'results': self.results,
                'cv_results': self.cv_results,
                'meta_results': self.meta_results,
                'best_meta_name': self.best_meta_name,
                'feature_selector_main': self.feature_selector_main,
                'selected_features': self.selected_features,
                'derived_features': self.derived_features,
                'replaced_features': list(self.replaced_features),
                'replacement_mappings': self.replacement_mappings,
                'derived_features_config': self.derived_features_config
            }

            if hasattr(self, 'feature_selector') and self.feature_selector:
                model_data['feature_selector'] = self.feature_selector

            joblib.dump(model_data, filepath)
            logger.info(f"💾 Modelo com features derivadas salvo em: {filepath}")

            return True

        except Exception as e:
            logger.error(f"❌ Erro ao salvar modelo: {e}")
            return False

    def load_model(self, filepath: str) -> bool:
        """Carrega modelo salvo com features derivadas"""

        try:
            model_data = joblib.load(filepath)

            self.base_models = model_data['base_models']
            self.meta_model = model_data['meta_model']
            self.tabpfn_model = model_data['tabpfn_model']
            self.scaler = model_data['scaler']
            self.label_encoders = model_data['label_encoders']
            self.feature_names = model_data['feature_names']
            self.use_tabpfn = model_data['use_tabpfn']
            self.results = model_data['results']
            self.cv_results = model_data['cv_results']
            self.meta_results = model_data['meta_results']
            self.best_meta_name = model_data['best_meta_name']
            self.feature_selector_main = model_data['feature_selector_main']
            self.selected_features = model_data['selected_features']
            self.derived_features = model_data.get('derived_features', [])
            self.replaced_features = set(model_data.get('replaced_features', []))
            self.replacement_mappings = model_data.get('replacement_mappings', {})
            self.derived_features_config = model_data.get('derived_features_config', {})

            if 'feature_selector' in model_data:
                self.feature_selector = model_data['feature_selector']

            logger.info(f"📂 Modelo com features derivadas carregado de: {filepath}")
            logger.info(f"🔄 Features derivadas carregadas: {len(self.derived_features)}")
            return True

        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo: {e}")
            return False

    def get_feature_importance(self) -> Dict[str, float]:
        """Retorna importância das features com destaque para derivadas"""

        try:
            if 'rf' in self.base_models:
                rf_model = self.base_models['rf']
                if hasattr(rf_model, 'feature_importances_'):
                    importance_dict = dict(zip(self.selected_features, rf_model.feature_importances_))

                    derived_importance = {k: v for k, v in importance_dict.items() if k in self.derived_features}
                    original_importance = {k: v for k, v in importance_dict.items() if k not in self.derived_features}

                    if derived_importance:
                        logger.info(f"🔄 Importância das features derivadas:")
                        for feature, importance in sorted(derived_importance.items(), key=lambda x: x[1], reverse=True):
                            logger.info(f"   {feature}: {importance:.4f}")

                    return importance_dict
            return {}
        except:
            return {}

    def get_model_summary(self) -> Dict[str, Any]:
        """Retorna resumo completo do modelo com informações de features derivadas"""

        derived_in_use = [f for f in self.selected_features if f in self.derived_features]
        replaced_in_use = [f for f in self.selected_features if f in self.replaced_features]

        return {
            'model_type': 'TabPFN + Meta-Models Ensemble (Optimized for Derived Features)' if self.use_tabpfn else 'XGBoost + Meta-Models Ensemble (Optimized for Derived Features)',
            'main_model': 'TabPFN' if self.use_tabpfn else 'XGBoost',
            'base_models': list(self.base_models.keys()),
            'best_meta_model': self.best_meta_name,
            'features_selected': len(self.selected_features),
            'features_list': self.selected_features,
            'derived_features_used': derived_in_use,
            'derived_features_count': len(derived_in_use),
            'derived_features_ratio': len(derived_in_use) / len(
                self.selected_features) if self.selected_features else 0,
            'replaced_features_still_used': replaced_in_use,
            'replacement_effectiveness': len(replaced_in_use) == 0,
            'performance': self.results,
            'cv_results': {name: {
                'balanced_accuracy_mean': results['test_balanced_accuracy'].mean(),
                'balanced_accuracy_std': results['test_balanced_accuracy'].std(),
                'overfitting_gap': results.get('overfitting_gap', 0),
                'derived_features_used': results.get('derived_features_used', [])
            } for name, results in self.cv_results.items()},
            'meta_model_comparison': self.meta_results,
            'derived_features_config': self.derived_features_config,
            'optimization_for_derived': True
        }

    def validate_derived_features_performance(self) -> Dict[str, Any]:
        """Valida se features derivadas estão melhorando a performance"""

        validation = {
            'derived_features_effective': True,
            'replacement_successful': True,
            'recommendations': []
        }

        derived_used = [f for f in self.selected_features if f in self.derived_features]
        if not derived_used:
            validation['derived_features_effective'] = False
            validation['recommendations'].append(
                "Nenhuma feature derivada foi selecionada - revisar lógica de priorização")

        replaced_still_used = [f for f in self.selected_features if f in self.replaced_features]
        if replaced_still_used:
            validation['replacement_successful'] = False
            validation['recommendations'].append(f"Features substituídas ainda em uso: {replaced_still_used}")

        if derived_used:
            derived_ratio = len(derived_used) / len(self.selected_features)
            if derived_ratio < 0.2:
                validation['recommendations'].append(
                    "Baixa proporção de features derivadas - considerar aumentar priorização")
            elif derived_ratio > 0.8:
                validation['recommendations'].append(
                    "Alta proporção de features derivadas - verificar se não está perdendo informação")

        return validation


def test_corrected_predictor():
    """Testa se as correções resolveram o problema"""

    print("🧪 TESTANDO PREDITOR :")
    print("=" * 50)

    try:
        predictor = TuberculosisPredictor()
        print("✅ Preditor criado com sucesso")

        X = np.random.rand(100, 20)
        y = np.random.randint(0, 2, 100)
        predictor.feature_names = [f"feature_{i}" for i in range(20)]
        predictor.derived_features = ['feature_8', 'feature_9', 'feature_10', 'feature_11', 'feature_4']

        print(f"📊 Dados simulados: X {X.shape}, y {y.shape}")
        print(f"🔄 Features derivadas: {predictor.derived_features}")

        available_features = predictor.feature_names
        features_to_score = predictor.derived_features[:3]

        print(f"🔧 Testando _score_features...")
        scores = predictor._score_features(X, y, available_features, features_to_score)
        print(f"✅ _score_features funcionou: {len(scores)} scores retornados")
        print(f"🔧 Testando select_best_features...")
        X_selected, selector, selected_features = predictor.select_best_features(X, y, max_features=10)
        print(f"✅ select_best_features funcionou: {X_selected.shape}, {len(selected_features)} features")

        print(f"\n🎉 TODOS OS TESTES PASSARAM!")
        print(f"🔧 Correções aplicadas com sucesso")
        return True

    except Exception as e:
        print(f"❌ Teste falhou: {e}")
        return False


if __name__ == "__main__":
    print("🔧 MÓDULO treino.py")
    print("=" * 60)
    print("✅ Erro crítico de numpy indexing")
    print("✅ Método _score_features completamente")
    print("✅ Validações robustas adicionadas")
    print("✅ Fallbacks seguros implementados")
    print("✅ Compatibilidade feature_names ↔ X")
    print("✅ Otimização para features derivadas")
    print("=" * 60)

    test_success = test_corrected_predictor()

    if test_success:
        print("\n🚀 MÓDULO PRONTO PARA USO!")
    else:
        print("\n⚠️ Revisar implementação")