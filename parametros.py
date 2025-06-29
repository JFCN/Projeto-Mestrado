import datetime
import json
from pathlib import Path
from typing import Dict, Any, Optional


class TuberculosisModelParamsGenerator:
    """
    Gerador de parâmetros específico para o projeto de tuberculose
    Extrai parâmetros dos modelos já configurados no seu projeto
    """

    def __init__(self, output_file="tuberculosis_ml_parameters.txt"):
        self.output_file = output_file
        self.models_params = {}

    def extract_base_models_params(self) -> Dict[str, Dict]:
        """Extrai parâmetros dos modelos base do seu projeto"""

        lr_params = {
            'C': 0.1,
            'class_weight': 'balanced',
            'random_state': 42,
            'max_iter': 1000,
            'penalty': 'l2',
            'solver': 'liblinear',
            'optimization_target': 'features_categoricas_derivadas'
        }

        rf_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'min_samples_split': 15,
            'min_samples_leaf': 8,
            'max_features': 'sqrt',
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1,
            'bootstrap': True,
            'oob_score': True,
            'optimization_target': 'features_categoricas_derivadas'
        }

        xgb_params = {
            'n_estimators': 50,
            'max_depth': 4,
            'learning_rate': 0.05,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.5,
            'reg_lambda': 0.5,
            'min_child_weight': 15,
            'gamma': 0.3,
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'logloss',
            'verbosity': 0,
            'tree_method': 'hist',
            'grow_policy': 'lossguide',
            'optimization_target': 'features_categoricas_derivadas'
        }

        return {
            'Logistic_Regression_Base': lr_params,
            'Random_Forest_Base': rf_params,
            'XGBoost_Base': xgb_params
        }

    def extract_meta_models_params(self) -> Dict[str, Dict]:
        """Extrai parâmetros dos meta-modelos candidatos"""

        meta_lr_params = {
            'C': 0.05,
            'class_weight': 'balanced',
            'random_state': 42,
            'max_iter': 1000,
            'penalty': 'l2',
            'solver': 'liblinear',
            'model_role': 'meta_model_candidate'
        }

        meta_rf_params = {
            'n_estimators': 30,
            'max_depth': 4,
            'min_samples_split': 15,
            'min_samples_leaf': 8,
            'max_features': 'sqrt',
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1,
            'model_role': 'meta_model_candidate'
        }

        meta_xgb_params = {
            'n_estimators': 25,
            'max_depth': 3,
            'learning_rate': 0.01,
            'reg_alpha': 1.0,
            'reg_lambda': 1.0,
            'min_child_weight': 20,
            'gamma': 0.5,
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'logloss',
            'verbosity': 0,
            'model_role': 'meta_model_candidate'
        }

        meta_svm_params = {
            'C': 0.1,
            'kernel': 'rbf',
            'class_weight': 'balanced',
            'probability': True,
            'gamma': 'scale',
            'random_state': 42,
            'model_role': 'meta_model_candidate'
        }

        return {
            'Meta_Logistic_Regression': meta_lr_params,
            'Meta_Random_Forest': meta_rf_params,
            'Meta_XGBoost': meta_xgb_params,
            'Meta_SVM': meta_svm_params
        }

    def extract_tabpfn_params(self) -> Dict[str, Dict]:
        """Extrai parâmetros do TabPFN e fallback XGBoost"""

        tabpfn_params = {
            'device': 'cpu',
            'random_state': 42,
            'max_samples': 1000,
            'max_features': 100,
            'optimization_target': 'features_derivadas',
            'fallback_available': True,
            'model_role': 'primary_model'
        }

        xgb_fallback_params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'logloss',
            'verbosity': 0,
            'tree_method': 'hist',
            'grow_policy': 'lossguide',
            'max_leaves': 0,
            'gamma': 0.1,
            'scale_pos_weight': 'auto_calculated',
            'model_role': 'fallback_model',
            'optimization_target': 'features_derivadas_categoricas'
        }

        return {
            'TabPFN_Primary': tabpfn_params,
            'XGBoost_Fallback': xgb_fallback_params
        }

    def extract_pipeline_config(self) -> Dict[str, Any]:
        """Extrai configurações do pipeline de treinamento"""

        return {
            'Pipeline_Configuration': {
                'use_tabpfn': True,
                'tabpfn_fallback': 'xgboost',
                'cv_folds': 10,
                'cv_strategy': 'stratified',
                'test_size': 0.2,
                'validation_size': 0.2,
                'random_state': 42,
                'balancing_strategy': 'undersample_exact',
                'meta_model_selection': 'auto',
                'meta_model_cv_folds': 5,
                'anti_overfitting': True,
                'data_split': '60_20_20_train_val_test'
            },
            'Derived_Features_Config': {
                'prioritize_derived_in_selection': True,
                'derived_feature_weight_multiplier': 1.2,
                'validate_derived_performance': True,
                'max_derived_features': 8,
                'min_correlation_threshold': 0.05,
                'min_original_features': 5
            },
            'Leakage_Detection_Config': {
                'check_obvious_leakage_only': True,
                'temporal_threshold': 0.10,
                'missing_threshold': 0.20,
                'auc_difference_threshold': 0.20,
                'treat_derived_as_safe': True,
                'validate_derived_temporal_safety': True,
                'mark_replaced_as_safe': True
            }
        }

    def extract_all_models_params(self):
        """Extrai todos os parâmetros dos modelos do projeto"""

        self.models_params.update(self.extract_base_models_params())
        self.models_params.update(self.extract_meta_models_params())
        self.models_params.update(self.extract_tabpfn_params())
        self.models_params.update(self.extract_pipeline_config())

    def add_custom_model_params(self, model_name: str, params: Dict[str, Any]):
        """Adiciona parâmetros de modelo customizado"""
        self.models_params[model_name] = params

    def generate_txt_report(self, include_metadata=True):
        """Gera relatório TXT completo dos parâmetros"""

        self.extract_all_models_params()

        with open(self.output_file, 'w', encoding='utf-8') as f:
            if include_metadata:
                f.write("TUBERCULOSIS ML PROJECT - MODEL PARAMETERS REPORT\n")
                f.write("=" * 80 + "\n")
                f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Project: Sistema IA Tuberculose Infantil\n")
                f.write(f"Architecture: TabPFN + Meta-Models + Anti-Overfitting\n")
                f.write(f"Optimization: Features Derivadas Categóricas\n")
                f.write("=" * 80 + "\n\n")

            f.write("1. MODELOS BASE (Base Models)\n")
            f.write("-" * 50 + "\n")
            base_models = ['Logistic_Regression_Base', 'Random_Forest_Base', 'XGBoost_Base']
            for model_name in base_models:
                if model_name in self.models_params:
                    f.write(f"\n{model_name}:\n")
                    for param, value in self.models_params[model_name].items():
                        f.write(f"  {param}: {value}\n")
            f.write("\n" + "=" * 80 + "\n\n")

            f.write("2. MODELO PRINCIPAL (Primary Model)\n")
            f.write("-" * 50 + "\n")
            primary_models = ['TabPFN_Primary', 'XGBoost_Fallback']
            for model_name in primary_models:
                if model_name in self.models_params:
                    f.write(f"\n{model_name}:\n")
                    for param, value in self.models_params[model_name].items():
                        f.write(f"  {param}: {value}\n")
            f.write("\n" + "=" * 80 + "\n\n")

            f.write("3. META-MODELOS (Meta-Model Candidates)\n")
            f.write("-" * 50 + "\n")
            meta_models = ['Meta_Logistic_Regression', 'Meta_Random_Forest', 'Meta_XGBoost', 'Meta_SVM']
            for model_name in meta_models:
                if model_name in self.models_params:
                    f.write(f"\n{model_name}:\n")
                    for param, value in self.models_params[model_name].items():
                        f.write(f"  {param}: {value}\n")
            f.write("\n" + "=" * 80 + "\n\n")

            f.write("4. CONFIGURAÇÕES DO PIPELINE\n")
            f.write("-" * 50 + "\n")
            config_sections = ['Pipeline_Configuration', 'Derived_Features_Config', 'Leakage_Detection_Config']
            for config_name in config_sections:
                if config_name in self.models_params:
                    f.write(f"\n{config_name}:\n")
                    for param, value in self.models_params[config_name].items():
                        f.write(f"  {param}: {value}\n")
            f.write("\n" + "=" * 80 + "\n\n")
            f.write("5. RESUMO DO PROJETO\n")
            f.write("-" * 50 + "\n")
            f.write("Arquitetura: TabPFN como modelo principal com ensemble de meta-modelos\n")
            f.write("Otimização: Features derivadas categóricas com substituições inteligentes\n")
            f.write("Anti-Overfitting: Divisão 60-20-20, CV rigoroso, validação temporal\n")
            f.write("Balanceamento: Undersample exato para datasets desbalanceados\n")
            f.write("Leakage Detection: Temporal validation com proteção para features derivadas\n")
            f.write(f"Total de configurações: {len(self.models_params)} seções\n")
            f.write("=" * 80 + "\n")

        print(f"✅ Relatório de parâmetros gerado: {self.output_file}")

    def save_as_json(self, json_file=None):
        """Salva parâmetros em formato JSON"""

        if json_file is None:
            json_file = self.output_file.replace('.txt', '.json')

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.models_params, f, indent=2, default=str)

        print(f"✅ Arquivo JSON gerado: {json_file}")


def generate_tuberculosis_params(output_dir="./"):
    """
    Função principal para gerar parâmetros do projeto tuberculose
    """

    output_file = Path(output_dir) / "tuberculosis_ml_parameters.txt"
    generator = TuberculosisModelParamsGenerator(str(output_file))
    generator.generate_txt_report(include_metadata=True)
    generator.save_as_json()

    return str(output_file)


if __name__ == "__main__":
    print("🚀 Gerando parâmetros dos modelos do projeto tuberculose...")

    output_file = generate_tuberculosis_params("./")

    print(f"\n📋 Arquivos gerados:")
    print(f"   📄 TXT: {output_file}")
    print(f"   📄 JSON: {output_file.replace('.txt', '.json')}")
    print("\n✅ Parâmetros extraídos com sucesso!")


def extract_from_trained_predictor(predictor_instance, output_file="trained_model_params.txt"):
    """
    Extrai parâmetros de uma instância já treinada do TuberculosisPredictor
    VERSÃO MELHORADA - captura tudo do modelo treinado
    """
    generator = TuberculosisModelParamsGenerator(output_file)

    # Extrai parâmetros padrão
    generator.extract_all_models_params()

    # NOVO: Adiciona parâmetros REAIS dos modelos treinados
    if hasattr(predictor_instance, 'base_models') and predictor_instance.base_models:
        trained_params = {}
        for name, model in predictor_instance.base_models.items():
            if hasattr(model, 'get_params'):
                trained_params[f"Real_Trained_{name}"] = model.get_params()

        if trained_params:
            generator.models_params.update(trained_params)

    # NOVO: Adiciona parâmetros do TabPFN se usado
    if hasattr(predictor_instance, 'tabpfn_model') and predictor_instance.tabpfn_model:
        try:
            if hasattr(predictor_instance.tabpfn_model, 'get_params'):
                generator.models_params['Real_Trained_TabPFN'] = predictor_instance.tabpfn_model.get_params()
        except:
            generator.models_params['Real_Trained_TabPFN'] = {
                'status': 'TabPFN model present but params not accessible'}

    # NOVO: Adiciona parâmetros do meta-modelo
    if hasattr(predictor_instance, 'meta_model') and predictor_instance.meta_model:
        if hasattr(predictor_instance.meta_model, 'get_params'):
            generator.models_params['Real_Trained_Meta_Model'] = {
                'tipo': getattr(predictor_instance, 'best_meta_name', 'Unknown'),
                'parametros': predictor_instance.meta_model.get_params()
            }

    # Adiciona resultados do treinamento
    if hasattr(predictor_instance, 'results') and predictor_instance.results:
        generator.models_params['Resultados_Treinamento_Real'] = predictor_instance.results

    # NOVO: Adiciona resultados de cross-validation
    if hasattr(predictor_instance, 'cv_results') and predictor_instance.cv_results:
        generator.models_params['CV_Results_Real'] = predictor_instance.cv_results

    # NOVO: Adiciona informações completas de features derivadas
    generator.models_params['Features_Info_Real'] = {
        'derived_features_list': getattr(predictor_instance, 'derived_features', []),
        'replaced_features': list(getattr(predictor_instance, 'replaced_features', [])),
        'selected_features': getattr(predictor_instance, 'selected_features', []),
        'feature_names': getattr(predictor_instance, 'feature_names', []),
        'total_features_used': len(getattr(predictor_instance, 'selected_features', [])),
        'derivadas_no_modelo': len([f for f in getattr(predictor_instance, 'selected_features', [])
                                    if f in getattr(predictor_instance, 'derived_features', [])])
    }

    generator.generate_txt_report(include_metadata=True)
    generator.save_as_json()

    return output_file


# NOVA FUNÇÃO: Extrai de resultados do pipeline completo
def extract_from_pipeline_results(pipeline_results, config_used, training_results,
                                  output_file="parametros_pipeline_completo.txt"):
    """
    Extrai parâmetros de resultados completos do pipeline de treinamento
    """
    generator = TuberculosisModelParamsGenerator(output_file)

    # Extrai parâmetros padrão
    generator.extract_all_models_params()

    # Adiciona configuração realmente usada
    generator.models_params['Configuracao_Real_Usada'] = config_used

    # Adiciona resultados detalhados do pipeline
    generator.models_params['Resultados_Pipeline'] = pipeline_results

    # Adiciona detalhes dos cenários de treinamento
    if training_results:
        scenarios_details = {}
        for scenario, result in training_results.items():
            if 'predictor' in result:
                predictor = result['predictor']
                scenarios_details[scenario] = {
                    'metrics': result.get('metrics', {}),
                    'features_used': result.get('features_used', []),
                    'derived_features_used': result.get('derived_features_used', []),
                    'best_meta_name': result.get('best_meta_name', 'N/A'),
                    'cv_results': getattr(predictor, 'cv_results', {}),
                    'feature_selector_info': {
                        'selected_features': getattr(predictor, 'selected_features', []),
                        'derived_features': getattr(predictor, 'derived_features', []),
                        'replaced_features': list(getattr(predictor, 'replaced_features', []))
                    }
                }

        generator.models_params['Detalhes_Cenarios_Treino'] = scenarios_details

    generator.generate_txt_report(include_metadata=True)
    generator.save_as_json()

    return output_file