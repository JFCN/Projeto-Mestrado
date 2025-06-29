import os
import shutil
import sys
from pathlib import Path
import yaml

def create_config_files():
    """Cria arquivos de configuração se não existirem"""

    if not Path("config.yaml").exists():
        print("⚠️ config.yaml não encontrado. Crie o arquivo de configuração principal.")
        return False

    if not Path("config_local.yaml").exists():
        if Path("config_local.yaml.example").exists():
            shutil.copy("config_local.yaml.example", "config_local.yaml")
            print("✅ config_local.yaml criado!")
        else:
            print("⚠️ config_local.yaml.example não encontrado")

    gitignore_content = """
# Configurações locais
config_local.yaml

# Resultados e logs
**/tb_ai_results/
*.log

# Cache Python
__pycache__/
*.pyc
*.pyo

# Jupyter
.ipynb_checkpoints/

# Dados
*.csv
*.xlsx
data/
datasets/

# Modelos treinados
*.pkl
*.joblib
models/

# Temporários
temp/
tmp/
.tmp/
"""

    gitignore_path = Path(".gitignore")
    if not gitignore_path.exists():
        with open(gitignore_path, 'w') as f:
            f.write(gitignore_content)

    return True

def validate_environment():
    """Valida se o ambiente está configurado corretamente"""

    issues = []

    if sys.version_info < (3, 8):
        issues.append("Python 3.8+ é necessário")

    required_modules = [
        'pandas', 'numpy', 'scikit-learn',
        'matplotlib', 'seaborn', 'yaml'
    ]

    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            issues.append(f"Módulo {module} não encontrado")

    optional_modules = {
        'docx': 'python-docx (relatórios Word)',
        'tabpfn': 'TabPFN (modelo principal)',
        'xgboost': 'XGBoost (fallback)'
    }

    missing_optional = []
    for module, description in optional_modules.items():
        try:
            __import__(module)
        except ImportError:
            missing_optional.append(f"{module} ({description})")

    required_dirs = ['modules']
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            issues.append(f"Diretório {dir_name}/ não encontrado")

    sys.path.append('.')
    sys.path.append('modules')

    system_modules = [
        'modules.analisador_temporal',
        'modules.detector_dataleak',
        'modules.treino',
        'modules.relatorio'
    ]

    for module in system_modules:
        try:
            __import__(module)
        except ImportError as e:
            issues.append(f"Módulo do sistema {module} com problemas: {e}")

    if issues:
        print("❌ PROBLEMAS ENCONTRADOS:")
        for issue in issues:
            print(f"   • {issue}")
    else:
        print("✅ Ambiente validado com sucesso!")

    if missing_optional:
        print("\n⚠️ MÓDULOS OPCIONAIS AUSENTES:")
        for module in missing_optional:
            print(f"   • {module}")
        print("💡 Instale com: pip install python-docx tabpfn xgboost")

    return len(issues) == 0

def interactive_config():
    """Configuração interativa"""

    config_path = Path("config_local.yaml")
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    if 'data' not in config:
        config['data'] = {}
    if 'system' not in config:
        config['system'] = {}
    if 'model' not in config:
        config['model'] = {}

    print("\n📁 CONFIGURAÇÃO DE CAMINHOS:")

    current_input = config.get('data', {}).get('input_path', '')
    print(f"Caminho atual dos dados: {current_input or 'Não definido'}")
    new_input = input("Novo caminho dos dados (Enter para manter): ").strip()
    if new_input:
        config['data']['input_path'] = new_input

    current_output = config.get('data', {}).get('output_base_dir', '')
    print(f"Diretório de saída atual: {current_output or 'Não definido'}")
    new_output = input("Novo diretório de saída (Enter para manter): ").strip()
    if new_output:
        config['data']['output_base_dir'] = new_output

    print("\n⚙️ CONFIGURAÇÕES DO MODELO:")

    current_tabpfn = config.get('model', {}).get('use_tabpfn', True)
    print(f"Usar TabPFN: {current_tabpfn}")
    use_tabpfn = input("Habilitar TabPFN? (y/n, Enter para manter): ").strip().lower()
    if use_tabpfn in ['y', 'yes', 'sim', 's']:
        config['model']['use_tabpfn'] = True
    elif use_tabpfn in ['n', 'no', 'não', 'nao']:
        config['model']['use_tabpfn'] = False

    current_features = config.get('model', {}).get('max_features', 15)
    print(f"Máximo de features: {current_features}")
    max_features = input("Novo máximo de features (Enter para manter): ").strip()
    if max_features and max_features.isdigit():
        config['model']['max_features'] = int(max_features)

    current_cv = config.get('model', {}).get('cv_folds', 10)
    print(f"Folds de cross-validation: {current_cv}")
    cv_folds = input("Novos folds CV (Enter para manter): ").strip()
    if cv_folds and cv_folds.isdigit():
        config['model']['cv_folds'] = int(cv_folds)

    print("\n🎨 CONFIGURAÇÕES DE RELATÓRIOS:")

    if 'reports' not in config:
        config['reports'] = {}

    current_word = config.get('reports', {}).get('generate_word', True)
    print(f"Gerar relatórios Word: {current_word}")
    gen_word = input("Gerar relatórios Word? (y/n, Enter para manter): ").strip().lower()
    if gen_word in ['y', 'yes', 'sim', 's']:
        config['reports']['generate_word'] = True
    elif gen_word in ['n', 'no', 'não', 'nao']:
        config['reports']['generate_word'] = False

    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    print("✅ Configuração salva!")
    return True

def show_usage_examples():
    print("EXEMPLOS DE USO:")

    examples = [
        {
            'title': 'Execução básica',
            'command': 'python main.py',
            'description': 'Usa configurações padrão'
        },
        {
            'title': 'Especificar dados',
            'command': 'python main.py --data-path "dados.csv" --output-dir "resultados/"',
            'description': 'Define caminhos via linha de comando'
        },
        {
            'title': 'Modo debug',
            'command': 'python main.py --debug --verbose',
            'description': 'Execução com logs detalhados'
        },
        {
            'title': 'Sem TabPFN',
            'command': 'python main.py --no-tabpfn --max-features 10',
            'description': 'Desabilita TabPFN e limita features'
        },
        {
            'title': 'Apenas relatório técnico',
            'command': 'python main.py --no-word --no-viz',
            'description': 'Gera apenas relatório técnico'
        },
        {
            'title': 'Variáveis de ambiente',
            'command': 'TB_AI_USE_TABPFN=false TB_AI_MAX_FEATURES=5 python main.py',
            'description': 'Configuração via variáveis de ambiente'
        }
    ]

    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['title']}: {example['command']}")

    env_vars = [
        'TB_AI_LOG_LEVEL', 'TB_AI_USE_TABPFN', 'TB_AI_MAX_FEATURES',
        'TB_AI_CV_FOLDS', 'TB_AI_INPUT_PATH', 'TB_AI_OUTPUT_DIR',
        'TB_AI_TEMPORAL_THRESHOLD', 'TB_AI_GENERATE_WORD'
    ]

    print("VARIÁVEIS DE AMBIENTE:")
    for var in env_vars:
        print(f"   {var}")

def main():
    print("SISTEMA IA TUBERCULOSE INFANTIL - CONFIGURAÇÃO")

    if not create_config_files():
        print("Falha na criação dos arquivos de configuração")
        return 1

    if not validate_environment():
        print("Ambiente com problemas, continuando...")

    response = input("Configurar interativamente? (y/n): ").strip().lower()

    if response in ['y', 'yes', 'sim', 's']:
        interactive_config()

    show_usage_examples()

    print("CONFIGURAÇÃO CONCLUÍDA")
    print("PRÓXIMOS PASSOS:")
    print("1. Edite config_local.yaml com seus caminhos específicos")
    print("2. Execute: python main.py")
    print("3. Verifique os resultados em tb_ai_results/")

    return 0

if __name__ == "__main__":
    exit(main())