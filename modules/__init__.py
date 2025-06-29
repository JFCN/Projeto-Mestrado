"""
Módulos do Sistema

Módulos disponíveis:
- analisador_temporal: Análise da linha temporal clínica
- analisador_leake: Detecção básica de data leakage
- treino: Treinamento TabPFN + Meta-models
- validacao_hipotese: Teste de Hipótese
- relatorio: Geração de relatórios completos

Autor: Janduhy Finizola da Cunha Neto
"""

__version__ = "1.0.0"
__author__ = "Janduhy Finizola da Cunha Neto"

try:
    from .analisador_temporal import ClinicalTimelineAnalyzer
    __all__ = ['ClinicalTimelineAnalyzer']
    print("✅ Módulo analisador_temporal carregado")
except ImportError as e:
    print(f"⚠️ analisador_temporal não disponível: {e}")

try:
    from .detector_dataleak import DataLeakageDetector
    __all__.append('DataLeakageDetector')
    print("✅ Módulo analisador_leake carregado")
except ImportError as e:
    print(f"⚠️ analisador_leake não disponível: {e}")

try:
    from .treino import TuberculosisPredictor
    __all__.append('TuberculosisPredictor')
    print("✅ Módulo treino carregado")
except ImportError as e:
    print(f"⚠️ treino não disponível: {e}")

try:
    from .validacao_hipotese import MedicalStatisticalValidator
    __all__.append('MedicalStatisticalValidator')
    print("✅ Módulo teste de hipótese carregado")
except ImportError as e:
    print(f"⚠️ teste de hipótese não disponível: {e}")

try:
    from .relatorio import ComprehensiveReportGenerator
    __all__.append('ComprehensiveReportGenerator')
    print("✅ Módulo relatorio carregado")
except ImportError as e:
    print(f"⚠️ relatorio não disponível: {e}")

def get_available_modules():
    """Retorna lista de módulos disponíveis"""
    return __all__ if '__all__' in globals() else []

def validate_module_dependencies():
    """Valida dependências dos módulos"""

    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 
        'matplotlib', 'seaborn', 'docx'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Pacotes faltantes: {missing_packages}")
        return False
    else:
        print("✅ Todas as dependências estão instaladas")
        return True