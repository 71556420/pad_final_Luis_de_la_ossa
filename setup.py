from setuptools import setup, find_packages

setup(
    name="bigdata_actividades",
    version="0.0.2",
    author="Luis de la Ossa",
    author_email="fernando.delaossa@est.iudigital.edu.co",
    description="Configuración para actividades de análisis de datos con NumPy",
    py_modules=["actividad_1", "actividad_2"],
    install_requires=[
        "numpy",
        "matplotlib"
    ]
)

