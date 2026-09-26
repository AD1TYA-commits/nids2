from setuptools import setup, find_packages

setup(
    name="nids-ae-dsp",
    version="1.0.0",
    description="Network Intrusion Detection System built on Pretrained AE-DSP Framework",
    author="Antigravity",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "lightgbm",
        "shap",
        "torch",
        "joblib",
    ],
)
