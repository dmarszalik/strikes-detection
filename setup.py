from setuptools import setup, find_packages

setup(
    name='strikes-detection',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'super-gradients==3.5.0',
        'numpy==1.23.5',
        'matplotlib==3.8.2',
        'opencv-python==4.8.1.78',
        'ultralytics==8.0.222',
        'torch==2.1.1',
        'scikit-learn==1.3.2',
    ],
)
