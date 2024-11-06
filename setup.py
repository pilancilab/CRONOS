from setuptools import setup, find_packages

setup(
    name='cronos',                        
    version='0.1.0',                      
    description='CRONOS: Convex Neural Networks via Operator Splitting',
    long_description=open('README.md').read(),  
    long_description_content_type='text/markdown',
    author='miria kaname feng',
    author_email='miria0@me.com',
    url='https://github.com/pilancilab/CRONOS',
    packages=find_packages(),    
    install_requires=[
        # todo
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'JAX :: GPU version todo',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
