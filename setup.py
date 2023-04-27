from setuptools import setup, find_namespace_packages

setup(name='mynnunet',
      packages=find_namespace_packages(include=["mynnunet", "mynnunet.*"]),
      version='0.0.1',
      description='mynnU-Net',
      author_email='jiangdaoran@163.com',
      license='Apache License Version 2.0, January 2004',
      python_requires=">=3.9",
      install_requires=[
          "torch>=2.0.0",
          "acvl-utils>=0.2",
          "dynamic-network-architectures>=0.2",
          "tqdm",
          "dicom2nifti",
          "scikit-image>=0.14",
          "medpy",
          "scipy",
          "batchgenerators>=0.25",
          "numpy",
          "scikit-learn",
          "scikit-image>=0.19.3",
          "SimpleITK>=2.2.1",
          "pandas",
          "graphviz",
          'tifffile',
          'requests',
          "nibabel",
          "matplotlib",
          "seaborn",
          "imagecodecs",
          "yacs"
      ],
      entry_points={
          'console_scripts': [
              'mynnUNet_train = mynnunet.run.run_training:run_training_entry',  # api available
          ],
      },
      keywords=['deep learning', 'image segmentation', 'medical image analysis',
                'medical image segmentation', 'nnU-Net', 'nnunet']
      )
