from setuptools import setup

setup(name='visualconv',
      version='0.2',
      description='Visualise different output layers of your Keras model, for a given image.',
      #url='http://github.com/storborg/funniest',
      author='Bharath A',
      author_email='bbharath201@gmail.com',
      #license='MIT',
      packages=['visualconv'],
      url="https://github.com/BharathA14/Visualconv/",
      install_requires=[
          'numpy',
          'tensorflow',
          'keras',
          'opencv-python',
          'matplotlib'
      ],
      zip_safe=False)