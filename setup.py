from setuptools import setup

setup(name='The_Payne',
      version='0.1',
      description='Tools for interpolating spectral models with neural networks.',
      author='Yuan-Sen Ting',
      author_email='ting@ias.edu',
      license='MIT',
      url='https://github.com/tingyuansen/The_Payne',
      package_dir = {'The_Payne/': ''},
      packages=[],
      package_data={},
      dependency_links = ['https://github.com/jobovy/apogee/tarball/master#egg=apogee'],
      install_requires=['numpy','scipy','matplotlib', 'apogee'])
