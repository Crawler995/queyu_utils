import setuptools


description_file = open('README.md', 'r', encoding='utf-8')

setuptools.setup(
    name='QueYu Utils',
    version='0.0.1',
    description='Utils which are frequently used in QueYu\'s code.',
    long_description=description_file.read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Crawler995/queyu_utils',
    author='queyu',
    author_email='zhang_995@foxmail.com',
    keywords=['deep learning', 'pytorch', 'utils'],
    packages=setuptools.find_packages(),
    license='MIT',
    package_dir={'queyu_utils': 'queyu_utils'}
)
