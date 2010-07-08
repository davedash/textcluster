from setuptools import setup, find_packages


setup(
        name='textcluster',
        version='0.0.2',
        description='Tool to cluster documents according to text similarity.',
        long_description=open('README.rst').read(),
        author='Dave Dash',
        author_email='dd+pypi@davedash.com',
        url='http://github.com/davedash/textcluster',
        license='BSD',
        packages=find_packages(),
        include_package_data=True,
        install_requires=['stemming'],
        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Developers',
            'License :: OSI Approved :: BSD License',
            'Operating System :: OS Independent',
            'Programming Language :: Python',
            'Topic :: Text Processing :: Indexing',
            'Topic :: Software Development :: Libraries :: Python Modules',
        ],
    )

