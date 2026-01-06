from setuptools import find_packages, setup

package_name = 'ebot3_4b'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hari',
    maintainer_email='nsjhariharan24@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        'nav=ebot3_4b.nav:main',
        'shape=ebot3_4b.nav1:main',
        ],
    },
)
