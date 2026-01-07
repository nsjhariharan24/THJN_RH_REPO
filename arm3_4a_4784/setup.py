from setuptools import find_packages, setup

package_name = 'arm3_4a_4784'

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
        'per=arm3_4a_4784.perception1:main',
        'fruit3=arm3_4a_4784.fruit3:main',
        'fruit_can=arm3_4a_4784.fruit_can:main',
        'fruit_can_p=arm3_4a_4784.fruit_can_p:main',
        'tcp=arm3_4a_4784.tcp:main',
        'test=arm3_4a_4784.test:main',
        'waytest=arm3_4a_4784.waytest:main',
        ],
    },
)
