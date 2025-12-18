from setuptools import find_packages, setup

package_name = 'task4a'

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
        'per1=task4a.perception_task4a:main',
        'per2=task4a.perception2:main',
        'per4=task4a.per4:main'
        'main1=task4a.manipulation_task4a:main',
        'mani2=task4a.manipulation2:main',
        'mani4=task4a.mani4:main',
        'perception_task4a=task4a.perception_task4a:main',
        ],
    },
)
