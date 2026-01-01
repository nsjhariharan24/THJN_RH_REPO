from setuptools import find_packages, setup

package_name = 'task4a_4784'

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
        'fruit1=task4a_4784.manipulation_fruit1:main',
        'fruit=task4a_4784.manipulation_fruit:main',
        'can=task4a_4784.manipulation_can:main',
        'per=task4a_4784.perception1:main',
        'combine=task4a_4784.manipulation_combine:main',
        'roll=task4a_4784.test_roll:main',
        'yaw=task4a_4784.test_yaw:main',
        'pitch=task4a_4784.test_pitch:main',
        ],
    },
)
