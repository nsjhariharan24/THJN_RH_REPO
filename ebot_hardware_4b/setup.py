from setuptools import find_packages, setup

package_name = 'ebot_hardware_4b'

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
        'ebot=ebot_hardware_4b.task4b_ebot:main',
        'shape=ebot_hardware_4b.shape_detector_task4b:main',
        'en=ebot_hardware_4b.en:main',
        'sd=ebot_hardware_4b.sd:main',
        'sh1=ebot_hardware_4b.sh1:main',
        'en1=ebot_hardware_4b.en1:main'
        ],
    },
)
