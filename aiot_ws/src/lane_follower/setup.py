from setuptools import find_packages, setup

package_name = 'lane_follower'

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
    maintainer='aa',
    maintainer_email='freshmea@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "lane_follower_node = lane_follower.lane_follower_node:main",
            "load_follower_node = lane_follower.load_follower_node:main",
            "load_map_slam = lane_follower.load_map_slam:main",
        ],
    },
)
