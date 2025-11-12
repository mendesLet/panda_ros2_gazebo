# bringup.launch.py
import launch
from launch.substitutions import LaunchConfiguration
import launch_ros
import os

def generate_launch_description():
    name = LaunchConfiguration('mode', default='')
    pkg_share = launch_ros.substitutions.FindPackageShare(
        package='panda_ros2_gazebo').find('panda_ros2_gazebo')
    parameter_file_path = os.path.join(pkg_share, "config", "params.yaml")

    panda_node = launch_ros.actions.Node(
        package='panda_ros2_gazebo',
        executable='panda',                 # your runner
        name='panda',
        parameters=[parameter_file_path, {'share_dir': pkg_share}],
        output='screen',
        arguments=['--mode', name]
    )

    # NL bridge that subscribes to /nl_command and sends ManipTask goals
    nl_bridge = launch_ros.actions.Node(
        package='nl_command_bridge',
        executable='node',      # make sure this entry point exists
        name='nl_bridge',
        parameters=[{'status_topic': '/manip_task/status',
                     'action_name': 'manip_task',
                     'spawn_reference_frame': 'world',
                     'default_box_size': 0.05}],
        output='screen'
    )

    return launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(
            'mode', default_value='tasklibrary',
            description='Use tasklibrary to run the action server'),
        panda_node,
        nl_bridge
    ])
