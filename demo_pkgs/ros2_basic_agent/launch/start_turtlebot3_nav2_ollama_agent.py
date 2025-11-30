from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Include the Nav2 simulation launch file
    nav2_launch_file_dir = os.path.join(get_package_share_directory('nav2_bringup'), 'launch')
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([nav2_launch_file_dir, '/tb3_simulation_launch.py']),
        launch_arguments={
            'headless': 'False',
            'slam': 'True'
        }.items()
    )

    # Define the AI agent node with a delay. Prefer running the installed
    # console script (Node) when the package is installed; otherwise run the
    # script directly from the package source using python3 so the launch
    # works from the workspace without an install step.
    pkg_share_dir = get_package_share_directory('ros2_basic_agent')
    # likely script locations in source layout
    candidate_scripts = [
        os.path.join(pkg_share_dir, 'ros2_ai_agent_ollama_turtlebo3_nav2.py'),
        os.path.abspath(os.path.join(pkg_share_dir, '..', 'ros2_basic_agent', 'ros2_ai_agent_ollama_turtlebo3_nav2.py')),
        os.path.join(pkg_share_dir, 'ros2_ai_agent_turtlebo3_nav2_ollama.py'),
    ]

    script_path = None
    for c in candidate_scripts:
        if os.path.exists(c):
            script_path = c
            break

    if script_path:
        from launch.actions import ExecuteProcess

        delayed_agent = TimerAction(
            period=10.0,
            actions=[
                # run directly from source with python3
                ExecuteProcess(
                    cmd=['python3', script_path],
                    output='screen',
                    emulate_tty=True,
                )
            ]
        )
    else:
        # Fall back to executing the script from the package share (handles
        # workspace installs that place the script under share) or the
        # installed console entry (Node will be used by users who installed).
        delayed_agent = TimerAction(
            period=10.0,
            actions=[
                Node(
                    package='ros2_basic_agent',
                    executable='ros2_ai_ollama_agent_nav2',
                    name='ros2_ai_ollama_agent_turtlebot3',
                    output='screen',
                    emulate_tty=True,
                )
            ]
        )

    return LaunchDescription([
        nav2_launch,
        delayed_agent
    ])
