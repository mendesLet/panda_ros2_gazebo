# Copyright (C) 2021 Bosch LLC CR, North America. All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

import sys

# ROS2 Python API libraries
import argparse, sys
import rclpy
from rclpy.utilities import remove_ros_args

# Panda example imports
from .examples.panda_teleop_control import PandaTeleopControl
from .examples.panda_follow_trajectory import PandaFollowTrajectory
from .examples.panda_pick_n_place import PandaPickAndPlace
from .examples.panda_pick_n_insert import PandaPickAndInsert
from .examples.panda_task_library import PandaTaskLibrary

def main(args=None):
    rclpy.init(args=args)
    user_argv = remove_ros_args(sys.argv)[1:]  # drop program name
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--mode', required=True,
                        choices=['follow','picknplace','pickninsert','teleop','tasklibrary'],)
    mode = parser.parse_args(user_argv).mode

    if mode == 'follow':
        node = PandaFollowTrajectory()
    elif mode == 'picknplace':
        node = PandaPickAndPlace()
    elif mode == 'pickninsert':
        node = PandaPickAndInsert()
    elif mode == 'tasklibrary':
        node = PandaTaskLibrary()
    else:
        node = PandaTeleopControl()

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()