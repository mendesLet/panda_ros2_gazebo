# runner.py
import argparse, sys
import rclpy
from rclpy.utilities import remove_ros_args
from rclpy.executors import MultiThreadedExecutor

# swap the import to the action-enabled node
from .examples.panda_task_library import PandaTaskLibrary
from .examples.panda_teleop_control import PandaTeleopControl
from .examples.panda_follow_trajectory import PandaFollowTrajectory
from .examples.panda_pick_n_place import PandaPickAndPlace
from .examples.panda_pick_n_insert import PandaPickAndInsert

def main(args=None):
    rclpy.init(args=args)
    user_argv = remove_ros_args(sys.argv)[1:]
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--mode', required=True,
                        choices=['follow','picknplace','pickninsert','teleop','tasklibrary'])
    mode = parser.parse_args(user_argv).mode

    if mode == 'follow':
        node = PandaFollowTrajectory()
    elif mode == 'picknplace':
        node = PandaPickAndPlace()
    elif mode == 'pickninsert':
        node = PandaPickAndInsert()
    elif mode == 'tasklibrary':
        node = PandaTaskLibrary()  # this is now the ActionServer version
    else:
        node = PandaTeleopControl()

    # use multithreaded executor so action execute + joint-state callbacks can run
    exec = MultiThreadedExecutor(num_threads=2)
    exec.add_node(node)
    try:
        exec.spin()
    finally:
        exec.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
