import rclpy
from rclpy.node import Node


def print_hello():
    print("hello, ros2! nice to meet you!")
    print("this is simlink really!!!")

def main():
    rclpy.init()
    node = Node("hello")
    node.create_timer(1, print_hello)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
