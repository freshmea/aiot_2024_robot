import rclpy
from rclpy.node import Node


def main():
    rclpy.init()
    node = Node("hello")
    print("hello, ros2! nice to meet you!")
    print("this is simlink really!!!")
    
    rclpy.spin(node)

if __name__ == "__main__":
    main()
