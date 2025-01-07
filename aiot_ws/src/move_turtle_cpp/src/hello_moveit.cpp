// #include "geometry_msgs/msg/pose.hpp"
// #include "moveit/move_group_interface/move_group_interface.h"
// #include "rclcpp/duration.hpp"
#include "rclcpp/rclcpp.hpp"
// #include "tf2_ros/buffer.h"
// #include <chrono>

// using namespace std::chrono_literals;

// class Hello_moveit : public rclcpp::Node
// {
// public:
//     Hello_moveit()
//         : Node("hello_moveit")
//     {
//         RCLCPP_INFO(get_logger(), "Hello Moveit");
//         _msg.orientation.w = 1.0;
//         _msg.position.x = 0.28;
//         _msg.position.y = -0.2;
//         _msg.position.z = 0.5;
//     }
//     void setnode(std::shared_ptr<rclcpp::Node> node)
//     {
//         node_shared_ptr_ = node;
//         using moveit::planning_interface::MoveGroupInterface;
//         // auto node_shared_ptr = shared_from_this();
//         auto move_group_interface = MoveGroupInterface(node_shared_ptr_, "panda_arm");
//         move_group_interface.setPoseTarget(_msg);
//         auto const ok = static_cast<bool>(move_group_interface.plan(_plan));
//         if (ok)
//         {
//             RCLCPP_INFO(get_logger(), "Plan Success");
//             move_group_interface.execute(_plan);
//         }
//         else
//         {
//             RCLCPP_INFO(get_logger(), "Plan Failed");
//         }
//     }

// private:
//     geometry_msgs::msg::Pose _msg;
//     moveit::planning_interface::MoveGroupInterface::Plan _plan;
//     std::shared_ptr<rclcpp::Node> node_shared_ptr_;
// };

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    // auto node = std::make_shared<Hello_moveit>();
    auto node = std::make_shared<rclcpp::Node>("hello_moveit");
    // node->setnode(node);
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
