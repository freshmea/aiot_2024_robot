#include "geometry_msgs/msg/transform_stamped.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2/LinearMath/Quaternion.h"
// #include "tf2_ros/buffer.h"
#include "tf2_ros/transform_broadcaster.h"
// #include "tf2_ros/transform_listener.h"
#include "turtlesim/msg/pose.hpp"
#include <chrono>

using namespace std::chrono_literals;

class FrameListener : public rclcpp::Node
{
public:
    explicit FrameListener()
        : Node("turtle_tf2_listener")
    {
        _tf_bc = std::make_shared<tf2_ros::TransformBroadcaster>(this);
        // _tf_buffer = std::make_unique<tf2_ros::Buffer>(get_clock());
        // _tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*_tf_buffer);
        _twist_pub = create_publisher<geometry_msgs::msg::Twist>("turtle2/cmd_vel", 10);
        _on_timer = create_wall_timer(1s,
                                      std::bind(&FrameListener::on_timer, this));
    }

private:
    void on_timer()
    {
        geometry_msgs::msg::TransformStamped t;
        // t = _tf_buffer->lookupTransform("turtle2", "turtle1", tf2::TimePointZero);
        geometry_msgs::msg::Twist msg;
        msg.angular.z = 1.0 * atan2(t.transform.translation.y, t.transform.translation.x);
        msg.linear.x = 0.5 * sqrt(pow(t.transform.translation.x, 2) + pow(t.transform.translation.y, 2));
        _twist_pub->publish(msg);
    }
    std::shared_ptr<tf2_ros::TransformBroadcaster> _tf_bc;
    rclcpp::Subscription<turtlesim::msg::Pose>::SharedPtr _sub;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr _twist_pub;
    rclcpp::TimerBase::SharedPtr _on_timer;
    // std::unique_ptr<tf2_ros::Buffer> _tf_buffer;

    // std::shared_ptr<tf2_ros::TransformListener> _tf_listener_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<FrameListener>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}