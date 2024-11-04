#include "geometry_msgs/msg/transform_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_ros/transform_broadcaster.h"
#include "turtlesim/msg/pose.hpp"
#include <chrono>

using namespace std::chrono_literals;

class FramePublisher : public rclcpp::Node
{
public:
    explicit FramePublisher()
        : Node("static_turtle_tf2_broadcaster")
    {
        _tf_bc = std::make_shared<tf2_ros::TransformBroadcaster>(this);
        _sub = create_subscription<turtlesim::msg::Pose>(
            "turtle1/pose", 10,
            std::bind(&FramePublisher::sub_callback, this, std::placeholders::_1));
    }

private:
    void sub_callback(const std::shared_ptr<turtlesim::msg::Pose> msg)
    {
        geometry_msgs::msg::TransformStamped t;
        t.header.stamp = get_clock()->now();
        t.header.frame_id = "world";
        t.child_frame_id = "turtle1";
        t.transform.translation.x = msg->x;
        t.transform.translation.y = msg->y;
        t.transform.translation.z = 0.0;
        tf2::Quaternion q;
        q.setRPY(0, 0, msg->theta);
        t.transform.rotation.x = q.x();
        t.transform.rotation.y = q.y();
        t.transform.rotation.z = q.z();
        t.transform.rotation.w = q.w();
        _tf_bc->sendTransform(t);
    }
    std::shared_ptr<tf2_ros::TransformBroadcaster> _tf_bc;
    rclcpp::Subscription<turtlesim::msg::Pose>::SharedPtr _sub;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<FramePublisher>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}