#include "geometry_msgs/msg/transform_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_ros/static_transform_broadcaster.h"
#include <chrono>

using namespace std::chrono_literals;

class StaticFramePublisher : public rclcpp::Node
{
public:
    explicit StaticFramePublisher(char *transformation[])
        : Node("static_turtle_tf2_broadcaster")
    {
        _tf_static_bc = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);
        make_transforms(transformation);
    }

private:
    void make_transforms(char *transformation[])
    {
        geometry_msgs::msg::TransformStamped t;
        t.header.stamp = get_clock()->now();
        t.header.frame_id = "odom";
        t.child_frame_id = transformation[1];

        t.transform.translation.x = atof(transformation[2]);
        t.transform.translation.y = atof(transformation[3]);
        t.transform.translation.z = atof(transformation[4]);
        tf2::Quaternion q;
        q.setRPY(
            atof(transformation[5]),
            atof(transformation[6]),
            atof(transformation[7]));
        t.transform.rotation.x = q.x();
        t.transform.rotation.y = q.y();
        t.transform.rotation.z = q.z();
        t.transform.rotation.w = q.w();

        _tf_static_bc->sendTransform(t);
    }
    std::shared_ptr<tf2_ros::StaticTransformBroadcaster>
        _tf_static_bc;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<StaticFramePublisher>(argv);
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}