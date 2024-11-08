#include "nav_msgs/msg/occupancy_grid.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include <chrono>

using namespace std;
using namespace std::chrono_literals;

class PublishMap : public rclcpp::Node
{
public:
    explicit PublishMap()
        : Node("publish_map_with_lidar"), _count(0), _row(0)
    {
        _pub = create_publisher<nav_msgs::msg::OccupancyGrid>("map", 10);
        _odom_sub = create_subscription<nav_msgs::msg::Odometry>(
            "odom",
            10,
            [this](const nav_msgs::msg::Odometry::SharedPtr msg)
            {
                _odom = *msg;
            });
        // sensor qos
        rclcpp::QoS rmw_qos_profile_sensor_data = rclcpp::SensorDataQoS();
        _laser_sub = create_subscription<sensor_msgs::msg::LaserScan>(
            "scan",
            rmw_qos_profile_sensor_data,
            [this](const sensor_msgs::msg::LaserScan::SharedPtr msg)
            {
                _laser = *msg;
            });
        _timer = create_wall_timer(1ms, std::bind(&PublishMap::pub_callback, this));
        // map info
        _msg.info.resolution = 0.1f;
        _msg.info.width = 100;
        _msg.info.height = 100;
        _msg.info.origin.position.x = -(_msg.info.width * _msg.info.resolution) / 2;
        _msg.info.origin.position.y = -(_msg.info.height * _msg.info.resolution) / 2;
        _msg.info.origin.position.z = 0;
        _msg.info.origin.orientation.x = 0;
        _msg.info.origin.orientation.y = 0;
        _msg.info.origin.orientation.z = 0;
        _msg.info.origin.orientation.w = 1;

        _msg.data.resize(_msg.info.width * _msg.info.height);
        for (auto &i : _msg.data)
        {
            i = -1;
        }
    }

private:
    int _count;
    int _row;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr _pub;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr _odom_sub;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr _laser_sub;
    rclcpp::TimerBase::SharedPtr _timer;
    nav_msgs::msg::OccupancyGrid _msg = nav_msgs::msg::OccupancyGrid();
    nav_msgs::msg::Odometry _odom = nav_msgs::msg::Odometry();
    sensor_msgs::msg::LaserScan _laser = sensor_msgs::msg::LaserScan();
    void pub_callback()
    {
        static int value = 0;
        _msg.header.frame_id = "odom";
        _msg.header.stamp = get_clock()->now();

        // turtlebot pose info
        auto turtle_x = _odom.pose.pose.position.x;
        auto turtle_y = _odom.pose.pose.position.y;
        auto turtle_yaw = atan2(
            2 * (_odom.pose.pose.orientation.w * _odom.pose.pose.orientation.z +
                 _odom.pose.pose.orientation.x * _odom.pose.pose.orientation.y),
            1 - 2 * (_odom.pose.pose.orientation.y * _odom.pose.pose.orientation.y +
                     _odom.pose.pose.orientation.z * _odom.pose.pose.orientation.z));

        // laser scan info
        int iter = 0;
        for (auto &scan_radian : _laser.ranges)
        {
            if (scan_radian == INFINITY || scan_radian == 0.0)
            {
                scan_radian = _laser.range_max;
            }
            auto scan_theta = _laser.angle_min + _laser.angle_increment * iter;
            auto scan_x = turtle_x + scan_radian * cos(turtle_yaw + scan_theta);
            auto scan_y = turtle_y + scan_radian * sin(turtle_yaw + scan_theta);
            iter++;
            // center of the map
            // auto center = _msg.info.origin.position.x + _msg.info.origin.position.y * _msg.info.width;
            auto center = _msg.info.width / 2 + _msg.info.height / 2 * _msg.info.width;
            u_int32_t index = center + int(scan_x * 10) + int(scan_y * 10) * _msg.info.width;
            if (index >= 0 && index < _msg.data.size())
            {
                _msg.data[index] = 100;
            }
            else
            {
                RCLCPP_INFO(get_logger(), "index out of range");
            }
        }

        _pub->publish(_msg);
    }
};

int main()
{
    rclcpp::init(0, nullptr);
    auto node = std::make_shared<PublishMap>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
