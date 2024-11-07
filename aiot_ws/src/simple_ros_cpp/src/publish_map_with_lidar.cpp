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

        _msg.data[0 + _count + (_msg.info.width * _row)] = 100;
        _count++;
        if (_count >= _msg.info.width)
        {
            _count = 0;
            _row++;
        }
        if (_row >= _msg.info.height)
        {
            _row = 0;
            value++;
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
