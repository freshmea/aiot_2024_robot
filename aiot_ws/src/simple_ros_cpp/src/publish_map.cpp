#include "nav_msgs/msg/occupancy_grid.hpp"
#include "rclcpp/rclcpp.hpp"
#include <chrono>

using namespace std;
using namespace std::chrono_literals;

class PublishMap : public rclcpp::Node
{
public:
    explicit PublishMap()
        : Node("publish_map"), _count(0), _row(0)
    {
        _pub = create_publisher<nav_msgs::msg::OccupancyGrid>("map", 10);
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
    nav_msgs::msg::OccupancyGrid _msg = nav_msgs::msg::OccupancyGrid();
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr _pub;
    rclcpp::TimerBase::SharedPtr _timer;
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

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PublishMap>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
