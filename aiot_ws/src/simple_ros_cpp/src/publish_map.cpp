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
        _timer = create_wall_timer(100ms, std::bind(&PublishMap::pub_callback, this));
    }

private:
    int _count;
    int _row;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr _pub;
    rclcpp::TimerBase::SharedPtr _timer;
    void pub_callback()
    {
        static int value = 0;
        auto msg = nav_msgs::msg::OccupancyGrid();
        msg.header.frame_id = "odom";
        msg.header.stamp = get_clock()->now();

        // map info
        msg.info.resolution = 0.1f;
        msg.info.width = 100;
        msg.info.height = 100;
        msg.info.origin.position.x = -(msg.info.width * msg.info.resolution) / 2;
        msg.info.origin.position.y = -(msg.info.height * msg.info.resolution) / 2;
        msg.info.origin.position.z = 0;
        msg.info.origin.orientation.x = 0;
        msg.info.origin.orientation.y = 0;
        msg.info.origin.orientation.z = 0;
        msg.info.origin.orientation.w = 1;

        msg.data.resize(msg.info.width * msg.info.height);
        for (auto &i : msg.data)
        {
            i = -1;
        }

        msg.data[0 + _count + (msg.info.width * _row)] = 0;
        _count++;
        if (_count > msg.info.width)
        {
            _count = value;
            _row++;
        }
        if (_row > msg.info.height)
        {
            _row = 0;
            value++;
        }
        _pub->publish(msg);
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
