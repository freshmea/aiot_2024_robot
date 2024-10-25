#include "rclcpp/rclcpp.hpp"
#include "rcutils/cmdline_parser.h"
#include "user_interface/msg/arithmetic_argument.hpp"
#include <cstdio>
#include <utility>

void print_help()
{
    printf("for argument node: \n");
    printf("node name [-h]\n");
    printf("Option:\n");
    printf("add some explanation!!!\n");
}

int main(int argc, char *argv[])
{
    if (rcutils_cli_option_exist(argv, argv + argc, "-h"))
    {
        print_help();
        return 0;
    }
}