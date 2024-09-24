# import test_package.module_a as a
# import test_package.module_b as b
import test_package.abc.module_abc as abc
from test_package import *


def main():
    # print(a.variable_a)
    # print(b.variable_b)
    print(module_a.variable_a)
    print(module_b.variable_b)
    print(abc.variable_abc)
    abc.abc_function()

if __name__ == "__main__":
    main()
