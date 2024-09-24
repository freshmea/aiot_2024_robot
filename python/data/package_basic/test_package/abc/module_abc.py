
# from python.data.package_basic.test_package import module_a


from .. import module_a

variable_abc = "abc variable"

def abc_function():
    print(module_a.variable_a)
    print(variable_abc)
