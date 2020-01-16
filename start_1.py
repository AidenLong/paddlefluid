# -*- coding utf-8 -*- #

import paddle.fluid as fluid


# 检验gpu版是否安装成功
# fluid.install_check.run_check()

# 简单案列 x + y = z
def example_start():
    # 创建变量
    x = fluid.layers.fill_constant(shape=[1], dtype='int64', value=5)
    y = fluid.layers.fill_constant(shape=[1], dtype='int64', value=6)
    z = x + y

    # 创建一个运行在CPU上的执行器
    exe = fluid.Executor(fluid.CPUPlace())
    # 执行这个程序并获取到结果
    print(exe.run(fluid.default_main_program(), fetch_list=[z]))


# if-else 案例 学习率设置时可能会用到
def example_ie():
    # 创建两个Tensor变量
    a = fluid.layers.fill_constant(shape=[2, 1], dtype='int64', value=4)
    b = fluid.layers.fill_constant(shape=[2, 1], dtype='int64', value=6)
    # 创建一个if-else块
    ifcond = fluid.layers.less_than(x=a, y=b)
    ie = fluid.layers.IfElse(ifcond)
    with ie.true_block():
        c = ie.input(a)
        c += 1
        ie.output(c)
    exe = fluid.Executor(fluid.CPUPlace())
    print(exe.run(fluid.default_main_program(), fetch_list=[c]))


# while 案例
def example_while():
    a = fluid.layers.fill_constant(shape=[2, 2], dtype='float32', value=1.0)
    i = fluid.layers.zeros(shape=[1], dtype='int64')
    until = fluid.layers.fill_constant([1], 'int64', 10)
    data_arr = fluid.layers.array_write(a, i)
    # 创建一个while块
    cond = fluid.layers.less_than(i, until)
    while_op = fluid.layers.While(cond=cond)
    with while_op.block():
        a = fluid.layers.array_read(data_arr, i)
        a = a + 1
        i = fluid.layers.increment(x=i, value=1, in_place=True)
        fluid.layers.less_than(x=i, y=until, cond=cond)
        fluid.layers.array_write(a, i, data_arr)

    ret = fluid.layers.array_read(data_arr, i - 1)
    exe = fluid.Executor(fluid.CPUPlace())
    print(exe.run(fluid.default_main_program(), fetch_list=[ret]))


if __name__ == '__main__':
    # example_start()
    # example_ie()
    example_while()
