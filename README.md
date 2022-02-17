# MiniTorch Module 1

<img src="https://minitorch.github.io/_images/match.png" width="100px">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module1.html

This assignment requires the following files from the previous assignments.

        minitorch/operators.py minitorch/module.py tests/test_module.py tests/test_operators.py project/run_manual.py



# forward和backward流程
例如 test task1_4：
```python
var = minitorch.Scalar(0)
var2 = Function1.apply(0, var)
var2.backward(d_output=5)
assert var.derivative == 5
```

`var = minitorch.Scalar(0)`
1. 初始化空的history ,
2. 梯度设置成None, 
3. 初始化name或者和var id
4. 初始化value ：0

`var2 = Function1.apply(0, var)`
1. 如果计算路径上有Variable, 那么，该Functioon需要 grad, 同时输入Variable的used参数 +1
2. 生成一个 Context， 传入上一步的grad参数
   1. 初始化是否更新梯度
   2. 初始化_saved_values=None
3. forward, 这个函数由子类实现， 传入Content 和raw_vals
   1. Content 保存传入的raw_values为saved_values，用于反向传播。如果初始化时Content不需要计算梯度，这此处不保存
   2. raw_values 计算，返回结果
4. 将本次计算的Function、Content、vals 打包成History
5. 将3的结果和4的History打包成新的Variable

`var2.backward(d_output=5)`
1. 反向传播需要知道传递到var2 的梯度值，所以需要d_output
2. 拓扑排序，确定正确的计算顺序
3. 对于拓扑排序中的每个节点分别反向传播 step级别，即看这个History保存的Function step
   1. Function 计算 backward ， backward list结果，[(inputs:Varable, deriv),...]
4. 不同反向传播路径上Variable的梯度相加

