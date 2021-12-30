variable_count = 1


# ## Module 1

# Variable is the main class for autodifferentiation logic for scalars
# and tensors.


class Variable:
    """
    Attributes:
        history (:class:`History` or None) : the Function calls that created this variable or None if constant
        derivative (variable type): the derivative with respect to this variable
        grad (variable type) : alias for derivative, used for tensors
        name (string) : a globally unique name of the variable
    """

    def __init__(self, history, name=None):
        global variable_count
        assert history is None or isinstance(history, History), history

        self.history = history
        self._derivative = None

        # This is a bit simplistic, but make things easier.
        variable_count += 1
        self.unique_id = "Variable" + str(variable_count)

        # For debugging can have a name.
        if name is not None:
            self.name = name
        else:
            self.name = self.unique_id
        self.used = 0

    def requires_grad_(self, val):
        """
        Set the requires_grad flag to `val` on variable.

        Ensures that operations on this variable will trigger
        backpropagation.

        Args:
            val (bool): whether to require grad
        """
        self.history = History()

    def backward(self, d_output=None):
        """
        Calls autodiff to fill in the derivatives for the history of this object.

        Args:
            d_output (number, opt): starting derivative to backpropagate through the model
                                   (typically left out, and assumed to be 1.0).
        """
        if d_output is None:
            d_output = 1.0
        backpropagate(self, d_output)

    @property
    def derivative(self):
        return self._derivative

    def is_leaf(self):
        "True if this variable created by the user (no `last_fn`)"
        return self.history.last_fn is None

    def accumulate_derivative(self, val):
        """
        Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
            val (number): value to be accumulated
        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self._derivative is None:
            self._derivative = self.zeros()
        self._derivative += val

    def zero_derivative_(self):  # pragma: no cover
        """
        Reset the derivative on this variable.
        """
        self._derivative = self.zeros()

    def zero_grad_(self):  # pragma: no cover
        """
        Reset the derivative on this variable.
        """
        self.zero_derivative_()

    def expand(self, x):
        "Placeholder for tensor variables"
        return x

    # Helper functions for children classes.

    def __radd__(self, b):
        return self + b

    def __rmul__(self, b):
        return self * b

    def zeros(self):
        return 0.0


# Some helper functions for handling optional tuples.


def wrap_tuple(x):
    "Turn a possible value into a tuple"
    if isinstance(x, tuple):
        return x
    return (x,)


def unwrap_tuple(x):
    "Turn a singleton tuple into a value"
    if len(x) == 1:
        return x[0]
    return x


# Classes for Functions.


class Context:
    """
    Context class is used by `Function` to store information during the forward pass.

    Attributes:
        no_grad (bool) : do not save gradient information
        saved_values (tuple) : tuple of values saved for backward pass
        saved_tensors (tuple) : alias for saved_values
    """

    def __init__(self, no_grad=False):
        self._saved_values = None
        self.no_grad = no_grad

    def save_for_backward(self, *values):
        """
        Store the given `values` if they need to be used during backpropagation.

        Args:
            values (list of values) : values to save for backward
        """
        if self.no_grad:
            return
        self._saved_values = values

    @property
    def saved_values(self):
        assert not self.no_grad, "Doesn't require grad"
        assert self._saved_values is not None, "Did you forget to save values?"
        return unwrap_tuple(self._saved_values)

    @property
    def saved_tensors(self):  # pragma: no cover
        return self.saved_values


class History:
    """
    `History` stores the history of `Function` operations that was
    used to construct the current Variable.

    Attributes:
        last_fn (:class:`FunctionBase`) : The last Function that was called.
        ctx (:class:`Context`): The context for that Function.
        inputs (list of inputs) : The inputs that were given when `last_fn.forward` was called.

    """

    def __init__(self, last_fn=None, ctx=None, inputs=None):
        self.last_fn = last_fn
        self.ctx = ctx
        self.inputs = inputs

    def backprop_step(self, d_output):
        """
        Run one step of backpropagation by calling chain rule.

        Args:
            d_output : a derivative with respect to this variable

        Returns:
            list of numbers : a derivative with respect to `inputs`
        """
        # TODO: Implement for Task 1.4.
        # raise NotImplementedError('Need to implement for Task 1.4')
        if not self.last_fn:
            return []
        return self.last_fn.chain_rule(self.ctx, self.inputs, d_output)


class FunctionBase:
    """
    A function that can act on :class:`Variable` arguments to
    produce a :class:`Variable` output, while tracking the internal history.

    Call by :func:`FunctionBase.apply`.

    """

    @staticmethod
    def variable(raw, history):
        # Implement by children class.
        raise NotImplementedError()

    @classmethod
    def apply(cls, *vals):
        """
        Apply is called by the user to run the Function.
        Internally it does three things:

        a) Creates a Context for the function call.
        b) Calls forward to run the function.
        c) Attaches the Context to the History of the new variable.

        There is a bit of internal complexity in our implementation
        to handle both scalars and tensors.

        Args:
            vals (list of Variables or constants) : The arguments to forward

        Returns:
            `Variable` : The new variable produced

        """
        # Go through the variables to see if any needs grad.
        raw_vals = []
        need_grad = False
        for v in vals:
            if isinstance(v, Variable):
                if v.history is not None:
                    need_grad = True
                v.used += 1
                raw_vals.append(v.get_data())
            else:
                raw_vals.append(cls.data_type(v)) # v为python原生类型时，直接计算结果可能类型不匹配

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls.forward(ctx, *raw_vals)
        assert isinstance(c, cls.data_type), "Expected return typ %s got %s" % (
            cls.data_type,
            type(c),
        )

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = History(cls, ctx, vals)
        return cls.variable(cls.data(c), back)

    @classmethod
    def chain_rule(cls, ctx, inputs, d_output):
        """
        Implement the derivative chain-rule.

        Args:
            ctx (:class:`Context`) : The context from running forward
            inputs (list of args) : The args that were passed to :func:`FunctionBase.apply` (e.g. :math:`x, y`)
            d_output (number) : The `d_output` value in the chain rule.

        Returns:
            list of (`Variable`, number) : A list of non-constant variables with their derivatives
            (see `is_constant` to remove unneeded variables)

        """
        # Tip: Note when implementing this function that
        # cls.backward may return either a value or a tuple.
        # TODO: Implement for Task 1.3.
        # raise NotImplementedError('Need to implement for Task 1.3')
        # ctx 只保留反向传播需要用到的参数
        back = cls.backward(ctx, d_output)
        back = wrap_tuple(back)
        
        return [(input, b) for input, b in zip(inputs, back) if not is_constant(input)]



# Algorithms for backpropagation


def is_constant(val):
    return not isinstance(val, Variable) or val.history is None


def topological_sort(variable):
    """
    Computes the topological order of the computation graph.

    Args:
        variable (:class:`Variable`): The right-most variable

    Returns:
        list of Variables : Non-constant Variables in topological order
                            starting from the right.
    """
    # TODO: Implement for Task 1.4.
    # raise NotImplementedError('Need to implement for Task 1.4')
    # 使用DFS 拓扑排序, 可利用的参数
    # variable ==> (value, history), 
    # history ==> (last_fn=None, ctx=None, inputs)
    # 要使用dfs，需要先定义好什么是节点
    # 把Variable和产生此Variable的Function合在一起当作图节点

    topo_order = []
    
    """
    def DFS(variable:Variable):
        if is_constant(variable):
            return
        if variable.used==0:
            topo_order.append(variable)
            if variable.history and variable.history.inputs:
                for input in variable.history.inputs:
                    if isinstance(input, Variable):
                        input.used -= 1
                        DFS(input)
    # 修改used参数没问题吗？
    # 需要检查是否是无环图吗？
    DFS(variable)
    
    """
    
    seen = set() 
    def DFS(variable:Variable):
        if variable.unique_id in seen:
            return
        if is_constant(variable):
            return
        if not variable.is_leaf(): 
            for v in variable.history.inputs:
                if not is_constant(v):
                    DFS(v)
        seen.add(variable.unique_id)
        topo_order.insert(0, variable)
        
    DFS(variable)
    
    return topo_order

def backpropagate(variable, deriv):
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    See :doc:`backpropagate` for details on the algorithm.

    Args:
        variable (:class:`Variable`): The right-most variable
        deriv (number) : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    # raise NotImplementedError('Need to implement for Task 1.4')
    # chain_rule 需要 (cls, ctx, inputs, d_output) 
    # variable ==> (value, history), 
    # history ==> (last_fn=None, ctx=None, inputs)
    topo_order = topological_sort(variable)
    var_2_driv = {
        variable.unique_id: deriv
    }

    for var in topo_order:
        if var.is_leaf():
            var.accumulate_derivative(var_2_driv[var.unique_id])
        else:
            back = var.history.backprop_step(var_2_driv[var.unique_id])
            # back [ (variable, deriv), ...]
            for b in back:
                var_b = b[0]
                deriv_b = b[1]
                if var_b.unique_id in var_2_driv:
                    var_2_driv[var_b.unique_id] +=deriv_b
                else:
                    var_2_driv[var_b.unique_id] = deriv_b

