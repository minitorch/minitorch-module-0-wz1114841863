from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple


class Module:
    """ Modules form a tree that store parameters and other
    submodules. They make up the basis of neural network stacks.

    Attributes
        _modules : Storage of the child modules
        _parameters : Storage of the module's parameters
        training : Whether the module is in training mode or evaluation mode
    """

    _modules: Dict[str, Module]
    _parameters: Dict[str, Parameter]
    training: bool

    def __init__(self) -> None:
        self._modules = {}
        self._parameters = {}
        self.training = True

    def modules(self) -> Sequence[Module]:
        """ Return the direct child modules of this module. """
        # 直接使用 self.__dict__["_modules"] 而不是 self._modules 的原因是为了
        # 更明确/直接地访问底层存储, 避免触发额外的逻辑或属性重载问题
        m: Dict[str, Module] = self.__dict__["_modules"]
        return list(m.values())

    def _set_mode_recursive(self, mode: bool) -> None:
        """  Recursively set the mode (training or evaluation) for this module and its descendants.

        Args:
            mode (bool): If True, set training mode. If False, set evaluation mode.
        """
        self.training = mode
        for child in self.modules():
            child._set_mode_recursive(mode=mode)

    def train(self) -> None:
        """ Set the mode of this module and all descendent modules to `train`. """
        self._set_mode_recursive(True)

    def eval(self) -> None:
        """ Set the mode of this module and all descendent modules to `eval`. """
        self._set_mode_recursive(False)

    def named_parameters(self) -> Sequence[Tuple[str, Parameter]]:
        """ Collect all the parameters of this module and its descendents.

        Returns:
            The name and `Parameter` of each ancestor parameter.
        """
        def _named_parameters(module, prefix=""):
            for name, param in module._parameters.items():
                yield f"{prefix}{name}", param   # 每次返回一个元组 (name, param)
            for child_name, child_module in module._modules.items():
                # yield from 是 Python 用于生成器的语法糖, 专为处理委托生成器的场景而设计.
                # 它能够直接传递值/支持双向通信/捕获返回值,并简化代码逻辑
                yield from _named_parameters(child_module, f"{prefix}{child_name}.")

        return list(_named_parameters(self))

    def parameters(self) -> Sequence[Parameter]:
        """ Enumerate over all the parameters of this module and its descendents. """
        return [param for __, param in self.named_parameters()]

    def add_parameter(self, k: str, v: Any) -> Parameter:
        """ Manually add a parameter. Useful helper for scalar parameters.

        Args:
            k: Local name of the parameter.
            v: Value for the parameter.

        Returns:
            Newly created parameter.
        """
        val = Parameter(v, k)
        self.__dict__["_parameters"][k] = val
        return val

    def add_module(self, name, module):
        """ Manually add a child_module. """
        self._modules[name] = module

    def __setattr__(self, key: str, val: Parameter) -> None:
        if isinstance(val, Parameter):
            self.__dict__["_parameters"][key] = val
        elif isinstance(val, Module):
            self.__dict__["_modules"][key] = val
        else:
            super().__setattr__(key, val)

    def __getattr__(self, key: str) -> Any:
        if key in self.__dict__["_parameters"]:
            return self.__dict__["_parameters"][key]

        if key in self.__dict__["_modules"]:
            return self.__dict__["_modules"][key]
        return None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def __repr__(self) -> str:
        def _addindent(s_: str, numSpaces: int) -> str:
            s2 = s_.split("\n")
            if len(s2) == 1:
                return s_
            first = s2.pop(0)
            s2 = [(numSpaces * " ") + line for line in s2]
            s = "\n".join(s2)
            s = first + "\n" + s
            return s

        child_lines = []

        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = child_lines

        main_str = self.__class__.__name__ + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str


class Parameter:
    """ A Parameter is a special container stored in a `Module`.

    It is designed to hold a `Variable`, but we allow it to hold
    any value for testing.
    """

    def __init__(self, x: Any, name: Optional[str] = None) -> None:
        self.value = x
        self.name = name
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

    def update(self, x: Any) -> None:
        """ Update the parameter value. """
        self.value = x
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

    def __repr__(self) -> str:
        return repr(self.value)

    def __str__(self) -> str:
        return str(self.value)


if __name__ == "__main__":
    # 创建模块
    root = Module()
    child1 = Module()
    child2 = Module()

    root.add_module("child1", child1)
    root.add_module("child2", child2)

    # 打印结构
    print(repr(root))
