from functools import partial

class FuncArgDict:

    def __init__(self, func, *hyper_args):
        self.__f = func
        self.__hyper_args = hyper_args


    def __hyper_arg_extend(self, args):
        new_args = []
        for arg in args:
            new_args.append(arg)
        for hyper_arg in self.__hyper_args:
            new_args.append(hyper_arg)
        return new_args

    def call(self, *args):
        all_args = self.__hyper_arg_extend(args)
        out = partial(self.__f, all_args[0])
        for i in range(1, len(all_args)-1):
            out = partial(out, all_args[i])
        return out(all_args[len(all_args)-1])


if __name__ == "__main__":
    def f(a, x, y, z):
        return a*(x**2 + y**2)

    func_arg_dict = FuncArgDict(f, 1, 2, 3)
    print("call(2): ", func_arg_dict.call(2))
