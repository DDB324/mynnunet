import importlib
import os
import pkgutil


def recursive_find_python_class(folder: str, class_name: str, current_module: str):
    tr = None
    for impoter, modname, ispkg in pkgutil.iter_modules([folder]):
        # print(modname, ispkg)
        if not ispkg:
            m = importlib.import_module(current_module + "." + modname)
            if hasattr(m, class_name):
                tr = getattr(m, class_name)
                break

    if tr is None:
        for impoter, modname, ispkg in pkgutil.iter_modules([folder]):
            if ispkg:
                next_current_module = current_module + "." + modname
                tr = recursive_find_python_class(folder=os.path.join(folder, modname),
                                                 class_name=class_name,
                                                 current_module=next_current_module)
            if tr is not None:
                break

    return tr
