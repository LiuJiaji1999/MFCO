import os
import sys

print("当前解释器路径:", sys.executable)
print("环境变量PYTHONPATH:", os.environ.get("PYTHONPATH"))
print("模块搜索路径:")
for p in sys.path:
    print("  ", p)

try:
    import einops
    print("einops 路径:", einops.__file__)
except ImportError:
    print("没有安装 einops")


# import tkinter as tk
# tk.Tk().mainloop()