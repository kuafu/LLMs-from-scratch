# 优化预训练超参数

基于[附录 D：为训练循环增添实用功能](../../appendix-D/01_main-chapter-code/appendix-D.ipynb)中的扩展训练函数，[hparam_search.py](hparam_search.py) 脚本通过网格搜索来寻找最优超参数。

>[!NOTE]
此脚本运行时间较长。你可以考虑减少顶部 `HPARAM_GRID` 字典中要探索的超参数配置数量。
