# Gemini 评估genre
## 脚本
run_eval.py: 评测入口
- 指定gemini版本
- 选择执行不同的任务

prompt.py: prompt在这里写。

task.py: 任务

evaluate.py: 这里是负责文件夹遍历和结果jsonl输出逻辑

gemini_cilent.py: 统一前端，和具体task无关。负责传递参数，控制gemini。

## TODO
py去除api key！！防止泄露！
- 配置文件或命令

清扫旧有的多余文件，让项目看起来清爽
