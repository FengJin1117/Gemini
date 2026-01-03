# Gemini 评估genre
## 脚本
run_eval.py: 评测入口
- 指定gemini版本
- 选择执行不同的任务

prompt.py: prompt在这里写。

task.py: 任务

evaluate.py: 这里是负责文件夹遍历和结果jsonl输出逻辑
- 这里存在检测逻辑。当jsonl中发现已经评测过，就不会重复评测。
- 支持断点存续。

gemini_cilent.py: 统一前端，和具体task无关。负责传递参数，控制gemini。

openai_client.py: openai形式的前端。
- 设计思路：我们认为openai和genai属于相互独立的两套逻辑，因此不在协议层强行融合。


prompt_loader: 根据genre，加载对应的extra_genre_prompt
- 配置文件：genre_extra_prompts.json

## 调用顺序
顶层入口：run_eval.py
初始化cilent
evaluate.py => task.py =>  

## TODO
py去除api key！！防止泄露！
- 配置文件或命令

清扫旧有的多余文件，让项目看起来清爽

## prompt设计最好用英文
因为和audio相关的数据，大多用的是英文text，用英文更更能激活知识神经元
用中文可能导致输出不了简洁的markdonw（干扰）

## gemini 调用
gemini 2.5 flash / lite: 一天20次。

设置环境变量
Windows PowerShell
$Env:GEMINI_API_KEY="YOUR_API_KEY"

