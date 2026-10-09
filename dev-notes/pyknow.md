# Python 知识点笔记

读 NEXA 源码时遇到的 Python 知识点，一个知识点一段核心解释 + 一个简短示例。

## @dataclass

一个装饰器，加在"主要用来存数据"的类上，自动生成 `__init__`、`__repr__`、`__eq__` 等样板代码，省去手写重复方法。

```python
@dataclass(frozen=True, slots=True)  # frozen: 创建后不可修改; slots: 更省内存
class AgentTool:
    name: str
    description: str
```

## @dataclass 和 Pydantic 的区别

- **dataclass** 只负责少写代码，**不校验类型**——传错类型照样收下，注解只是给人看的。
- **Pydantic (BaseModel)** 构造时**真正校验和转换数据**，还能序列化成 JSON、定义校验器。

经验法则：内部自己用的数据容器用 dataclass；要发给外部（网络/JSON）的数据用 Pydantic。

```python
# dataclass：AgentTool 只在 Python 内部使用
# Pydantic：UserMessage 要序列化成 JSON 发给模型 API
class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: str | list[TextContent]
```

## __all__

一个字符串列表，声明"本模块对外提供的名字"。当别人写 `from xxx import *` 时，只导入 `__all__` 里列出的名字；同时也给读代码的人指明公开 API 的范围。

```python
__all__ = ["OpenAICompatibleProvider"]  # 这个文件对外只有这一个类
```

## AsyncIterator[AgentEvent] — 异步事件流

表示"一个异步的流，里面会陆续流出一串 AgentEvent"。函数里用 `yield` 产出值，调用方用 `async for` 消费；好处是每次取值之间可以 `await` 干别的事（如等网络响应），不阻塞。

```python
async def run(...) -> AsyncIterator[AgentEvent]:
    yield AgentStartEvent()

# 调用方：事件到达一个处理一个，不用等全部结束
async for event in harness.prompt("你好"):
    print(event)
```

## yield

普通函数用 `return` 一次返回就结束；函数里用了 `yield` 就变成**生成器**——每次产出一个值后暂停，调用方要下一个值时才继续执行，适合边运行边吐结果。

```python
def counter():
    yield 1  # 产出 1，暂停
    yield 2  # 要下一个时，从这里继续


for n in counter():  # 依次拿到 1, 2
    print(n)
```

## yield 事件流的意义

Agent 一次运行可能要几十秒（等模型 + 跑工具），用 `yield` 把每一步（开始、执行工具、结束……）立刻推给调用方，CLI/TUI 就能**边运行边显示进度**，而不是让用户盯着黑屏干等。

```python
yield AgentStartEvent()  # agent 开始
yield ToolExecutionStartEvent()  # 开始执行工具 → 界面立刻显示"正在执行 bash..."
yield AgentEndEvent(...)  # 全部结束
```

## @property

把方法伪装成属性：定义时是方法，使用时**不带括号**。适合由其他字段计算得出、希望像字段一样自然读取的值；property 应快速返回，不放耗时操作。

```python
class UserMessage(WireModel):
    @property
    def text(self) -> str:  # 由 content 计算得出
        ...


message.text  # ✓ 像读一个普通字段
message.text()  # ✗ 报错，text 不是方法调用
```

## type X = Annotated[...]（Python 3.12+）

类型别名的新写法：给一个联合类型起名字，等价于传统写法 `X = ...`，只是更清爽。`Annotated` 可以附加额外信息，`Field(discriminator="role")` 告诉 Pydantic 按 `role` 字段的值判断具体是哪种消息类。

```python
type AgentMessage = Annotated[
    UserMessage | AssistantMessage | ToolResultMessage,
    Field(discriminator="role"),  # role="user" → 解析成 UserMessage
]
```

## __init__.py

把文件夹变成 Python 包的文件，在包被导入时最先执行。主要作用：
1. 声明"这是个包"，可以 `from 包 import ...`。
2. 定义对外接口：把子模块的公开 API 汇总到一个入口，使用者只需记一个路径。

```python
# session/__init__.py 汇总导出，让下面两种写法等价：
from nexa_agent.session import JsonlStorage, SessionState
from nexa_agent.session.storage import JsonlStorage  # 等价，但要记子模块路径
```

注意：`__init__.py` 在包导入时最先执行，别放复杂逻辑，否则容易循环导入。

## TypeAdapter：PEP 695 别名的 Pydantic 序列化

`type X = Annotated[...]` 是类型别名（`TypeAliasType`），不能直接对它调 `model_validate` / `model_dump_json`。用 `TypeAdapter(X)` 包装成可校验、可序列化的适配器即可。

```python
type Entry = Annotated[MessageEntry | ModelChangeEntry | LabelEntry, Field(discriminator="type")]

_ADAPTER: TypeAdapter[Entry] = TypeAdapter(Entry)  # 模块级单例
_ADAPTER.dump_json(entry, exclude_none=True)  # 序列化
_ADAPTER.validate_python(data)  # 反序列化
```


## @classmethod — 类方法

绑定在类上的方法，第一个参数 `cls` 代表调用它的类；实例方法的 `self` 则代表具体对象。常用于提供另一种创建对象的方式：先处理数据，再用 `cls(...)` 创建对象，自动调用 `__init__()`。

```python
class Person:
    def __init__(self, name):
        self.name = name

    def introduce(self):  # 实例方法
        return f"我是 {self.name}"

    @classmethod
    def from_text(cls, text):  # 类方法，无需先创建实例
        return cls(text.strip())


person = Person.from_text("  小明  ")  # cls 是 Person
print(person.introduce())  # 我是 小明
```

`CodingSession.load(config)` 也是这个模式：准备好历史、工具等材料，再通过 `cls(...)` 创建会话。使用 `cls` 而不写死类名，还能让子类调用时创建子类对象。

## if isinstance

`isinstance(对象, 类)` 判断对象是否是这个类或其子类的实例，返回 `True` / `False`。配合 `if`，可以按对象类型执行不同逻辑。

```python
if isinstance(event, MessageEndEvent):
    message = event.message  # 只有消息结束事件才提取完整消息
```

## 
