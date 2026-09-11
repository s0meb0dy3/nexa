my-coding-agent/
├── dev-notes/                      # 开发笔记
│   ├── pyknow.md                   #   Python 知识点笔记（dataclass/Pydantic、yield、TypeAdapter…）
│   ├── session.md                  #   会话层设计说明（各文件职责）
│   └── notes.md                    #   本地临时笔记（含密钥，已 gitignore）
│
├── src/nexa_agent/                 # ── 核心层：可移植的 agent 大脑（零外部层依赖）──
│   ├── messages.py                 #   消息模型（User/Assistant/ToolResult，Pydantic + role 判别）
│   ├── tools.py                    #   AgentTool / AgentToolResult 定义
│   ├── loop.py                     #   AgentLoop 主循环：模型→工具→再模型
│   ├── harness.py                  #   AgentHarness：持续对话封装（历史/订阅/取消）
│   ├── events.py                   #   AgentEvent 9 种事件（生命周期通知）
│   ├── provider.py                 #   ModelProvider 协议（核心拥有，实现层来实现它）
│   ├── provider_events.py          #   ProviderEvent 5 种事件（模型语言）
│   ├── types.py                    #   JSONValue 等基础类型别名
│   └── session/                    #   append-only 会话持久化层
│       ├── entries.py              #     账本条目模型（Message/ModelChange/Label/SessionInfo/Leaf）
│       ├── jsonl.py                #     条目 ↔ JSONL 行序列化（坏行容忍、带行号报错）
│       ├── storage.py              #     JsonlStorage：只追加、绝不改写
│       ├── tree.py                 #     按 parent_id 走回根（path_to_entry）
│       └── memory.py               #     回放：条目 → SessionState（messages/model/label）
│
├── src/nexa_ai/                    # ── 实现层：怎么和模型说话（依赖核心）──
│   ├── openai_compatible.py        #   OpenAI 兼容 Provider（DeepSeek 等，HTTP/SSE 流式）
│   └── fake.py                     #   FakeProvider（测试用，按脚本回放事件）
│
├── src/nexa_coding/                # ── 应用层：把大脑变成真应用（组合核心+实现）──
│   ├── tools.py                    #   四大编码工具：read / write / edit / bash
│   ├── config.py                   #   配置文件解析（~/.nexa/config.toml 供应商档案）
│   ├── cli.py                      #   入口：nexa -p（print）/ --tui / --provider / --model / --output
│   ├── session.py                  #   CodingSession：持久化编码会话（load 回放、prompt 落盘、/skill 展开）
│   ├── system_prompt.py            #   build_system_prompt：确定性纯函数拼系统提示词
│   ├── skills.py                   #   技能加载 / /skill:name展开 / 索引
│   ├── paths.py                    #   NexaPaths：所有用户数据位置的唯一真相源（含 config.toml）
│   ├── resources.py                #   资源路径四级发现（~/.nexa → ~/.agents → 项目）+ frontmatter 解析
│   ├── prompt_templates.py         #   提示模板（{{var}} 渲染）
│   ├── rendering/                  #   print 模式的事件渲染器
│   │   ├── base.py                 #     EventRenderer 协议（render/finish）
│   │   ├── plain.py                #     text：只出最终答案
│   │   ├── json.py                 #     json：每事件一行 JSON
│   │   └── transcript.py           #     transcript：过程→stderr，答案→stdout
│   ├── tui/                        #   交互式终端界面
│   │   ├── state.py                #     TuiState 纯显示状态（零 Textual 依赖）
│   │   ├── adapter.py              #     事件→状态翻译（零 Textual 依赖，可单测）
│   │   └── app.py                  #     NexaTuiApp：唯一碰 Textual 的渲染层（worker 后台跑）
│   └── demo.py                     #   教学演示脚本
│
├── 用户数据（不在仓库内）
│   ├── ~/.nexa/config.toml         #   供应商档案（base_url/model/api_key）；权限 600，含密钥
│   └── ~/.nexa/sessions/<slug>-<hash>/default.jsonl  #   按项目隔离的会话账本
│
└── tests/                          # 12 个测试文件，109 个测试
    ├── test_loop.py / test_harness.py / test_coding_tools.py / test_cli.py
    ├── test_session.py             #   会话层 5 个不变量
    ├── test_coding_session.py      #   CodingSession 持久化链路
    ├── test_config.py / test_paths.py
    ├── test_skills.py / test_system_prompt.py
    ├── test_rendering.py           #   三种渲染模式
    ├── test_tui_adapter.py         #   TUI 逻辑层（不开终端）
    └── test_events_demo.py         #   事件机制教学示例