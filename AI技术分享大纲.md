# AI 技术分享大纲

## 一、基础概念速通（5 min）

> 目的：建立共同语言

| 概念               | 一句话解释                                                         |
|------------------|---------------------------------------------------------------|
| LLM（大语言模型）       | 能理解和生成文本的超大模型                                                 |
| Token            | 模型处理文本的最小单位                                                   |
| Prompt           | 你给模型的输入指令                                                     |
| Context Window   | 模型一次能"看到"多少内容                                                 |
| Embedding        | 把文字变成向量，用于语义搜索                                                |
| RAG（检索增强生成）      | 让模型"查资料再回答"，解决知识过时问题                                          |
| Fine-tuning      | 用特定数据让模型更专注某个领域                                               |
| Function Calling | 让模型调用外部工具，完成复杂任务                                              |
| Tools            | 外部工具，如计算器、搜索等                                                 |
| Memory           | 模型的记忆，用于保存历史对话内容                                              |
| MCP              | Model Control Protocol，模型控制协议                                 |
| Agent            | 智能体，能自主决策、调用工具、完成任务                                           |
| Agentic AI       | 代理式人工智能，能自主规划和调整目标                                            |
| a2a              | agent to agent，专注于不同供应商的智能体之间的任务分配和能力发现。                      |
| ACP(Google)      | Agent Client Protocol (智能体客户端协议)，解决 AI 智能体与代码编辑器 (IDE) 的集成问题。 |
| ACP(IBM)         | Agent Communication Protocol (智能体通信协议), 解决不同智能体间的结构化协作与消息传递。  |

---

## 一、定义

AI Agent
是一种能够自主感知环境、进行推理并采取行动，以完成特定目标的软件实体，它可以是简单的聊天机器人，也可以是复杂的自动化工具，但核心特征是在预定义边界内工作。你可以把它想象成一个拥有“大脑”（大模型）、“眼睛”（感知）、“手”（工具）和“记忆”（经验）的数字员工。

| 特性维度 | AI Agent (智能体) | Agentic AI (代理式人工智能) |
|:-----|:---------------|:---------------------|
| 核心定位 | 高效的执行者，专注于单一任务 | 智慧的统筹者，专注于复杂目标       |
| 自主性  | 在预设规则内自主执行     | 高度自主，能主动规划和调整目标      |
| 决策逻辑 | 反应式、基于规则       | 主动式、基于目标拆解与推理        |
| 系统构成 | 通常是单一、独立的实体    | 由多个 AI Agent 协同构成的系统 |
| 学习能力 | 有限，通常在特定任务内优化  | 持续学习，能跨任务优化整体策略      |

常见 AI Agent 应用场景：

- Chat Agent（聊天机器人）
- Code Agent（代码代理）
- Data Agent（数据代理）

---

## 三、AI Agent 技术栈（15 min）

1. 大模型（LLM）
2. Memory
3. 工具调用（Function Calling）
4. 工具（Tools）
5. MCP 协议

---

### 3.1 直接调用模式（直接调用大模型）

- 问题 → 模型回答
- 输入 Prompt → 拿输出 → 完事

#### 3.1.1 OpenAI API 调用

**请求参数说明：**

| 参数          | 类型      | 必填 | 说明                            |
|-------------|---------|----|-------------------------------|
| model       | string  | 是  | 模型名称，如 `gpt-4o`、`gpt-4o-mini` |
| messages    | array   | 是  | 对话消息数组                        |
| temperature | float   | 否  | 生成随机性，0-2 之间，越高越随机            |
| max_tokens  | integer | 否  | 最大生成 token 数                  |
| stream      | boolean | 否  | 是否使用流式输出                      |
| tools       | array   | 否  | 可用工具列表                        |
| tool_choice | string  | 否  | 强制使用特定工具                      |

**messages 格式：**

```json
[
  {
    "role": "system",
    "content": "你是一个专业的Python编程助手"
  },
  {
    "role": "user",
    "content": "请解释什么是装饰器"
  }
]
```

#### 3.1.2 国内厂商 API 适配

由于国内大多数厂商都兼容 OpenAI 格式，只需修改 `baseUrl` 即可无缝切换：

| 厂商          | Base URL                            | 备注           |
|-------------|-------------------------------------|--------------|
| DeepSeek    | `https://api.deepseek.com/v1`       | 性价比高，支持超长上下文 |
| Kimi (月之暗面) | `https://api.moonshot.cn/v1`        | 支持 128k 上下文  |
| 阿里云百炼       | `https://dashscope.aliyuncs.com/v1` | 通义千问系列       |
| 智谱 GLM      | `https://open.bigmodel.cn/v1`       | GLM-4 系列     |
| 小米 MiMo     | `https://api.mimomodel.com/v1`      | 小米自研         |

#### 代码示例

**Java 代码示例（使用 OkHttp）：**

```java
package llm.openai.chat;

import com.alibaba.dashscope.exception.ApiException;
import com.alibaba.dashscope.exception.InputRequiredException;
import com.alibaba.dashscope.exception.NoApiKeyException;
import com.google.common.collect.Lists;
import com.openai.models.chat.completions.ChatCompletion;
import com.openai.models.chat.completions.ChatCompletionCreateParams;
import com.openai.models.chat.completions.ChatCompletionMessageParam;
import com.openai.models.chat.completions.ChatCompletionSystemMessageParam;
import com.openai.models.chat.completions.ChatCompletionUserMessageParam;
import java.util.List;
import lombok.extern.slf4j.Slf4j;

@Slf4j
public class OpenAISimpleChat {


  public static ChatCompletion callWithMessage(
      List<ChatCompletionMessageParam> messages) throws ApiException, NoApiKeyException, InputRequiredException {

    ChatCompletionCreateParams params = ChatCompletionCreateParams.builder()
        .messages(messages)
        .model("dashscope/qwen3.5")
        .build();

    return Single.openAIClient.chat().completions().create(params);
  }

  /**
   * 调用示例 分析文本并提取变量名
   */
  public static void callParameterExtraction() throws ApiException, NoApiKeyException, InputRequiredException {

    ChatCompletionSystemMessageParam systemMessageParam = ChatCompletionSystemMessageParam.builder()
        .content("你是一个参数提取工具，可以从输入的文本中提取出要提问的变量/名，变量名可以包含英文数字和下划线")
        .build();

    ChatCompletionUserMessageParam userMessageParam = ChatCompletionUserMessageParam.builder()
        .content("分析 abc_dd_v2 的血缘关系")
        .build();

    List<ChatCompletionMessageParam> messages = Lists.newArrayList();
    messages.add(ChatCompletionMessageParam.ofSystem(systemMessageParam));
    messages.add(ChatCompletionMessageParam.ofUser(userMessageParam));
    ChatCompletion generationResult = callWithMessage(messages);

    log.info("Generation Result: {}", generationResult);
    log.info("Generation Result Content: {}", generationResult.choices().getFirst().message().content());

    // 第二次对话
    generationResult.choices().forEach(choice -> {
      messages.add(ChatCompletionMessageParam.ofAssistant(choice.message().toParam()));
    });

    userMessageParam = ChatCompletionUserMessageParam.builder()
        .content("分析 abc_hh2 的血缘关系")
        .build();

    messages.add(ChatCompletionMessageParam.ofUser(userMessageParam));
    generationResult = callWithMessage(messages);

    log.info("Generation Result: {}", generationResult);
    log.info("Generation Result Content: {}", generationResult.choices().getFirst().message().content());

  }

  public static void main(String[] args) throws NoApiKeyException, InputRequiredException {
    callParameterExtraction();
  }

}

```

**Python 代码示例：**

```python
from openai import OpenAI

client = OpenAI(api_key="your-api-key")

def chat(model: str, messages: list[dict]) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=1000
    )
    return response.choices[0].message.content

messages = [
    {"role": "system", "content": "你是一个专业的Python编程助手"},
    {"role": "user", "content": "请解释什么是装饰器"}
]

result = chat("gpt-4o", messages)
print(result)
```

#### 3.1.3 错误处理

**常见错误类型：**

| 错误码 | 说明            | 处理建议          |
|-----|---------------|---------------|
| 401 | API Key 无效或过期 | 检查并更新 API Key |
| 429 | 请求频率超限        | 添加重试延迟，使用指数退避 |
| 500 | 服务器内部错误       | 重试，通常是临时问题    |
| 503 | 服务不可用         | 降级到备用模型或厂商    |

### 3.2 Memory 模式（记忆）

**短期记忆**：模型在当前对话轮次内，记录用户输入和模型输出，用于后续决策

**长期记忆**：模型在多个对话轮次内，记录用户输入和模型输出，用于后续决策

#### 3.2.1 长期记忆处理方案

**方案一：向量数据库存储**

将对话历史 embedding 后存储到向量数据库，检索时通过语义相似度召回相关记忆。

```java
public class MemoryStore {

  private final VectorStore vectorStore;
  private final OpenAIClient llmClient;

  public void addToMemory(String userId, String content) throws IOException {
    String embedding = llmClient.embed(content);
    MemoryRecord record = new MemoryRecord(userId, content, embedding, Instant.now());
    vectorStore.save(record);
  }

  public List<String> recall(String userId, String query, int topK) throws IOException {
    String queryEmbedding = llmClient.embed(query);
    return vectorStore.search(userId, queryEmbedding, topK)
        .stream()
        .map(MemoryRecord::getContent)
        .collect(Collectors.toList());
  }
}
```

**Python 实现：**

```python
import chromadb
from openai import OpenAI

client = OpenAI()
vector_store = chromadb.Client()

collection = vector_store.create_collection("memory")

def add_memory(user_id: str, content: str):
    embedding = client.embeddings.create(
        input=content,
        model="text-embedding-3-small"
    ).data[0].embedding
    collection.add(
        embeddings=[embedding],
        documents=[content],
        ids=[f"{user_id}_{time.time()}"]
    )

def recall(user_id: str, query: str, top_k: int = 5) -> list[str]:
    query_embedding = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    ).data[0].embedding
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results["documents"][0] if results["documents"] else []
```

**方案二：摘要式记忆**

定期将对话历史压缩成摘要，减少 token 消耗同时保留关键信息。

```python
def summarize_conversation(conversation: list[dict]) -> str:
    summary_prompt = f"""请将以下对话内容总结为简洁的摘要，保留关键信息和用户偏好：

{chr(10).join([f"{msg['role']}: {msg['content']}" for msg in conversation])}

摘要："""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": summary_prompt}]
    )
    return response.choices[0].message.content

class SummarizedMemory:
    def __init__(self, max_turns: int = 10):
        self.conversations: dict[str, list[dict]] = {}
        self.summaries: dict[str, str] = {}
        self.max_turns = max_turns

    def add_turn(self, user_id: str, user_msg: str, assistant_msg: str):
        if user_id not in self.conversations:
            self.conversations[user_id] = []
        self.conversations[user_id].append(
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg}
        )
        if len(self.conversations[user_id]) > self.max_turns:
            self.summaries[user_id] = summarize_conversation(
                self.conversations[user_id]
            )
            self.conversations[user_id] = []
```

#### 3.2.2 上下文压缩方案

**技术一：LLM-Based 压缩**

使用 LLM 判断哪些内容重要，保留核心信息。

```java
public class ContextCompressor {

  private final OpenAIClient llmClient;

  public String compress(String context, int maxTokens) throws IOException {
    String prompt = String.format("""
        请将以下文本压缩到约 %d tokens，保留核心信息和关键细节：
        
        原文：
        %s
        
        压缩后：
        """, maxTokens, context);

    return llmClient.chat("gpt-4o", List.of(
        new Message("user", prompt)
    ));
  }
}
```

**技术二：Selective Context**

通过计算 token 重要性得分，保留高得分内容。

```python
from collections import Counter
import re

def tokenize(text: str) -> list[str]:
    return re.findall(r'\b\w+\b', text.lower())

def calculate_importance(text: str) -> float:
    tokens = tokenize(text)
    freq = Counter(tokens)
    tfidf = {tok: count / len(tokens) for tok, count in freq.items()}
    return sum(tfidf.values())

def selective_context(context: str, max_tokens: int) -> str:
    chunks = context.split("\n\n")
    scored = [(chunk, calculate_importance(chunk)) for chunk in chunks]
    scored.sort(key=lambda x: x[1], reverse=True)

    result = []
    current_tokens = 0
    for chunk, _ in scored:
        chunk_tokens = len(tokenize(chunk))
        if current_tokens + chunk_tokens <= max_tokens:
            result.append(chunk)
            current_tokens += chunk_tokens
    return "\n\n".join(result)
```

**技术三：滑动窗口 + 摘要混合**

结合滑动窗口和摘要两种策略，兼顾近期细节和长期关键信息。

```python
class HybridMemory:
    def __init__(self, window_size: int = 10, summary_threshold: int = 50):
        self.window_size = window_size
        self.summary_threshold = summary_threshold
        self.recent: list[dict] = []
        self.summaries: list[str] = []

    def add(self, user_msg: str, assistant_msg: str):
        self.recent.append(
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg}
        )
        if len(self.recent) > self.window_size * 2:
            window_summary = summarize_conversation(self.recent[:self.window_size * 2])
            self.summaries.append(window_summary)
            self.recent = self.recent[self.window_size * 2:]

    def get_context(self) -> str:
        parts = []
        if self.summaries:
            parts.append("【历史摘要】\n" + "\n".join(self.summaries[-3:]))
        if self.recent:
            parts.append("【最近对话】\n" + "\n".join(
                f"{m['role']}: {m['content']}" for m in self.recent[-6:]
            ))
        return "\n\n".join(parts)
```

#### 3.2.3 记忆总结对比

| 方案                | 优点                | 缺点           | 适用场景       |
|-------------------|-------------------|--------------|------------|
| 向量数据库             | 语义检索精准，可扩展        | 需要额外存储，检索有延迟 | 大量非结构化记忆   |
| 摘要式记忆             | 保留核心信息，token 消耗稳定 | 可能丢失细节       | 跨会话长期记忆    |
| LLM 压缩            | 压缩质量高，保留语义        | 计算成本高        | 需要精确保留关键信息 |
| Selective Context | 快速，简单             | 可能丢失上下文关联    | 实时性要求高的场景  |
| 滑动窗口 + 摘要         | 兼顾近期和长期           | 实现复杂         | 长期交互场景     |

### 3.3 工具调用（Function Calling）

Function Calling 允许模型在响应时调用预定义的函数，使 LLM 能够与外部系统交互。

#### 3.3.1 OpenAI Function Calling 格式

**工具定义（tools）格式：**

```json
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "获取指定城市的天气信息",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {
              "type": "string",
              "description": "城市名称，如北京、上海"
            },
            "unit": {
              "type": "string",
              "enum": [
                "celsius",
                "fahrenheit"
              ],
              "description": "温度单位"
            }
          },
          "required": [
            "city"
          ]
        }
      }
    }
  ]
}
```

**模型响应格式（返回 tool_calls）：**

```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_abc123",
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": "{\"city\": \"北京\", \"unit\": \"celsius\"}"
            }
          }
        ]
      }
    }
  ]
}
```

#### 3.3.2 Java 实现

```java
public class FunctionCallingClient {

  private final OpenAIClient llmClient;

  public static final Map<String, Tool> TOOLS = Map.of(
      "get_weather", new GetWeatherTool(),
      "calculate", new CalculateTool(),
      "get_current_time", new GetCurrentTimeTool()
  );

  public record ToolCall(String id, String name, String arguments) {

  }

  public String chatWithTools(List<Message> messages) throws IOException {
    Map<String, Object> requestBody = new HashMap<>();
    requestBody.put("model", "gpt-4o");
    requestBody.put("messages", messages);
    requestBody.put("tools", buildToolsDefinition());

    Request request = new Request.Builder()
        .url(baseUrl + "/v1/chat/completions")
        .addHeader("Authorization", "Bearer " + apiKey)
        .post(RequestBody.create(objectMapper.writeValueAsString(requestBody)))
        .build();

    try (Response response = client.newCall(request).execute()) {
      Map<String, Object> responseBody = parseResponse(response);
      List<Map<String, Object>> toolCalls = getToolCalls(responseBody);

      if (toolCalls == null || toolCalls.isEmpty()) {
        return getContent(responseBody);
      }

      for (Map<String, Object> toolCall : toolCalls) {
        Map<String, Object> function = (Map<String, Object>) toolCall.get("function");
        String toolName = (String) function.get("name");
        String arguments = (String) function.get("arguments");
        String toolCallId = (String) toolCall.get("id");

        Object result = executeTool(toolName, arguments);
        messages.add(new Message("assistant", null, List.of(new ToolCall(toolCallId, toolName, arguments))));
        messages.add(new Message("tool", objectMapper.writeValueAsString(result),
            null, toolCallId));
      }

      return chatWithTools(messages);
    }
  }

  private Object executeTool(String toolName, String arguments) {
    Tool tool = TOOLS.get(toolName);
    if (tool == null) {
      return Map.of("error", "未知工具: " + toolName);
    }
    return tool.execute(arguments);
  }

  private List<Map<String, Object>> buildToolsDefinition() {
    return List.of(
        Map.of(
            "type", "function",
            "function", Map.of(
                "name", "get_weather",
                "description", "获取指定城市的天气信息",
                "parameters", Map.of(
                    "type", "object",
                    "properties", Map.of(
                        "city", Map.of("type", "string", "description", "城市名称"),
                        "unit", Map.of("type", "string", "enum", List.of("celsius", "fahrenheit"))
                    ),
                    "required", List.of("city")
                )
            )
        )
    );
  }
}
```

### 3.4 工具（Tools）

#### 3.4.1 时间查询工具

```python
from datetime import datetime
import pytz

@tool_call("get_current_time")
def get_current_time(timezone: str = "Asia/Shanghai") -> dict:
    tz = pytz.timezone(timezone)
    now = datetime.now(tz)
    return {
        "time": now.strftime("%Y-%m-%d %H:%M:%S"),
        "timezone": timezone,
        "weekday": now.strftime("%A"),
        "date": now.strftime("%Y-%m-%d")
    }

@tool_call("convert_timezone")
def convert_timezone(time_str: str, from_tz: str, to_tz: str) -> dict:
    from_zone = pytz.timezone(from_tz)
    to_zone = pytz.timezone(to_tz)

    dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    dt = from_zone.localize(dt)

    converted = dt.astimezone(to_zone)
    return {
        "original": f"{time_str} ({from_tz})",
        "converted": f"{converted.strftime('%Y-%m-%d %H:%M:%S')} ({to_tz})"
    }
```

#### 3.4.2 计算器工具

```python
import math
import re

@tool_call("calculate")
def calculate(expression: str) -> dict:
    try:
        safe_dict = {
            "abs": abs, "round": round, "min": min, "max": max,
            "pow": pow, "sqrt": math.sqrt, "sin": math.sin,
            "cos": math.cos, "tan": math.tan, "log": math.log,
            "pi": math.pi, "e": math.e
        }
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return {"expression": expression, "result": float(result)}
    except Exception as e:
        return {"expression": expression, "error": str(e)}

@tool_call("calculate_statistics")
def calculate_statistics(numbers: list[float]) -> dict:
    if not numbers:
        return {"error": "数字列表不能为空"}

    sorted_numbers = sorted(numbers)
    n = len(numbers)

    return {
        "count": n,
        "sum": sum(numbers),
        "mean": sum(numbers) / n,
        "median": (sorted_numbers[n // 2] if n % 2 == 1
                   else (sorted_numbers[n // 2 - 1] + sorted_numbers[n // 2]) / 2),
        "min": min(numbers),
        "max": max(numbers),
        "variance": sum((x - sum(numbers) / n) ** 2 for x in numbers) / n,
        "std_dev": math.sqrt(sum((x - sum(numbers) / n) ** 2 for x in numbers) / n)
    }
```

#### 3.4.3 搜索工具

```python
from typing import Any

@tool_call("web_search")
def web_search(query: str, num_results: int = 5) -> dict:
    search_results = [
        {"title": f"关于 {query} 的结果 1", "url": "https://example.com/1", "snippet": "这是搜索结果的摘要..."},
        {"title": f"关于 {query} 的结果 2", "url": "https://example.com/2", "snippet": "这是搜索结果的摘要..."},
    ]
    return {"query": query, "results": search_results[:num_results]}

@tool_call("knowledge_base_search")
def knowledge_base_search(query: str, top_k: int = 3) -> dict:
    relevant_docs = [
        {"content": "文档1的内容摘要...", "source": "内部知识库", "relevance": 0.95},
        {"content": "文档2的内容摘要...", "source": "技术文档", "relevance": 0.87},
    ]
    return {"query": query, "documents": relevant_docs[:top_k]}
```

#### 3.4.4 Java 工具实现

```java
public interface Tool {

  String getName();

  String getDescription();

  Map<String, Object> getParameters();

  Object execute(Map<String, Object> args);
}

public class GetCurrentTimeTool implements Tool {

  @Override
  public String getName() {
    return "get_current_time";
  }

  @Override
  public String getDescription() {
    return "获取当前时间，支持不同时区";
  }

  @Override
  public Map<String, Object> getParameters() {
    return Map.of(
        "type", "object",
        "properties", Map.of(
            "timezone", Map.of(
                "type", "string",
                "description", "时区，如 Asia/Shanghai",
                "default", "Asia/Shanghai"
            )
        )
    );
  }

  @Override
  public Object execute(Map<String, Object> args) {
    String timezone = (String) args.getOrDefault("timezone", "Asia/Shanghai");
    ZonedDateTime now = ZonedDateTime.now(ZoneId.of(timezone));
    return Map.of(
        "time", now.format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")),
        "timezone", timezone,
        "weekday", now.getDayOfWeek().name()
    );
  }
}

public class CalculatorTool implements Tool {

  private static final Set<String> SAFE_FUNCTIONS = Set.of(
      "abs", "round", "min", "max", "pow", "sqrt", "sin", "cos", "tan", "log", "pi", "e"
  );

  @Override
  public String getName() {
    return "calculate";
  }

  @Override
  public String getDescription() {
    return "执行数学计算表达式";
  }

  @Override
  public Map<String, Object> getParameters() {
    return Map.of(
        "type", "object",
        "properties", Map.of(
            "expression", Map.of(
                "type", "string",
                "description", "数学表达式，如 2+2, sqrt(16), sin(pi/2)"
            )
        ),
        "required", List.of("expression")
    );
  }

  @Override
  public Object execute(Map<String, Object> args) {
    String expression = (String) args.get("expression");
    try {
      double result = evalExpression(expression);
      return Map.of("expression", expression, "result", result);
    }
    catch (Exception e) {
      return Map.of("expression", expression, "error", e.getMessage());
    }
  }

  private double evalExpression(String expr) {
    return new ExpressionEvaluator().evaluate(expr);
  }
}
```

### 3.5 MCP 协议

MCP（Model Context Protocol）是一种开放协议，用于将 AI 模型连接到外部工具和数据源。与 Function Calling 不同，MCP
提供了一种标准化的方式来定义、管理和调用工具，特别适合连接远程服务。

#### 3.5.1 MCP 架构

```
┌─────────────┐     MCP      ┌─────────────┐
│   AI 应用   │◄────────────►│  MCP Host   │
└─────────────┘              └──────┬──────┘
                                   │
                    ┌──────────────┼──────────────┐
                    │              │              │
               ┌────▼────┐   ┌────▼────┐   ┌────▼────┐
               │  Local  │   │ Remote  │   │ Remote  │
               │  Tool   │   │  Tool   │   │  Tool   │
               └─────────┘   │ Server  │   │ Server  │
                             └─────────┘   └─────────┘
```

#### 3.5.2 MCP Server 分类

| 类型     | 说明                         | 示例         |
|--------|----------------------------|------------|
| Local  | 运行在本地进程，通过 stdin/stdout 通信 | 文件系统、Git   |
| Remote | 通过 HTTP/WebSocket 远程调用     | 数据库、API 服务 |

#### 3.5.3 Python 连接 MCP Server

**使用 mcp SDK 连接：**

```python
from mcp.client import MCPClient
import asyncio

async def connect_to_mcp_server():
    async with MCPClient("http://localhost:8090/mcp") as client:
        tools = await client.list_tools()
        print(f"可用工具: {[t.name for t in tools]}")

        result = await client.call_tool("get_weather", {"city": "北京"})
        print(result)

asyncio.run(connect_to_mcp_server())
```

**使用 LangChain MCP 集成：**

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

async def main():
    async with MultiServerMCPClient(
        {
            "weather": {
                "command": "python",
                "args": ["/path/to/weather_server.py"],
                "transport": "stdio"
            },
            "search": {
                "url": "http://localhost:8090/mcp",
                "transport": "http"
            }
        }
    ) as client:
        tools = client.get_tools()
        llm = ChatOpenAI(model="gpt-4o")

        # 绑定工具到 LLM
        llm_with_tools = llm.bind_tools(tools)
        response = llm_with_tools.invoke("北京今天天气怎么样？")
        print(response)
```

#### 3.5.4 Java 连接 MCP Server

**使用 Spring AI Alibaba MCP：**

```java

@SpringBootApplication
public class MCPApplication {

  public static void main(String[] args) {
    SpringApplication.run(MCPApplication.class, args);
  }

  @Bean
  public MCPClient mcpClient() {
    return MCPClient.of("http://localhost:8090/mcp");
  }
}

@Service
public class WeatherService {

  private final MCPClient mcpClient;

  public WeatherService(MCPClient mcpClient) {
    this.mcpClient = mcpClient;
  }

  public String queryWeather(String city) {
    return mcpClient.call("get_weather", Map.of("city", city))
        .map(Object::toString)
        .orElse("查询失败");
  }
}
```

**原生 MCP 协议实现：**

```java
public class NativeMCPClient {

  private final OkHttpClient httpClient = new OkHttpClient();
  private final String baseUrl;

  public NativeMCPClient(String baseUrl) {
    this.baseUrl = baseUrl;
  }

  public Map<String, Object> listTools() throws IOException {
    Request request = new Request.Builder()
        .url(baseUrl + "/tools/list")
        .get()
        .build();

    try (Response response = httpClient.newCall(request).execute()) {
      return parseResponse(response);
    }
  }

  public Map<String, Object> callTool(String toolName, Map<String, Object> arguments) throws IOException {
    Map<String, Object> requestBody = new HashMap<>();
    requestBody.put("name", toolName);
    requestBody.put("arguments", arguments);

    Request request = new Request.Builder()
        .url(baseUrl + "/tools/call")
        .post(RequestBody.create(toJson(requestBody), MediaType.APPLICATION_JSON))
        .build();

    try (Response response = httpClient.newCall(request).execute()) {
      return parseResponse(response);
    }
  }

  private String toJson(Map<String, Object> obj) {
    return new ObjectMapper().writeValueAsString(obj);
  }

  private Map<String, Object> parseResponse(Response response) throws IOException {
    return new ObjectMapper().readValue(
        response.body().string(),
        new TypeReference<Map<String, Object>>() {
        }
    );
  }
}
```

#### 3.5.5 MCP Server 开发示例

**Python MCP Server：**

```python
from mcp.server import MCPServer
from mcp.types import Tool
import asyncio

server = MCPServer()

@server.tool()
def get_weather(city: str, unit: str = "celsius") -> dict:
    """获取城市天气信息"""
    return {"city": city, "temperature": 22, "unit": unit, "condition": "晴"}

@server.tool()
def calculate(expression: str) -> dict:
    """数学计算"""
    result = eval(expression, {"__builtins__": {}}, {
        "sqrt": __import__("math").sqrt,
        "pi": __import__("math").pi
    })
    return {"expression": expression, "result": result}

async def main():
    await server.run(transport="stdio")

if __name__ == "__main__":
    asyncio.run(main())
```

**启动本地 MCP Server：**

```bash
python weather_server.py
```

#### 3.5.6 MCP 工具注册到 Agent

将 MCP 工具与 Function Calling 结合：

```python
from mcp.client import MCPClient

async def create_mcp_enabled_agent():
    async with MCPClient("http://mcp-server:8090") as mcp_client:
        mcp_tools = await mcp_client.list_tools()

        tools_definition = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.inputSchema
                }
            }
            for t in mcp_tools
        ]

        def call_mcp_tool(tool_name: str, arguments: dict):
            return asyncio.run(mcp_client.call_tool(tool_name, arguments))

        return tools_definition, call_mcp_tool
```

### 3.6 基于以上技术实现一个简单的 Agent

基于以上技术栈（LLM + Memory + Function Calling + Tools），我们可以构建一个简单的 Agent。

#### 3.6.1 Agent 核心架构

```
┌─────────────────────────────────────────────────────────┐
│                        Agent                            │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│  │   Memory    │◄──►│     LLM     │◄──►│    Tools    │ │
│  │  (记忆管理)  │    │   (大脑)    │    │   (四肢)    │ │
│  └─────────────┘    └─────────────┘    └─────────────┘ │
│         │                  │                  │         │
│         └──────────────────┼──────────────────┘         │
│                            ▼                            │
│                   ┌─────────────────┐                  │
│                   │   Orchestrator   │                  │
│                   │    (协调器)      │                  │
│                   └─────────────────┘                  │
└─────────────────────────────────────────────────────────┘
```

#### 3.6.2 Python 简单 Agent 实现

```python
from dataclasses import dataclass, field
from typing import Callable
import json

@dataclass
class Message:
    role: str
    content: str
    tool_call_id: str = None

@dataclass
class AgentConfig:
    name: str
    role: str
    tools: list[dict] = field(default_factory=list)
    memory_limit: int = 10

class SimpleAgent:

    def __init__(self, config: AgentConfig, llm_client, memory_store):
        self.config = config
        self.llm = llm_client
        self.memory = memory_store
        self.conversation_history: list[Message] = []

    def add_system_prompt(self):
        return f"""你是一个{self.config.role}。

你可以使用以下工具来完成任务：
{json.dumps(self.config.tools, ensure_ascii=False, indent=2)}

当需要使用工具时，调用相应的函数。
当你已经获得足够的信息来回答用户问题时，直接回答。"""

    def invoke(self, user_input: str) -> str:
        messages = [Message("system", self.add_system_prompt())]
        recalled = self.memory.recall(user_input)
        if recalled:
            messages.append(Message("system", f"【相关记忆】\n{recalled}"))
        messages.extend(self.conversation_history[-self.config.memory_limit:])
        messages.append(Message("user", user_input))

        response = self.llm.chat_with_tools(messages, self.config.tools)

        while response.tool_calls:
            for tool_call in response.tool_calls:
                result = self._execute_tool(
                    tool_call.function.name,
                    json.loads(tool_call.function.arguments)
                )
                messages.append(Message("assistant", "", tool_call.id))
                messages.append(Message("tool", json.dumps(result), tool_call.id))
            response = self.llm.chat_with_tools(messages, self.config.tools)

        final_response = response.content
        self.conversation_history.extend([
            Message("user", user_input),
            Message("assistant", final_response)
        ])
        self.memory.add(user_input, final_response)

        return final_response

    def _execute_tool(self, tool_name: str, arguments: dict) -> dict:
        tool_map = {
            "get_weather": get_weather,
            "calculate": calculate,
            "get_current_time": get_current_time,
        }
        return tool_map.get(tool_name, lambda **args: {"error": "未知工具"})(**arguments)
```

#### 3.6.3 Java 简单 Agent 实现

```java
public class SimpleAgent {

  private final AgentConfig config;
  private final LLMClient llmClient;
  private final MemoryStore memoryStore;
  private final List<Message> conversationHistory = new ArrayList<>();

  public record Message(String role, String content, String toolCallId) {

  }

  public SimpleAgent(AgentConfig config, LLMClient llmClient, MemoryStore memoryStore) {
    this.config = config;
    this.llmClient = llmClient;
    this.memoryStore = memoryStore;
  }

  public String invoke(String userInput) throws IOException {
    List<Message> messages = new ArrayList<>();
    messages.add(new Message("system", buildSystemPrompt(), null));

    List<String> recalled = memoryStore.recall(userInput);
    if (!recalled.isEmpty()) {
      messages.add(new Message("system", "【相关记忆】\n" + String.join("\n", recalled), null));
    }

    if (conversationHistory.size() > config.memoryLimit() * 2) {
      List<Message> recentHistory = conversationHistory.subList(
          conversationHistory.size() - config.memoryLimit() * 2,
          conversationHistory.size()
      );
      messages.addAll(recentHistory);
    }
    else {
      messages.addAll(conversationHistory);
    }

    messages.add(new Message("user", userInput, null));

    LLMResponse response = llmClient.chatWithTools(messages, config.tools());

    while (response.hasToolCalls()) {
      for (ToolCall toolCall : response.toolCalls()) {
        Object result = executeTool(toolCall.name(), toolCall.arguments());
        messages.add(new Message("assistant", null, toolCall.id()));
        messages.add(new Message("tool", toJson(result), toolCall.id()));
      }
      response = llmClient.chatWithTools(messages, config.tools());
    }

    String finalResponse = response.content();
    conversationHistory.add(new Message("user", userInput, null));
    conversationHistory.add(new Message("assistant", finalResponse, null));
    memoryStore.add(userInput, finalResponse);

    return finalResponse;
  }

  private String buildSystemPrompt() {
    return String.format("你是一个%s。\n\n你可以使用以下工具来完成任务：\n%s",
        config.role(),
        toJson(config.tools())
    );
  }

  private Object executeTool(String toolName, Map<String, Object> arguments) {
    return switch (toolName) {
      case "get_weather" -> getWeather(arguments);
      case "calculate" -> calculate(arguments);
      case "get_current_time" -> getCurrentTime(arguments);
      default -> Map.of("error", "未知工具: " + toolName);
    };
  }
}

public record AgentConfig(
    String name,
    String role,
    List<Map<String, Object>> tools,
    int memoryLimit
) {

}
```

#### 3.6.4 Agent 运行示例

```python
from openai import OpenAI

llm_client = OpenAI(api_key="your-api-key")
memory_store = HybridMemory(window_size=10)

config = AgentConfig(
    name="助手",
    role="编程助手，可以回答技术问题、执行计算、查询天气等",
    tools=TOOLS_DEFINITION,
    memory_limit=20
)

agent = SimpleAgent(config, llm_client, memory_store)

while True:
    user_input = input("你: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    response = agent.invoke(user_input)
    print(f"助手: {response}")
```

#### 3.6.5 ReAct 模式（推理 + 行动）

一种更高级的 Agent 循环模式：

```python
class ReActAgent:

    def __init__(self, config: AgentConfig, llm_client, memory_store):
        self.config = config
        self.llm = llm_client
        self.memory = memory_store

    def invoke(self, user_input: str, max_iterations: int = 5) -> str:
        thoughts = []
        actions = []
        observations = []

        context = [Message("system", self.config.system_prompt)]
        context.extend(self.memory.recall(user_input))
        context.append(Message("user", user_input))

        for i in range(max_iterations):
            prompt = self._build_react_prompt(thoughts, actions, observations)
            context.append(Message("user", prompt))

            response = self.llm.chat(context)

            if response.function_call:
                action = response.function_call
                actions.append(action)
                observation = self._execute_tool(action.name, action.arguments)
                observations.append(observation)
                thoughts.append(response.thought)
            else:
                final_answer = response.content
                self.memory.add(user_input, final_answer)
                return final_answer

        return "已达到最大迭代次数，无法完成任务"

    def _build_react_prompt(self, thoughts, actions, observations) -> str:
        prompt = "请按照以下格式思考和行动：\n\n"
        for i, (thought, action, observation) in enumerate(zip(thoughts, actions, observations)):
            prompt += f"思考 {i+1}: {thought}\n"
            prompt += f"行动 {i+1}: {action}\n"
            prompt += f"观察 {i+1}: {observation}\n\n"
        prompt += "请给出你的下一个思考和行动："
        return prompt
```

#### 3.6.6 技术选型对比

| 框架         | 语言     | 特点              | 适用场景          |
|------------|--------|-----------------|---------------|
| LangChain  | Python | 功能全面，文档丰富       | 快速原型开发        |
| LangGraph  | Python | 状态机模式，支持复杂流程    | 复杂 Agent 编排   |
| Spring AI  | Java   | 企业级，Spring 生态集成 | Java 企业应用     |
| LlamaIndex | Python | 专注知识检索增强        | RAG 应用        |
| AutoGen    | Python | 多 Agent 协作      | Agent Team 场景 |
| CrewAI     | Python | 角色扮演，多 Agent 协作 | 复杂任务分解        |

---

## 四、Agent 进阶：多 Agent 协作（10 min）

> 目的：讲 A2A 协议 + Agent Team 概念，这是亮点

### 4.1 为什么单个 Agent 不够用？

- 上下文有限、专业度不足、容易"跑偏"

### 4.2 Agent Team 技术说明

**定义**：Agent Team（多智能体协作）指多个具备不同角色、工具与记忆的 Agent，在统一目标下通过消息传递与编排策略协同完成任务；与「一个模型包打天下」的单体
Agent 相比，更接近「项目组」式的分工与接力。

**常见编排形态**：

| 形态                  | 要点                         | 适用              |
|---------------------|----------------------------|-----------------|
| 中心化编排（Orchestrator） | 协调者拆解任务、分派子 Agent、汇总结果     | 流程清晰、需要强可控的企业场景 |
| 层级 / 主管链            | 上层 Agent 规划，下层 Agent 执行与回报 | 复杂流水线、多级审批类任务   |
| 对等协作                | Agent 平等对话、协商分工（如群聊式）      | 创意、讨论、多视角评审     |
| 人机混合                | 关键节点由人确认或补位                | 高风险操作、合规与责任边界   |

**工程上通常要解决的问题**：

- **任务分解与交接**：谁负责哪一段输出、输入输出契约（结构化字段，而非纯自然语言糊在一起）。
- **上下文与记忆**：共享白板（团队记忆）与私有工作记忆如何划分，避免全员上下文爆炸。
- **工具与权限**：每个 Agent 可见的 MCP / API 不同，避免「一个 Agent 拿错钥匙」导致越权。
- **冲突与终止**：多 Agent 意见不一致时的裁决策略（投票、协调者拍板、回退到人）。
- **可观测性**：Trace / 日志要能还原「哪一步是哪个 Agent 做的」，否则线上难以排障。

**与单 Agent 的取舍**：协作带来更高上限与专业分工，但会增加延迟、Token 成本与调试复杂度；适合任务天然可拆分、或单 Agent
反复失败的场景。

### 4.3 Agent Team 典型架构

```
协调者 (Orchestrator)
    ├── 研究 Agent（负责搜索信息）
    ├── 写作 Agent（负责生成内容）
    ├── 审核 Agent（负责质量把关）
    └── 执行 Agent（负责对外操作）
```

- 分工明确，各司其职
- 类比：一个项目组，不是一个全栈独苗

### 4.4 A2A 协议（Agent2Agent）

**定位**：开放标准，解决「不同框架、不同厂商的 Agent 如何互相发现、发任务、拿结果」的问题；类比 HTTP 让浏览器与服务器互通，A2A 让
Agent 与 Agent 在**不暴露彼此内部实现**的前提下协作。

**核心思路**：

- **能力发现**：远端 Agent 通过标准化的 **Agent Card**（JSON 元数据，常见入口为 `/.well-known/agent.json`）声明自己是谁、能做什么、如何调用。
- **任务模型**：以 **Task** 为生命周期单位，围绕任务收发状态更新、产物（Artifact）、取消与查询等，而不是随意拼接自然语言私聊。
- **传输与形态**：常见实现为 **JSON-RPC 2.0 over HTTP(S)**，并支持同步调用、**SSE 流式**推送中间结果、以及长任务的异步 /
  轮询等模式。
- **不透明执行（Opaque Execution）**：协作方只按协议交换任务与消息，**不要求**共享对方的提示词、记忆、工具实现细节，便于安全边界与多租户。

**适用**：跨团队/跨产品的 Agent 编排、平台型「Agent 市场」、需要可观测任务链路的生产环境。

**参考
**：[A2A 规范（a2aproject）](https://a2aproject.github.io/A2A/latest/specification/)、[Google 开发者博客介绍](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/)

### 4.5 ACP 协议（Agent Client Protocol）

**定位**：开放标准，解决 **Agent 与宿主客户端（Host）** 之间的会话问题——宿主可以是 IDE、CLI、桌面应用等；Agent
在宿主提供的沙箱能力（文件、终端、权限确认等）下完成工作。**不是** A2A 那种「远程 Agent 与 Agent 对等协作」的同一件事。

**核心思路**：

- **JSON-RPC 2.0**：方法调用 + 单向通知，便于宿主与 Agent 进程解耦。
- **典型流程**：`initialize`（能力协商）→ 按需 `authenticate` → `session/new` 或 `session/load` → `session/prompt`
  驱动多轮交互；宿主在需要时回调 Agent 声明的 **client 方法**（如申请读文件、执行命令前的授权）。
- **传输**：常见为 **stdio**（子进程标准输入输出按行 JSON-RPC），也在演进 **可流式 HTTP** 等绑定，便于远程宿主。

**与 MCP、A2A 的区分（一句话）**：

| 协议  | 主要关系                        |
|-----|-----------------------------|
| MCP | 模型 ↔ 工具/数据源（上下文与工具生态）       |
| A2A | Agent ↔ Agent（跨系统任务协作）      |
| ACP | Agent ↔ 客户端宿主（会话、权限、IDE 集成） |

**参考**：[Agent Client Protocol 概览](https://agentclientprotocol.com/protocol/overview)

> 说明：若产品文档里把「跨企业即时通讯上的协作」也简写为 ACP，建议在对外分享时**单独定义业务含义**，避免与业界 **Agent Client
Protocol** 混用。


---

## 五、当前系统的应用

---

## 六、一个想法：基于 ACP, A2A 协议实现一个协作式 AI 系统

涉及技术

- agent 创建: LLM, Memory, Function Calling, Tools, MCP, Skill
- agent 协作: Agent Team, ACP, A2A 协议
- 企业 IM:

本系统以企业级场景为例，实现一个协作式 AI 系统，支持多个 Agent 之间的协作，完成复杂任务。

假设有一个组织架构如下：

```
风控开发 负责人 嬴政
    ├── 王翦 负责风控模型平台的管理，包括模型评估、部署、监控和维护。
    ├── 蒙恬 负责自营需求的分析和处理。包括需求的接收、解析、验证和处理。
    ├── 李信 负责跑批需求的处理。包括跑批任务的接收、解析、验证和处理。
    └── 章邯 负责风控系统与外部的交互，包括数据的导入、导出、对外操作。
模型开发 负责人 刘邦
    ├── 韩信 负责通用模型的训练、评估和维护。
    └── 萧何 负责自营模型的部署、监控和维护。
数据分析和风控规则管理 负责人 唐太宗
    ├── 房玄龄 负责自营规则相关业务的数据分析，规则的制定和维护。
    └── 杜如晦 负责自营规则相关业务的数据分析，规则的制定和维护。
```

其中每个人可以根据自己的专业和知识，负责不同的任务。每个人可以根据自己的日常工作内容、个人知识、个人工具等配置出一个对应的专属的
Agent，负责完成自己的任务。同时可以将自己的专属agent 分享给其他指定成员，指定部门以此完成任务的协作。

具体预期实现如下：

### 6.1 创建专属的 Agent

> 王翦 负责风控模型平台的管理，包括模型评估、部署、监控和维护。

以王翦为例

1. 配置名称，角色与设定
    - 名称：王剪
    - 角色：风控模型平台的管理
    - 设定：负责风控模型平台的管理，包括模型评估、部署、监控和维护。
2. （可选）配置私有知识库
3. （可选）配置专属工具（手动配置或者 MCP Server）
    - 模型评估工具
    - 模型部署工具
    - 模型监控工具
4. （可选）配置 Skill
5. （可选）配置 工作流（workflow）
6. 配置权限范围
    - 仅对指定成员可见
    - 仅对指定部门可见
    - 仅对指定角色可见

### 6.2 单聊协作（真人与 Agent 协作）

为了简化表述将个人专属的 Agent 称为"分身"。

假设所有的人已经根据自己的工作内容、个人知识、个人工具等配置出一个对应的专属的 Agent，并设置了合理的分享权限。

假设王翦将分身权限设置同部门使用，当李信想与王剪协作时，他可以优先与王翦的分身进行对话，完成任务。如果王翦分身无法完成任务时，分身可以根据情况智能通知真人进行协调（请祖师爷上身）。

跨部门同样的原理，当不同部门的成员想与王部门的成员协作时，也可以通过分身进行对话，完成任务。如果分身无法完成任务时，分身可以根据情况智能通知真人进行协调。

### 6.3 组内协作

在组织结构模式下，默认会为每个部门的 leader 配置一个专属的 manager agent，负责协调部门内的 agent 协作。

以风控开发为例，假设王翦、蒙恬、李信、章邯将分身配置为一级部门可用。当嬴政发起一个任务时，4 个分身可以在部门manager(
内置部门manager) 下进行协作。如果分身无法完成任务时，分身可以根据情况智能通知真人进行协调。如果真人无法协调。

### 6.4 群聊协作

假设有需要配置一个全新的模型需要房玄龄，韩信，王翦协作配置完成数据分析，模型训练，模型部署和评估。则可以将 3
个组成群聊的方式，在建群的时候，默认会讲对应 3 人的分身加入群聊，组成 3人3分身的 6 人群聊。

当房玄龄发起一个任务时，房玄龄分身可适当补充细节，韩信分身，王翦分身优先处理任务。如果分身无法完成任务时，分身可以根据情况智能通知对应真人进行协调。

### 6.5 外部沟通模式（ACP）

当两个不同的企业员工想进行协作时，可以通过扫码的方式组建群聊，实现外部沟通。类似群聊功能，只不过人员来自不同的企业。

### ToC 协作

当前模式下，也可以放弃企业功能，改为通过 ACP 协议进行协作，实现 C 端业务，类似给每个微信用户配置分身。

---


