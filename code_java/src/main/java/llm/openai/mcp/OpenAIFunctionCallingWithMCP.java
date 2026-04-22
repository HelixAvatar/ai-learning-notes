package llm.openai.mcp;

import com.alibaba.fastjson2.JSON;
import com.google.common.base.Strings;
import com.openai.models.chat.completions.ChatCompletion;
import com.openai.models.chat.completions.ChatCompletionCreateParams;
import com.openai.models.chat.completions.ChatCompletionMessageFunctionToolCall.Function;
import com.openai.models.chat.completions.ChatCompletionMessageParam;
import com.openai.models.chat.completions.ChatCompletionToolMessageParam;
import io.modelcontextprotocol.spec.McpSchema.ListToolsResult;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import llm.openai.Single;
import lombok.extern.slf4j.Slf4j;

@Slf4j
public class OpenAIFunctionCallingWithMCP {

  public static void main(String[] args) {
    String now = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd hh:mm:ss"));
    log.info("当前时间: {}", now);

    McpTool mcpTool = new McpTool("http://127.0.0.1:18000");

    ListToolsResult listToolsResult = mcpTool.listTools();

    listToolsResult.tools().forEach(tool -> log.info("Tool: {}", tool));

    ChatCompletionCreateParams.Builder chatBuilder = ChatCompletionCreateParams.builder()
        .model("dashscope/qwen3.5")
        .maxCompletionTokens(12048)
        .addSystemMessage("You are a helpful assistant with access to tools. "
            + "Use tools when needed to answer questions accurately. "
            + "Always explain what you're doing when using tools.");

    // 注册工具
    mcpTool.getFunctionDefinitions().forEach(chatBuilder::addFunctionTool);

    List<String> questions = List.of(
        "今天天气怎样？",
        "1 + 11 得多少？",
        "分析 芝麻分 的血缘关系",
        "明天 10 预定会议，分享 AI Function Calling"
    );

    for (int index = 0; index < questions.size(); index++) {
      log.info("第 {} 个请求: {}", index + 1, questions.get(index));
      chatBuilder.addUserMessage(questions.get(index));
      functionCallingDeal(mcpTool, chatBuilder);
    }

    printAllMessage(chatBuilder.build().messages());
  }

  static void functionCallingDeal(McpTool toolBox, ChatCompletionCreateParams.Builder chatBuilder) {
    final int maxIterations = 100;
    int iterations = 0;
    while (iterations < maxIterations) {
      iterations++;

      ChatCompletionCreateParams build = chatBuilder.build();
      log.debug("request before: {}", build.messages());
      AtomicBoolean hasToolCall = new AtomicBoolean(false);
      Single.OPEN_AI_CLIENT.chat().completions().create(chatBuilder.build())
          .choices()
          .stream()
          .map(ChatCompletion.Choice::message) // 只需要关系消息部分，暂时不处理其余部分
          .peek(chatBuilder::addMessage) // 将消息添加到会话中
          .flatMap(message -> message.toolCalls().stream().flatMap(Collection::stream)) // 展开 toolCalls 进行后续工具调用
          .forEach(toolCall -> {
            log.info("toolCall: {}", toolCall);

            Function function = toolCall.asFunction().function();

            Object result = toolBox.callTool(function.name(), JSON.parseObject(function.arguments()));

            // Add the tool call result to the conversation.
            chatBuilder.addMessage(ChatCompletionToolMessageParam.builder()
                .toolCallId(toolCall.asFunction().id())
                .contentAsJson(result)
                .build());
            hasToolCall.set(true);
          });

      if (!hasToolCall.get()) {
        break;
      }
      log.debug("request has tool call");
      if (iterations >= maxIterations) {
        log.warn("functionCallingDeal 已达到最大循环次数 {}，仍有未完成的工具调用链路", maxIterations);
      }
    }

    ChatCompletionCreateParams build = chatBuilder.build();
    log.debug("request after: {}", build.messages());
  }

  private static void printAllMessage(List<ChatCompletionMessageParam> messages) {
    log.info(Strings.repeat("\n", 10));
    log.info(Strings.repeat("=", 80));
    log.info("All messages: {} 条", messages.size());
    messages.forEach(message -> log.info("Message: {}", message));
  }
}
