package llm.openai.functioncalling;

import com.fasterxml.jackson.annotation.JsonClassDescription;
import com.fasterxml.jackson.annotation.JsonPropertyDescription;
import com.google.common.base.Strings;
import com.openai.models.chat.completions.ChatCompletion;
import com.openai.models.chat.completions.ChatCompletionCreateParams;
import com.openai.models.chat.completions.ChatCompletionMessageFunctionToolCall;
import com.openai.models.chat.completions.ChatCompletionMessageParam;
import com.openai.models.chat.completions.ChatCompletionToolMessageParam;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import llm.openai.Single;
import lombok.ToString;
import lombok.extern.slf4j.Slf4j;

@Slf4j
public class OpenAIFunctionCalling {


  @JsonClassDescription("分析特征的血缘关系")
  static class FeatureDataLineage {

    @JsonPropertyDescription("特征名称, 变量名可以包含英文数字和下划线")
    public String featureName;

    public String execute() {
      log.info("分析特征的血缘关系: {}", featureName);
      return "特征 " + featureName + " 的血缘关系分析结果如下：\n" +
          """
              ## 引入特征
              - a: 特征 a
              - b: 特征 b
              - c: 特征 c
              
              ## 计算过程
              featureName = a + b * c
              """;
    }
  }

  @ToString
  @JsonClassDescription("日程创建")
  static class ScheduleManagementCreate {

    @JsonPropertyDescription("日程名称")
    public String name;
    @JsonPropertyDescription("日程描述")
    public String description;
    @JsonPropertyDescription("日程时间, 格式为 yyyy-MM-dd hh:mm:ss")
    public String time;

    public String execute() {
      log.info("创建日程: {} 具体参数 {}", name, this);
      return "日程 " + name + " 创建成功";
    }
  }

  public static void main(String[] args) {
    String now = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd hh:mm:ss"));
    log.info("当前时间: {}", now);

    ChatCompletionCreateParams.Builder chatBuilder = ChatCompletionCreateParams.builder()
        .model("dashscope/qwen3.5")
        .maxCompletionTokens(12048)
        .addTool(FeatureDataLineage.class)
        .addTool(ScheduleManagementCreate.class)
        .addSystemMessage("当前时间为 " + now);

    List<String> questions = List.of(
        "今天天气怎样？",
        "分析 abc_dd_v2 的血缘关系",
        "明天 10 预定会议，分享 AI Function Calling",
        "进行旅游规划，要求从北京出去，第一站北海，第二站云南，最后回到北京。历时8-10天。"
    );

    for (int index = 0; index < questions.size(); index++) {
      log.info("第 {} 个请求: {}", index + 1, questions.get(index));
      chatBuilder.addUserMessage(questions.get(index));
      functionCallingDeal(chatBuilder);
    }

    printAllMessage(chatBuilder.build().messages());
  }

  static void functionCallingDeal(ChatCompletionCreateParams.Builder chatBuilder) {
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
            Object result = callFunction(toolCall.asFunction().function());
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

  private static Object callFunction(ChatCompletionMessageFunctionToolCall.Function function) {
    return switch (function.name()) {
      case "FeatureDataLineage" -> function.arguments(FeatureDataLineage.class).execute();
      case "ScheduleManagementCreate" -> function.arguments(ScheduleManagementCreate.class).execute();
      default -> throw new IllegalArgumentException("Unknown function: " + function.name());
    };
  }

  private static void printAllMessage(List<ChatCompletionMessageParam> messages) {
    log.info(Strings.repeat("\n", 10));
    log.info(Strings.repeat("=", 80));
    log.info("All messages: {} 条", messages.size());

    messages.forEach(message -> log.info("Message: {}", message));

  }
}
