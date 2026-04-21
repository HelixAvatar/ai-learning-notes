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
import llm.openai.Single;
import lombok.extern.slf4j.Slf4j;

@Slf4j
public class OpenAISimpleChat {


  public static ChatCompletion callWithMessage(
      List<ChatCompletionMessageParam> messages) throws ApiException, NoApiKeyException, InputRequiredException {

    ChatCompletionCreateParams params = ChatCompletionCreateParams.builder()
        .messages(messages)
        .model("dashscope/qwen3.5")
        .build();

    return Single.OPEN_AI_CLIENT.chat().completions().create(params);
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
