package llm.openai;

import com.openai.client.OpenAIClient;
import com.openai.client.okhttp.OpenAIOkHttpClient;

public class Single {
  public static final OpenAIClient OPEN_AI_CLIENT = OpenAIOkHttpClient.builder()
      .baseUrl("http://127.0.0.1:28080/v1")
      .apiKey("empty")
      .build();
}
