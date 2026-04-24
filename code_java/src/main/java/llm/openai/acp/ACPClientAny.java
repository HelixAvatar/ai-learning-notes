package llm.openai.acp;

import com.agentclientprotocol.sdk.client.*;
import com.agentclientprotocol.sdk.client.transport.*;
import com.agentclientprotocol.sdk.spec.AcpSchema.*;
import java.util.List;
import lombok.extern.slf4j.Slf4j;

@Slf4j
public class ACPClientAny {

  public static void main(String[] args) {
    var params = AgentParameters.builder("opencode").arg("acp").build();
    var transport = new StdioAcpClientTransport(params);

    AcpSyncClient client = AcpClient.sync(transport)
        .sessionUpdateConsumer(notification -> {
          try {
            if (notification.update() instanceof AgentMessageChunk msg) {
              log.info("Agent message: {}", msg);
            }
          }
          catch (Exception e) {
            log.error("Error processing agent update:", e);
          }
        })
        .build();

    client.initialize();
    var session = client.newSession(new NewSessionRequest("/workspace", List.of()));
    var response = client.prompt(new PromptRequest(
        session.sessionId(),
        List.of(new TextContent("帮我用 Python 写一个排序算法"))
    ));

    log.info("Response: {}", response);
    client.close();
  }
}
