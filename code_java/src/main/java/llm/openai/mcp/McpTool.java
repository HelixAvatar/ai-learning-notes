package llm.openai.mcp;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.openai.core.JsonValue;
import com.openai.models.FunctionDefinition;
import com.openai.models.FunctionParameters;
import com.openai.models.FunctionParameters.Builder;
import io.modelcontextprotocol.client.McpClient;
import io.modelcontextprotocol.client.McpSyncClient;
import io.modelcontextprotocol.client.transport.HttpClientStreamableHttpTransport;
import io.modelcontextprotocol.spec.McpSchema.CallToolRequest;
import io.modelcontextprotocol.spec.McpSchema.ClientCapabilities;
import io.modelcontextprotocol.spec.McpSchema.JsonSchema;
import io.modelcontextprotocol.spec.McpSchema.ListToolsResult;
import io.modelcontextprotocol.spec.McpSchema.Tool;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class McpTool {

  private McpSyncClient client;

  public McpTool(String url) {
    HttpClientStreamableHttpTransport transport = HttpClientStreamableHttpTransport
        .builder(url)
        .build();
    client = McpClient.sync(transport)
        .requestTimeout(Duration.ofSeconds(60))
        .capabilities(ClientCapabilities.builder()
            .roots(true)       // Enable roots capability
            .build())
        .build();
    client.initialize();
  }

  public ListToolsResult listTools() {
    return client.listTools();
  }

  public List<FunctionDefinition> getFunctionDefinitions() {
    List<FunctionDefinition> functionDefinitions = new ArrayList<>();
    for (Tool tool : client.listTools().tools()) {
      JsonSchema jsonSchema = tool.inputSchema();

      FunctionParameters.Builder paramsBuilder = FunctionParameters.builder()
          .putAdditionalProperty("type", JsonValue.from(jsonSchema.type()))
          .putAdditionalProperty("properties", JsonValue.from(jsonSchema.properties()));
      if (jsonSchema.required() != null && !jsonSchema.required().isEmpty()) {
        paramsBuilder.putAdditionalProperty("required", JsonValue.from(jsonSchema.required()));
      }

      functionDefinitions.add(FunctionDefinition.builder()
          .name(tool.name())
          .description(tool.description())
          .parameters(paramsBuilder.build())
          .build());
    }
    return functionDefinitions;
  }


  public Object callTool(String name, Map<String, Object> arguments) {
    CallToolRequest callToolRequest = CallToolRequest.builder().name(name).arguments(arguments).build();
    return client.callTool(callToolRequest);
  }
}
