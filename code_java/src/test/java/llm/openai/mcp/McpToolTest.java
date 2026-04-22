package llm.openai.mcp;


import io.modelcontextprotocol.spec.McpSchema.ListToolsResult;
import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;

@Slf4j
class McpToolTest {

  @Test
  void listTools() {
    McpTool mcpTool = new McpTool("http://127.0.0.1:18000");
    ListToolsResult listToolsResult = mcpTool.listTools();
    log.info("listToolsResult: {}", listToolsResult);
  }
}