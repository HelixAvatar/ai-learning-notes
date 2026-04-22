"""
FastMCP quickstart example.

Run from the repository root:
    uv run examples/snippets/servers/fastmcp_quickstart.py
"""
import time

from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("Demo", json_response=True, port=18000)


# Add an addition tool
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b


@mcp.tool()
def get_current_time() -> str:
    """Get the current time"""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


# Run with streamable HTTP transport
if __name__ == "__main__":
    mcp.run(transport="streamable-http")