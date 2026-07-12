"""MCP application instance.

Module ini hanya membuat objek ``mcp``. Registrasi tools dilakukan dari
``code_radar.server.__init__`` agar dependency graph tidak membentuk cycle
untuk static analyzer.
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("code-radar")
