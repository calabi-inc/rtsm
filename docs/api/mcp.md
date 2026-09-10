# MCP (Model Context Protocol)

RTSM exposes its spatial memory as MCP tools, so any AI agent can query "where is the coffee mug?" without custom REST code. MCP is the standard protocol for connecting AI models to external tools — supported by Claude, Cursor, LangGraph, CrewAI, and others.

## Two Transport Options

| Transport | Use case | How it works |
|-----------|----------|--------------|
| **SSE (embedded)** | Agent connects over HTTP | MCP server mounted on RTSM's API server at `/mcp/sse` |
| **stdio (standalone)** | Agent launches RTSM MCP as a subprocess | `rtsm-mcp` binary, communicates via stdin/stdout |

## Quick Start: SSE (embedded)

### 1. Enable in config

```yaml
# config/rtsm.yaml
mcp:
  enable: true
```

### 2. Start RTSM

```bash
pip install "rtsm[gpu]"
python -m rtsm --replay recordings/session1
```

The MCP endpoint is now live at `http://localhost:8002/mcp/sse`.

### 3. Connect your agent

**Claude Code** — add to your MCP settings:

```json
{
  "mcpServers": {
    "rtsm": {
      "url": "http://localhost:8002/mcp/sse"
    }
  }
}
```

**Cursor** — add to `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "rtsm": {
      "url": "http://localhost:8002/mcp/sse"
    }
  }
}
```

**Python (any agent framework):**

```python
from mcp import ClientSession
from mcp.client.sse import sse_client

async with sse_client("http://localhost:8002/mcp/sse") as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()

        # Query spatial memory
        result = await session.call_tool(
            "rtsm.semantic_query",
            arguments={"query": "coffee mug", "top_k": 3}
        )
        print(result.content[0].text)
```

## Quick Start: stdio (standalone)

The stdio transport is useful when your agent framework launches MCP servers as subprocesses.

### 1. Install

```bash
pip install "rtsm[gpu]"
```

### 2. Configure your agent

**Claude Code** — add to MCP settings:

```json
{
  "mcpServers": {
    "rtsm": {
      "command": "rtsm-mcp",
      "env": {
        "RTSM_API_URL": "http://localhost:8002"
      }
    }
  }
}
```

The `rtsm-mcp` command connects to a running RTSM instance via its REST API and exposes the same 6 tools over stdio.

## Available Tools

| Tool | Description | Example use |
|------|-------------|-------------|
| `rtsm.semantic_query` | Search by natural language | "where is the coffee mug?" |
| `rtsm.spatial_query` | Find objects near a 3D point | Objects within 2m of [1.0, 0.5, 0.8] |
| `rtsm.relational_query` | Find objects near a named object | "what is next to the laptop?" |
| `rtsm.list_objects` | List all tracked objects | Get full scene inventory |
| `rtsm.get_object` | Get details for one object | Full info by object ID |
| `rtsm.status` | System health and stats | Pipeline status, object counts |

### Tool Parameters

**`rtsm.semantic_query`**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | required | Natural language search query |
| `top_k` | int | 5 | Max results to return |
| `threshold` | float | 0.2 | Min cosine similarity (0-1) |

**`rtsm.spatial_query`**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x` | float | required | X coordinate (meters) |
| `y` | float | required | Y coordinate (meters) |
| `z` | float | required | Z coordinate (meters) |
| `radius_m` | float | 1.0 | Search radius in meters |

**`rtsm.relational_query`**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | required | Name of reference object |
| `radius_m` | float | 1.0 | Search radius in meters |

**`rtsm.get_object`**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `object_id` | string | required | Object ID |
| `include_vectors` | bool | false | Include embedding vectors |

## Example: Agent Asks "Where is the mug?"

```
Agent → rtsm.semantic_query(query="mug", top_k=1)

RTSM → {
  "query": "mug",
  "robot_pose": {
    "xyz": [0.12, 0.05, 0.31],
    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
    "timestamp": 1712345678.5,
    "frame_epoch": 0,
    "sensor_ts_ns": 683333172055083,
    "pose_clock": "sender",
    "age_s": 0.041,
    "stale": false
    /* diagnostic counters omitted -- see the REST API reference for the full payload */
  },
  "results": [
    {
      "id": "obj_231",
      "score": 0.85,
      "label_hint": "mug",
      "confirmed": true,
      "xyz_world": [2.31, -0.15, 1.18]
    }
  ]
}
```

The agent gets both the mug's position AND the robot's current position in one atomic response. It can immediately compute heading and distance to navigate there — no second API call needed.

All search responses (`semantic_query`, `spatial_query`, `relational_query`) include `robot_pose`. The `status` tool also returns it. RTSM stores but does not compute pose — it's a passthrough from the sensor (ARKit, RTABMap, etc.).
