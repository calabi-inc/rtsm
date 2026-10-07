# API

The memory is served over HTTP on one port; a WebSocket carries the dashboard stream and an MCP endpoint exposes the same queries to agent frameworks.

| service | address | notes |
|---|---|---|
| REST API | `http://localhost:8002` | objects, search, stats, health |
| MCP | `http://localhost:8002/mcp/sse` | when the `mcp` extra is installed and enabled |
| Calabi Lens ingest | `ws://localhost:8765/stream` | the phone's RGB-D + pose stream |
| [ROS 2 ingest](ros2.md) | subscriptions on the local ROS 2 graph (`--ros2`) | subscriber only; topics by rule or `io.ros2.topics`; `rtsm ros2 probe` first |
| Dashboard WebSocket | `ws://localhost:8083/ws` | only with `--viz` |

<div class="grid cards" markdown>

-   :material-api:{ .lg .middle } **REST API**

    ---

    Objects and their snapshots, semantic and spatial search, statistics, analytics, health, reset.

    [:octicons-arrow-right-24: REST](rest-api.md)

-   :material-language-python:{ .lg .middle } **Python Client**

    ---

    `rtsm.client.RtsmClient`: the REST API from Python with `requests` only, no perception stack installed.

    [:octicons-arrow-right-24: Python](python-client.md)

-   :material-robot-outline:{ .lg .middle } **MCP for AI Agents**

    ---

    Six tools over SSE or stdio for MCP clients (Claude, Cursor, LangGraph): the object, search and statistics queries of the REST API.

    [:octicons-arrow-right-24: MCP](mcp.md)

-   :material-lan-connect:{ .lg .middle } **WebSocket**

    ---

    The dashboard stream: point clouds and object updates, and the message types the viewer understands.

    [:octicons-arrow-right-24: WebSocket](websocket.md)

-   :material-robot-industrial:{ .lg .middle } **ROS 2 Ingest**

    ---

    The subscriber contract: topic roles and message types, encodings, pairing, pose, registration, QoS, refusal codes, `/stats.source`.

    [:octicons-arrow-right-24: ROS 2](ros2.md)

</div>
