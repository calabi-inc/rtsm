# Getting Started

From an empty machine to a first query in three pages. RTSM runs headless by default: a REST API on port 8002, an ingest WebSocket for Calabi Lens on 8765, and the 3-D dashboard only when you ask for it.

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } **Installation**

    ---

    `pip install rtsm[gpu]` with the PyTorch CUDA index, the Docker image, the Jetson path, and what each extra pulls in.

    [:octicons-arrow-right-24: Install](installation.md)

-   :material-rocket-launch:{ .lg .middle } **Quick Start**

    ---

    Run the bundled demo, hit the API, make a semantic and a spatial query, record and replay a session.

    [:octicons-arrow-right-24: First query](quick-start.md)

-   :material-tune:{ .lg .middle } **Configuration**

    ---

    One YAML file, `rtsm config explain | show | validate`, the tuning controls that matter and the ones that do not.

    [:octicons-arrow-right-24: Configure](configuration.md)

</div>

!!! tip "No camera?"
    `rtsm demo` replays a clip that ships inside the package, and `rtsm eval <bag>` runs the whole pipeline on a public ROS bag. Neither needs hardware beyond a CUDA GPU.
