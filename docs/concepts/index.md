# Concepts

What RTSM does to a frame, and what it keeps. Three pages, each short enough to read before opening the code.

<div class="grid cards" markdown>

-   :material-sitemap:{ .lg .middle } **Architecture**

    ---

    The path of one frame, top to bottom: sources, the ingest front-end, the gate, perception, association, memory, outputs.

    [:octicons-arrow-right-24: Architecture](architecture.md)

-   :material-eye-outline:{ .lg .middle } **Perception Pipeline**

    ---

    Segmentation backends, mask heuristics, top-K selection, CLIP encoding, vocabulary classification, and what each stage costs.

    [:octicons-arrow-right-24: Pipeline](perception-pipeline.md)

-   :material-brain:{ .lg .middle } **Memory Model**

    ---

    Proto and confirmed objects, view bins, the embedding gallery, label scores, promotion, and the long-term index.

    [:octicons-arrow-right-24: Memory](memory-model.md)

</div>
