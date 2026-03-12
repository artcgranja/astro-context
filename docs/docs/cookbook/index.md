---
title: Cookbook
description: Practical recipes and patterns for building with anchor
---

# Cookbook

Ready-to-use recipes that go beyond the basics. Each recipe is a self-contained
guide with working code you can copy into your project.

---

<div class="grid cards" markdown>

-   :material-chat-processing:{ .lg .middle } **Chatbot with Memory**

    ---

    Build a conversational chatbot with sliding-window memory, automatic eviction,
    and persistent facts.

    [:octicons-arrow-right-24: Chatbot with Memory](chatbot-with-memory.md)

-   :material-database-search:{ .lg .middle } **RAG Pipeline**

    ---

    Complete Retrieval-Augmented Generation with dense retrieval, hybrid search,
    reranking, and provider-formatted output.

    [:octicons-arrow-right-24: RAG Pipeline](rag-pipeline.md)

-   :material-file-document-multiple:{ .lg .middle } **Document Ingestion**

    ---

    Ingest raw text and files, chunk with different strategies, enrich metadata,
    and index the results into a retriever.

    [:octicons-arrow-right-24: Document Ingestion](document-ingestion.md)

-   :material-code-braces:{ .lg .middle } **Custom Retriever**

    ---

    Implement the `Retriever` protocol to create your own retriever and integrate
    it with `ContextPipeline` and `HybridRetriever`.

    [:octicons-arrow-right-24: Custom Retriever](custom-retriever.md)

-   :material-chart-bar:{ .lg .middle } **Evaluation Workflow**

    ---

    Evaluate retrieval quality with standard metrics and compare configurations
    using A/B testing with statistical significance.

    [:octicons-arrow-right-24: Evaluation Workflow](evaluation-workflow.md)

-   :material-robot:{ .lg .middle } **Agent with Tools**

    ---

    Build an agent with custom tools, skills, and memory using the `@tool`
    decorator and `SkillRegistry`.

    [:octicons-arrow-right-24: Agent with Tools](agent-with-tools.md)

-   :material-factory:{ .lg .middle } **Production Patterns**

    ---

    Battle-tested patterns for deploying anchor in production: error handling,
    observability, performance tuning, and testing strategies.

    [:octicons-arrow-right-24: Production Patterns](production-patterns.md)

</div>
