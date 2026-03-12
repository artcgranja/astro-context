---
title: Concepts
description: Core ideas and design philosophy behind astro-context.
---

# Concepts

Understanding the **why** behind astro-context helps you use the library
effectively and extend it with confidence. These pages cover the foundational
ideas, the system architecture, and the contracts that make everything
pluggable.

<div class="grid cards" markdown>

-   :material-lightbulb-on:{ .lg .middle } **Context Engineering**

    ---

    The philosophy of treating context as a first-class engineering problem --
    why "just stuff it in the prompt" doesn't scale.

    [:octicons-arrow-right-24: Read more](context-engineering.md)

-   :material-sitemap:{ .lg .middle } **Architecture**

    ---

    The 6-stage pipeline that transforms a raw query into a fully assembled,
    token-budgeted context window.

    [:octicons-arrow-right-24: Read more](architecture.md)

-   :material-puzzle:{ .lg .middle } **Protocols**

    ---

    PEP 544 structural subtyping for every extension point -- bring your own
    retriever, memory, formatter, or store without inheriting a base class.

    [:octicons-arrow-right-24: Read more](protocols.md)

-   :material-chart-donut:{ .lg .middle } **Token Budgets**

    ---

    Per-source allocation, overflow tracking, and priority-based eviction to
    keep your context window within model limits.

    [:octicons-arrow-right-24: Read more](token-budgets.md)

</div>
