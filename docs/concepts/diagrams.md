# System Diagrams

Interactive architectural diagrams for the JiT-TFG project.

## UnifiedSampler Architecture

Overview of the UnifiedSampler class design and model dispatch.

![UnifiedSampler Architecture](../diagrams/unified_sampler_architecture.drawio)

## Time Conventions

Comparison of timestep conventions between Flow Matching (JiT, SiT, PixelFlow) and DDPM (DiT).

![Time Conventions](../diagrams/time_conventions.drawio)

## TFG Algorithm Flow

The TFG algorithm is split into three focused diagrams for clarity.

### Overview

High-level flow showing the recurrence loop with variance and mean guidance.

![TFG Overview](../diagrams/tfg_algorithm_flow.drawio?page-index=0)

### Variance Guidance Detail

Detailed steps for computing delta_t (gradient with respect to z).

![Variance Guidance](../diagrams/tfg_algorithm_flow.drawio?page-index=1)

### Mean Guidance Detail

Detailed steps for computing delta_0 via gradient ascent on x0.

![Mean Guidance](../diagrams/tfg_algorithm_flow.drawio?page-index=2)

## Complete Pipeline

End-to-end generation pipeline from user API to final image output.

![Complete Pipeline](../diagrams/complete_pipeline.drawio)

!!! tip "Interactive Features"
    Hover over diagrams to access zoom controls. Click and drag to pan within the diagram.
