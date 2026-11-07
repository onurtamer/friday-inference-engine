# ADR 0001: Product target and platform profiles

- Status: accepted; items 1 and 5 amended by ADR 0003
- Date: 2026-10-27

## Context

The intended primary host is an ARM64 board with large unified memory
(Radxa Orion O6, 64 GB RAM). FRIDAY is expected to have both a Service
distribution and a later OS distribution.

## Decision

1. Treat Orion O6 ARM64 as the primary product target.
   *(Amended by ADR 0003: NVIDIA CUDA and ARM64 CPU are both first-class
   v1 backends.)*
2. Provide two deployment products over the same Core and Engine contracts:
   FRIDAY Service and FRIDAY OS.
3. Begin FRIDAY OS from supported vendor Linux kernel and board support
   packages; do not start a custom kernel.
4. Keep host ISA, accelerator provider, and memory topology as independent
   capability dimensions. User-facing installer profiles may bundle common
   combinations such as ARM64 and x86-64/CUDA.
5. Preserve the existing CUDA engine as a read-only reference until a
   separate reuse review is approved.
   *(Amended by ADR 0003: the existing engine is the v1 code base.)*

## Consequences

- CPU ARM64 inference is a runtime path to validate on O6.
- Vendor NPU implementations remain optional and separately gated.
- Datasheet figures are planning inputs, not throughput guarantees.
- No O6 benchmark claim is made before real-board measurement.
