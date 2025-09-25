# model-bringup

Bring-up lab for running **GLM-4.5 (MoE)** on **RDU** inside a **Docker** environment.
This environment is distributed as a one-off build for **SoftBank**.

The project provides:

* A thin compatibility layer to register GLM-4.5 with a Transformers-like interface.
* Unit tests to keep changes safe and debuggable.

---

## Prerequisites

* **Docker** (for the SoftBank one-off image).

---

## Repository Layout

```
model-bringup/
├─ README.md
├─ configs
│  ├─ glm4-5.json   
│  ├─ o0.yaml     
├─ prompt.jsonl
├─ text_generation_compile.py
├─ unit_test.py

```
---

## Quick Start

## Run inside Docker (preferred)

> The Docker environment is provided as a one-off build for SoftBank.

## Unit Tests

Run all tests:

```bash
python3 -m pytest unit_test.py --forked
```


