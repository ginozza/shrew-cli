# Architecture: shrew-cli

`shrew-cli` is the command-line interface for the Shrew ecosystem. It allows users to interact with `.sw` files directly, providing tools for execution, validation, benchmarking, and debugging.

## Core Concepts

- **Runner**: Executes `.sw` models akin to an interpreter (`shrew run model.sw`).
- **Validator**: Checks `.sw` files for syntax errors and valid graph topology without executing them.
- **Benchmarker**: Runs performance tests on models to measure throughput and latency.
- **Inspector**: Dumps the structure of a model (shapes, layers, params) to stdout or JSON.

## File Structure

| File | Description | Lines of Code |
| :--- | :--- | :--- |
| `main.rs` | The entry point for the executable. Parses command-line arguments (using `clap`) and dispatches to the appropriate handler (run, bench, check, info). | 384 |
