# Documentation Index

This folder contains detailed technical documentation for the SMS Smishing Detection project. The project README provides a quickstart; these docs go deeper.

| Document | Description |
|----------|-------------|
| [architecture.md](architecture.md) | System architecture, data flow, and module dependency graph |
| [data-pipeline.md](data-pipeline.md) | Dataset provenance, preprocessing steps, duplicate/overlap handling |
| [model-training.md](model-training.md) | Experimental design, model configs, training loop, best model selection |
| [statistical-analysis.md](statistical-analysis.md) | R analysis workflow, hypothesis tests, effect sizes, interpretation |
| [demo-app.md](demo-app.md) | FastAPI application, endpoints, Docker deployment, prediction flow |
| [outputs-reference.md](outputs-reference.md) | Complete catalog of generated files with schemas and descriptions |
| [ios-app-proposal.md](ios-app-proposal.md) | Feasibility study and proposal for an iOS SMS filtering extension |

## Keeping Docs Current

When making changes to the codebase, update the relevant documentation file. If a change spans multiple areas (e.g., adding a new model affects training, evaluation, and the app), update all affected docs.
