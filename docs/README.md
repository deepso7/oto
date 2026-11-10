# Oto docs

| Doc | What it covers | Changes |
|---|---|---|
| [Product](product.md) | What Oto does, who it is for, the targets we hold it to | Rarely |
| [Architecture](architecture.md) | Process and thread layout, the virtual microphone on each OS, the signal chain, the UI | When the design changes |
| [Models](models.md) | The noise-suppression models we evaluated and why | When research changes |
| [Roadmap](roadmap.md) | Phases, risks and settled decisions | Each phase |
| [Status](status.md) | What exists in the code today, with measurements | Every merged change that alters behaviour |
| [Research sources](research.md) | Links, and what the research could not verify | Rarely |

## Plans

A plan describes one piece of work before it is built. Plans live in [`plans/`](plans/), are numbered in the order they were written, and are never renumbered.

| # | Plan | Status |
|---|---|---|
| 0001 | [DeepFilterNet3 as the default model](plans/0001-deepfilternet3-default.md) | Accepted |

Each plan starts with a status line: **Proposed** (written, not agreed), **Accepted** (agreed, not started), **In progress**, **Done** or **Dropped**. When a plan is done, its outcome moves into [Status](status.md) and, if it changed the design, into [Architecture](architecture.md) or [Models](models.md). The plan file stays as the record of why.

To add one, copy the headings of the latest plan: Goal, Background, Decisions, Work, Done when, Risks, Out of scope.
