# VTG experiment history policy

- This private repository records VTG experiments, not Codex chat transcripts. Never export sessions, conversation catalogs, raw chats or general operation logs.
- Group related research questions. Keep full benchmarks, pilots, input-order comparisons, attention measurements, sampling interventions and failure analyses distinct.
- Each run must link purpose, model/revision, dataset/cohort/sample count, settings, changes from parent runs, quantitative results, outputs, limitations, unfinished work and associated code.
- Preserve exact current code bytes with SHA-256. Do not claim current code is the historical executing revision without provenance. Keep model weights, datasets, raw arrays, frame caches, credentials and unrelated topics out.
- Distinguish source modification time from observation time; original report claims from direct verification; in-progress from completed; deleted/retired results from current evidence. Do not rerun experiments to document them.
- Important enduring workflow policy belongs in AGENTS.md; routine troubleshooting and collector administration do not belong in experiment history.
- Read EXPERIMENTS.md, EXPERIMENTS.json and linked reports at latest main. CURRENT.md/STATUS.json are experiment pointers/status only. Earlier conversation archives remain recoverable from prior Git commits.
