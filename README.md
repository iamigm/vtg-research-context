# Video Temporal Grounding

Private VTG research context. Read CURRENT.md and STATUS.json at the latest main commit. HISTORY.md preserves user messages and final Codex reports. BASELINE.md is a dated initial summary.

Triggers: new message starts an activity period; each additional message resets a 30-minute inactivity timer. Export only after 30 minutes with no new user message and the Codex turn has ended. Independent fallback checks: 09:00, 12:00, 15:00, 18:00, 21:00 and 00:00 Asia/Seoul.

The watcher is event-driven using Linux inotify on the exact VTG transcript. It does not poll the transcript every minute. The daemon must run on the web IDE server; it survives browser/Mac disconnection, but requires restart after a server/container reboot.
