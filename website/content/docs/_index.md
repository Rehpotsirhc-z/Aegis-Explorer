---
title: Documentation
linkTitle: Docs
description: How to install, set up and use the Aegis Explorer Chrome extension.
---

Aegis Explorer is a Chrome extension that hides harmful images and text on the pages your child visits. You choose which kinds of content to filter, and settings are protected by a password only you know.

{{< cards >}}
  {{< card link="getting-started/" title="Getting started" subtitle="Install the extension and create your admin password." >}}
  {{< card link="filtering/" title="Filtering settings" subtitle="Age presets, custom categories and detection sensitivity." >}}
  {{< card link="statistics/" title="Statistics" subtitle="Read the daily log of blocked content and activity." >}}
  {{< card link="troubleshooting/" title="Troubleshooting" subtitle="Fixes for common problems, and known limitations." >}}
{{< /cards >}}

## How filtering works

When a page loads, Aegis Explorer:

1. **Holds back the page's images** until they've been checked. Text is collected as it appears on the page, including content that loads later as you scroll.
2. **Checks images on the computer first.** An explicit-image model runs inside the browser. Images it flags are hidden right away and never leave the computer.
3. **Sends the rest to be classified.** Remaining images are checked by our image model for explicit content, drugs, gambling, games and profanity. Short pieces of text are checked by a language model for the same categories.
4. **Hides what matches your settings.** Blocked images and text stay hidden. Everything else is shown as normal.

Only the categories you've turned on are blocked, and only when the model's confidence is above your [Detection Confidence](filtering/#detection-confidence) setting.

## Design philosophy

Aegis Explorer is purposely designed to strike a balance between online safety and independence:

- No filter is perfect; conversation and trust matter
- Not meant to be unbreakable, but **practical**
- Helps parents guide safe internet habits without extreme restrictions

The goal is mitigation, not total lockdown.

For what data is sent and what's kept, see the [privacy policy](/privacy/).
