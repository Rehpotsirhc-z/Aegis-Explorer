---
title: Privacy Policy
description: What information the Aegis Explorer extension and website handle, where it goes, and what we keep.
toc: true
---

**Effective date:** October 7, 2026

This policy explains what information the Aegis Explorer Chrome extension (the "extension") and the aegisexplorer.org website handle. Both are operated by Aegis Technologies LLC ("we", "us").

{{< callout type="info" >}}
  **The short version.** There's no account and no sign-in. Your settings and logs stay on your computer. To decide what to block, the extension sends image addresses and short pieces of page text to our server, which checks them and doesn't save them. Text is checked using OpenAI's API. We don't sell data or show ads.
{{< /callout >}}

## Information stored on your computer

The extension keeps the following in Chrome's local storage for extensions, on the computer where it's installed:

- **Your admin password and settings**, including the selected preset, the categories you block, and Detection Confidence.
- **A log of blocked content**: the time each image was blocked and its category. It doesn't include the image, the website, or the page address. Entries older than 30 days are deleted.
- **An activity log**: the times the extension was running, used for the Active Times graph.

This information isn't sent to us. It's deleted when the extension is removed from Chrome.

The admin password is stored as entered, without encryption, so choose one you don't use anywhere else.

## Information sent to our servers

To decide whether something should be hidden, the extension sends the following to our server at aegisexplorer.org:

- **Image addresses.** The web address (URL) of each image that still needs checking after the in-browser check. Our server downloads the image from that address to check it. Images embedded directly in a page as data are sent as the image data itself. Images flagged by the in-browser model aren't sent.
- **Page text.** Short pieces of visible text from the page, split into sentences.

The extension doesn't send the address of the page you're on, your browsing history, cookies, form entries, or any account or contact details. As with any internet connection, our server sees the IP address the request comes from.

### How we handle it

- Images are checked in memory and not saved.
- Text results are kept in a temporary in-memory cache so the same sentence isn't sent for checking twice. The cache is cleared whenever the server restarts and isn't written to a database.
- We don't use the content we check to build profiles of users, and because we don't keep it, it isn't used to train our models.

## OpenAI

Our server sends the page text it receives to OpenAI's API, which classifies it. OpenAI receives the text only: not your IP address or anything else about you. Under [OpenAI's API data policy](https://developers.openai.com/api/docs/guides/your-data), data sent through the API isn't used to train OpenAI's models, and is kept for up to 30 days for abuse monitoring.

We don't share information with anyone else.

## What we don't do

- We don't require an account, name, email address, or age to use the extension.
- We don't include analytics, tracking or advertising in the extension.
- We don't sell, rent or trade information.

## Children's privacy

Aegis Explorer is designed to be installed and configured by a parent, guardian or school. The extension doesn't ask for or collect personal information from children, such as names, ages or contact details. The page content it sends for checking is used only to decide whether to block it, and isn't kept.

## This website

aegisexplorer.org doesn't use cookies, analytics or advertising trackers. If you choose a light or dark theme, that choice is saved in your browser's local storage and never sent to us. Like most websites, our web server may keep standard access logs, such as IP address, pages requested and browser type, for security and troubleshooting.

## Emails you send us

If you email us at [contact@aegisexplorer.org](mailto:contact@aegisexplorer.org), we'll use your email address and message only to reply to you.

## Changes to this policy

If our practices change, we'll update this page and the effective date at the top.

## Contact

Aegis Technologies LLC<br>
[contact@aegisexplorer.org](mailto:contact@aegisexplorer.org)
