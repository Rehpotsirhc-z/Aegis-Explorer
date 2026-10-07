---
title: Troubleshooting
weight: 4
description: Fixes for common problems with Aegis Explorer, and its known limitations.
---

## Common problems

### A harmless image was hidden

Models sometimes get it wrong. If it happens often, raise **Detection Confidence** on the **Management** tab a little (for example, from 50% to 60%). See [Detection Confidence](../filtering/#detection-confidence).

### Something harmful wasn't blocked

Check that the right category is turned on for your preset on the **Blocking** tab. If it is, try lowering **Detection Confidence**. Reloading the page can also help if it was still loading when content appeared.

### Images take a moment to appear

That's expected: images are held back until they've been checked. If an image can't be checked at all, for example because our servers can't be reached, it's shown after a short wait rather than leaving the page broken.

### Harmless text is hidden

If harmless text is being hidden often, raise **Detection Confidence**.

### Filtering isn't happening in Incognito windows

Chrome turns extensions off in Incognito by default. Go to `chrome://extensions`, click **Details** under Aegis Explorer and turn on **Allow in Incognito**.

### I forgot the admin password

There's no recovery option. Remove the extension from `chrome://extensions` and install it again from the [Chrome Web Store](https://chromewebstore.google.com/detail/aaepaimdjblkleipgbkgmokokdmdhpob). This resets all settings and logs.

## Known limitations

- **Chrome only.** Aegis Explorer is a Chrome extension and doesn't filter other browsers or apps on the same computer.
- **Pages, not sites.** It filters images and text on the page. It doesn't block whole websites, and it doesn't filter video or audio.
- **Embedded content.** Content inside embedded frames, such as some ads and video players, may not be checked.
- **It can be removed.** Anyone with access to Chrome's extensions page can turn it off unless the browser is managed by an administrator. The [Active Times](../statistics/#active-times) graph shows when it wasn't running.

## Still stuck?

Email us at [contact@aegisexplorer.org](mailto:contact@aegisexplorer.org) with what you were doing and what you expected to happen.
