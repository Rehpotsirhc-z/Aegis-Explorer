---
title: Filtering settings
weight: 2
description: Choose an age preset or custom categories, and set how confident Aegis Explorer must be before blocking.
---

Filtering is configured on the **Blocking** and **Management** tabs of the settings page.

## Categories

Aegis Explorer can filter five kinds of content. Each one applies to both images and text.

{{< category-list >}}

## Age presets

The **Blocking** tab offers three presets and a **Custom** option. Click the arrow next to a preset to see exactly what it blocks.

{{< preset-table >}}

Choosing a preset replaces your current category choices with that preset's.

### Custom

Select **Custom** and expand it to see a checkbox for each category. Check the categories you want blocked. Changes apply to pages loaded after you make them.

## Detection Confidence

On the **Management** tab, **Detection Confidence** sets how sure Aegis Explorer must be before it blocks something. It's a percentage from 0 to 100, and the default is 50%.

{{< screenshot src="/images/settings-management.png" alt="The Management tab, with a Change Password button and the Detection Confidence field set to 50%." width="1920" height="780" >}}

- **Higher values** block only content the models are very sure about. You'll see fewer false alarms, but some harmful content may get through.
- **Lower values** block more aggressively. More harmful content is caught, but more harmless content may be hidden too.

The settings page suggests 30–50% for schools. For most families, the default is a good place to start: if you notice harmless pictures going missing, raise it a little; if things are getting through, lower it.

## Changing your password

On the **Management** tab, click **Change Password**, enter your old password and the new one twice, and click **Submit**. You'll be signed out and asked to sign in with the new password.
