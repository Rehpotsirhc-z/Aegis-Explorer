---
title: Aegis Explorer
layout: hextra-home
description: >-
  Aegis Explorer protects children online by detecting and blocking
  potentially harmful images and text using customizable filters.
aliases:
  - /demo/
---

<section class="ae-hero">
<h1 class="ae-hero-title">Aegis Explorer</h1>
<p class="ae-hero-lead">AI-Powered Real-Time Content Filtering for Child Safety</p>
<div class="ae-actions">
<a class="ae-btn ae-btn-primary" href="https://chromewebstore.google.com/detail/aaepaimdjblkleipgbkgmokokdmdhpob" target="_blank" rel="noreferrer">Add to Chrome</a>
<a class="ae-btn ae-btn-secondary" href="#how-it-works">How it works</a>
</div>
</section>

<div class="ae-gap-lg"></div>

{{< cards >}}
  {{< card
      title="Real-Time Content Filtering"
      image="real-time_content_filtering.png"
      subtitle="AI scans every page instantly to detect and block harmful images and text before they appear."
  >}}

  {{< card
      title="Powerful Usage Statistics"
      image="powerful_usage_statistics_1.png"
      subtitle="Activity logs help you understand what was filtered and when the extension was active."
  >}}

  {{< card
      title="Extensible Presets"
      image="extensible_presets.png"
      subtitle="Fine-tune filtering behavior with customizable categories, sensitivity levels, and safety profiles."
  >}}
{{< /cards >}}

<section class="ae-section" id="problem">
<h2 class="ae-h2">The Problem</h2>
</section>

{{< hextra/feature-grid >}}
  {{< hextra/feature-card
      title="97% Online Access"
      subtitle="U.S. kids aged 3–18 have internet access at home.¹"
      style="background: radial-gradient(ellipse at 50% 80%, rgba(255,180,120,0.15), rgba(255,180,120,0));"
  >}}
  {{< hextra/feature-card
      title="Adult-Oriented Platforms"
      subtitle="Most sites aren't designed with children in mind."
      style="background: radial-gradient(ellipse at 50% 80%, rgba(255,150,100,0.15), rgba(255,150,100,0));"
  >}}
  {{< hextra/feature-card
      title="Exposure to Harm"
      subtitle="Violence, explicit content, unsafe interactions."
      style="background: radial-gradient(ellipse at 50% 80%, rgba(255,120,100,0.15), rgba(255,120,100,0));"
  >}}
{{< /hextra/feature-grid >}}

<div class="ae-gap-sm"></div>

{{< hextra/feature-grid cols="2" >}}
  {{< hextra/feature-card
      title="59% Abusive Interactions"
      subtitle="Teens report harmful or abusive online experiences.²"
      style="background: radial-gradient(ellipse at 50% 80%, rgba(255,90,80,0.15), rgba(255,90,80,0));"
  >}}
  {{< hextra/feature-card
      title="46% Cyberbullying"
      subtitle="Nearly half of U.S. teens have experienced cyberbullying.³"
      style="background: radial-gradient(ellipse at 50% 80%, rgba(255,60,60,0.15), rgba(255,60,60,0));"
  >}}
{{< /hextra/feature-grid >}}

<ol class="ae-sources">
<li>National Center for Education Statistics, <a href="https://nces.ed.gov/programs/coe/indicator/cch" target="_blank" rel="noreferrer">Children's Internet Access at Home</a>, Condition of Education (2021 data).</li>
<li>Pew Research Center, <a href="https://www.pewresearch.org/internet/2018/09/27/a-majority-of-teens-have-experienced-some-form-of-cyberbullying/" target="_blank" rel="noreferrer">A Majority of Teens Have Experienced Some Form of Cyberbullying</a>, 2018.</li>
<li>Pew Research Center, <a href="https://www.pewresearch.org/internet/2022/12/15/teens-and-cyberbullying-2022/" target="_blank" rel="noreferrer">Teens and Cyberbullying 2022</a>, 2022.</li>
</ol>

<section class="ae-section" id="how-it-works">
<h2 class="ae-h2">How it works</h2>
<ol class="ae-steps">
<li>
<h3>Images are hidden until checked</h3>
<p>When a page loads, its images are hidden until they have been classified. Text on the page is collected as it appears, including content that loads later.</p>
</li>
<li>
<h3>Images and text are classified</h3>
<p>Images are first checked by a local model in the browser. Anything that still needs checking is sent to the Aegis Explorer server to be classified.</p>
</li>
<li>
<h3>Blocked content stays hidden</h3>
<p>Images and text that match the selected categories stay hidden. Everything else is shown.</p>
</li>
</ol>
</section>

<section class="ae-section" id="presets">
<h2 class="ae-h2">Age presets</h2>
{{< preset-table >}}
</section>

<section class="ae-section" id="controls">
<h2 class="ae-h2">Parent controls</h2>
{{< screenshot src="/images/statistics-activity.png" alt="The Statistics tab of the Aegis Explorer settings page: a graph of blocked items over the day, and an Active Times graph showing two browsing sessions, in the morning and the afternoon." width="1506" height="1081" >}}
</section>

<section class="ae-section" id="principles">
<h2 class="ae-h2">Why Aegis Explorer</h2>
</section>

{{< hextra/feature-grid >}}
  {{< hextra/feature-card
      title="Designed for Families"
      subtitle="Keeps harmful images and text away using fast AI models."
      style="background: radial-gradient(ellipse at 50% 80%,rgba(81,175,239,0.15),hsla(0,0%,100%,0));"
  >}}
  {{< hextra/feature-card
      title="Private by Default"
      subtitle="Processes only what’s necessary---nothing is stored, and everything else stays local."
      style="background: radial-gradient(ellipse at 50% 80%,rgba(198,120,221,0.15),hsla(0,0%,100%,0));"
  >}}
  {{< hextra/feature-card
      title="Clear, Simple Oversight"
      subtitle="See when content was blocked and when the extension was active."
      style="background: radial-gradient(ellipse at 50% 80%,rgba(152,190,101,0.15),hsla(0,0%,100%,0));"
  >}}
{{< /hextra/feature-grid >}}

<section class="ae-section" id="demo">
<h2 class="ae-h2">Demo</h2>
<div class="ae-videos">
<figure>
<video controls preload="none" playsinline poster="/images/demo2-poster.jpg" width="1920" height="1080"><source src="/demo2.mp4" type="video/mp4"></video>
</figure>
<figure>
<video controls preload="none" playsinline poster="/images/demo1-poster.jpg" width="1920" height="1080"><source src="/demo1.mp4" type="video/mp4"></video>
</figure>
</div>
</section>

<section class="ae-section" id="faq">
<h2 class="ae-h2">FAQ</h2>
</section>

<div class="ae-faq">

{{% details title="Can my child turn it off?" closed="true" %}}
Changing any setting requires the admin password you create during setup. Like any browser extension, it can be removed from Chrome's extensions page unless the browser is managed, but the Active Times graph on the Statistics tab shows when the extension wasn't running.
{{% /details %}}

{{% details title="Does it block whole websites?" closed="true" %}}
No. It hides individual images and text and leaves the rest of the page alone.
{{% /details %}}

{{% details title="Will it slow down browsing?" closed="true" %}}
Images can appear a moment later than usual because they're hidden until they've been checked.
{{% /details %}}

{{% details title="Is it ever wrong?" closed="true" %}}
Yes. No filter is perfect: sometimes harmless content is hidden, and sometimes harmful content isn't caught. Detection Confidence adjusts that balance.
{{% /details %}}

{{% details title="What happens if your servers are down?" closed="true" %}}
Pages keep working. Images that can't be checked are shown rather than left hidden, and the in-browser explicit-image check keeps running.
{{% /details %}}

{{% details title="Which browsers does it work in?" closed="true" %}}
Aegis Explorer is a Google Chrome extension for desktop, installed from the Chrome Web Store.
{{% /details %}}

</div>

<section class="ae-cta">
<h2>Install Aegis Explorer</h2>
<div class="ae-actions">
<a class="ae-btn ae-btn-primary" href="https://chromewebstore.google.com/detail/aaepaimdjblkleipgbkgmokokdmdhpob" target="_blank" rel="noreferrer">Add to Chrome</a>
<a class="ae-btn ae-btn-secondary" href="/docs/getting-started/">Setup guide</a>
</div>
</section>
