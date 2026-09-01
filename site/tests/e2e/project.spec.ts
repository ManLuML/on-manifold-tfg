import AxeBuilder from '@axe-core/playwright';
import { expect, test } from '@playwright/test';

const legacyAssets = [
  'favicon.svg', 'og_card.png',
  'rho_fid_vs_child_fid.png', 'rho_fid_vs_child_fid.webp',
  'rho_fid_vs_validity.png', 'rho_fid_vs_validity.webp',
  'style_gram_vs_validity.png', 'style_gram_vs_validity.webp',
  'teaser.png', 'teaser.webp',
  'vis_off_dit.jpg', 'vis_off_dit.webp',
  'vis_off_jit.jpg', 'vis_off_jit.webp',
  'vis_on_dit.jpg', 'vis_on_dit.webp',
  'vis_on_jit.jpg', 'vis_on_jit.webp',
];

test('project metadata, claims, resources, and text-only affiliations are complete', async ({ page }) => {
  const externalRuntimeRequests: string[] = [];
  page.on('request', (request) => {
    const url = new URL(request.url());
    if (url.origin !== 'http://127.0.0.1:4324') externalRuntimeRequests.push(request.url());
  });

  await page.goto('./');
  await expect(page).toHaveTitle(/Not All Prediction Targets.*ECCV 2026/);
  await expect(page.locator('link[rel="canonical"]')).toHaveAttribute('href', 'https://manluml.github.io/on-manifold-tfg/');
  await expect(page.locator('meta[name="description"]')).toHaveAttribute('content', /fails gracefully or catastrophically/);
  await expect(page.locator('meta[property="og:image"]')).toHaveAttribute('content', 'https://manluml.github.io/on-manifold-tfg/og/project-card.png');
  await expect(page.locator('meta[name="twitter:card"]')).toHaveAttribute('content', 'summary_large_image');
  await expect(page.locator('meta[name="twitter:image"]')).toHaveAttribute('content', 'https://manluml.github.io/on-manifold-tfg/og/project-card.png');
  await expect(page.locator('meta[name="citation_author"]')).toHaveCount(2);
  await expect(page.locator('meta[name="citation_pdf_url"]')).toHaveAttribute('content', 'https://arxiv.org/pdf/2607.00647');
  await expect(page.locator('meta[name="citation_version"]')).toHaveCount(0);
  const scholarlyArticle = JSON.parse(await page.locator('script[type="application/ld+json"]').textContent() ?? '{}');
  expect(scholarlyArticle['@type']).toBe('ScholarlyArticle');
  expect(scholarlyArticle.datePublished).toBeUndefined();
  expect(scholarlyArticle.associatedMedia).toMatchObject({ name: 'Public arXiv preprint', datePublished: '2026-07-01' });
  await expect(page.getByRole('link', { name: /^Paper/ }).first()).toHaveAttribute('href', 'https://arxiv.org/pdf/2607.00647');
  await expect(page.getByRole('link', { name: /^Code/ }).first()).toHaveAttribute('href', 'https://github.com/ManLuML/on-manifold-tfg');
  await expect(page.getByRole('link', { name: /^Data/ }).first()).toHaveAttribute('href', '#resources');
  await expect(page.getByRole('link', { name: /^Cite/ }).first()).toHaveAttribute('href', '#bibtex');
  await expect(page.getByRole('link', { name: /View the ECCV record/ })).toHaveAttribute('href', 'https://eccv.ecva.net/virtual/2026/poster/4934');
  await expect(page.locator('.affiliations li').nth(0)).toContainText('Maum AI');
  await expect(page.locator('.affiliations li').nth(1)).toContainText('Seoul National University of Science and Technology');
  await expect(page.getByText(/Republic of Korea/)).toHaveCount(0);
  await expect(page.locator('img[src*="affiliations"], img[src*="maum"], img[src*="seoultech"]')).toHaveCount(0);
  await expect(page.getByText('ECCV 2026', { exact: true }).first()).toBeVisible();
  await expect(page.getByRole('list', { name: 'Prediction targets ranked from most to least robust for training-free guidance' })).toContainText('x-prediction');
  await expect(page.getByRole('heading', { name: 'ECCV 2026 · accepted' })).toBeVisible();
  await expect(page.locator('#abstract, #method, #results, #bibtex')).toHaveCount(4);
  await expect(page.getByText('93.3%', { exact: true }).first()).toBeVisible();
  await expect(page.getByText('21.5%', { exact: true }).first()).toBeVisible();
  await expect(page.getByText('0.5%', { exact: true }).first()).toBeVisible();
  await expect(page.getByText('9,152 generated images')).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Can guidance hit the target and keep the image realistic?' })).toBeVisible();
  await expect(page.getByText('Scope and limitations')).toHaveCount(0);
  await expect(page.getByText(/Camera-ready/i)).toHaveCount(0);
  expect(externalRuntimeRequests).toEqual([]);
});

test('desktop hero keeps the conclusion-sized title readable at a glance', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('./');
  const lineCount = await page.locator('h1').evaluate((heading) => {
    const range = document.createRange();
    range.selectNodeContents(heading);
    return new Set([...range.getClientRects()].filter((rect) => rect.width > 1).map((rect) => Math.round(rect.top))).size;
  });
  expect(lineCount).toBeLessThanOrEqual(2);
});

test('failure and mechanism stories are scroll-native and complete', async ({ page }) => {
  await page.goto('./#results');
  await expect(page.getByRole('tab')).toHaveCount(0);
  await expect(page.getByRole('slider')).toHaveCount(0);
  await expect(page.locator('.failure-band')).toHaveCount(3);
  for (const band of await page.locator('.failure-band').all()) await expect(band).toBeVisible();
  await expect(page.locator('.recovery-rows article')).toHaveCount(3);
  await expect(page.getByText('That endpoint does not rank robustness.')).toBeVisible();
});

test('formula and manifold figure visual baselines remain balanced', async ({ page }) => {
  await page.goto('./#method');
  await expect(page.locator('.recovery-rows')).toHaveScreenshot('project-formulas-desktop.png');
  await expect(page.locator('.manifold-evidence-grid')).toHaveScreenshot('project-manifold-figure-desktop.png');
});

test('static fallback is complete without JavaScript', async ({ browser }) => {
  const context = await browser.newContext({ javaScriptEnabled: false });
  const page = await context.newPage();
  await page.goto('http://127.0.0.1:4324/on-manifold-tfg/');
  await expect(page.locator('[role="tabpanel"], [role="tab"], input[type="range"]')).toHaveCount(0);
  await expect(page.locator('.failure-band')).toHaveCount(3);
  await expect(page.locator('.recovery-rows article')).toHaveCount(3);
  await expect(page.locator('.manifold-evidence-grid figure')).toHaveCount(2);
  await expect(page.locator('#bibtex-text')).toContainText('@inproceedings');
  await context.close();
});

test('all rendered project images and same-origin resources load', async ({ page }) => {
  const failed: string[] = [];
  page.on('response', (response) => {
    const url = new URL(response.url());
    if (url.origin === 'http://127.0.0.1:4324' && response.status() >= 400) failed.push(`${response.status()} ${url.pathname}`);
  });
  await page.goto('./');
  const images = await page.locator('img').evaluateAll(async (elements) => {
    await Promise.all(elements.map(async (element) => {
      const image = element as HTMLImageElement;
      image.loading = 'eager';
      try { await image.decode(); } catch { /* Report through dimensions below. */ }
    }));
    return elements.map((element) => {
      const image = element as HTMLImageElement;
      return { src: image.currentSrc || image.src, width: image.naturalWidth, height: image.naturalHeight };
    });
  });
  expect(images.filter((image) => image.width === 0 || image.height === 0)).toEqual([]);
  expect(failed).toEqual([]);
});

test('copy feedback is visible and announced', async ({ page }) => {
  await page.goto('./#bibtex');
  await expect(page.locator('#bibtex-text')).toContainText('Computer Vision -- ECCV 2026');
  await expect(page.locator('#bibtex-text')).toContainText('note      = {To appear}');
  await expect(page.locator('#bibtex-text')).not.toContainText(/doi|pages|volume/i);
  await page.getByRole('button', { name: 'Copy BibTeX' }).click();
  await expect(page.locator('#copy-status')).toContainText(/copied|selected/i);
});

test('accessibility, English-only output, and reduced motion pass', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto('./');
  const results = await new AxeBuilder({ page }).analyze();
  expect(results.violations).toEqual([]);
  const publicText = [await page.locator('body').innerText(), await page.title()].join('\n');
  expect(publicText).not.toMatch(/[\u3131-\u318E\uAC00-\uD7A3]/);
  expect(publicText).not.toMatch(/[—–]|makes that promise measurable|points in the same direction|Read the sequence, then test the cause|Broadly yes/i);
  expect(await page.locator('html').evaluate((element) => getComputedStyle(element).scrollBehavior)).toBe('auto');
});

test('320px reflow keeps text affiliations and content within the viewport', async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 568 });
  await page.goto('./');
  await expect(page.locator('.affiliations')).toBeVisible();
  const dimensions = await page.evaluate(() => ({ width: document.documentElement.clientWidth, scrollWidth: document.documentElement.scrollWidth }));
  expect(dimensions.scrollWidth).toBeLessThanOrEqual(dimensions.width);
  const titleLines = await page.locator('h1').evaluate((heading) => {
    const range = document.createRange();
    range.selectNodeContents(heading);
    return new Set([...range.getClientRects()].filter((rect) => rect.width > 1).map((rect) => Math.round(rect.top))).size;
  });
  expect(titleLines).toBeLessThanOrEqual(5);
  const compoundLines = await page.locator('h1 .nowrap').evaluate((text) => {
    const range = document.createRange();
    range.selectNodeContents(text);
    return new Set([...range.getClientRects()].map((rect) => Math.round(rect.top))).size;
  });
  expect(compoundLines).toBe(1);
  await expect(page.locator('.dimension-evidence figure')).toBeHidden();
  expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
  await expect(page).toHaveScreenshot('project-320.png');
  await expect(page.locator('.affiliations')).toHaveScreenshot('project-affiliations-320.png');
});

test('mobile failure sequence keeps individual samples legible', async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 568 });
  await page.goto('./#results');
  const image = page.locator('.failure-band.x img');
  await image.evaluate(async (element) => {
    const target = element as HTMLImageElement;
    target.loading = 'eager';
    await target.decode();
  });
  expect(await image.evaluate((element) => (element as HTMLImageElement).currentSrc)).toContain('jit-failure-mobile-640');
  await expect(page.locator('.failure-band.x picture')).toHaveScreenshot('project-failure-x-320.png');
});

test('200% text sizing reflows without clipped content', async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 568 });
  await page.goto('./');
  await page.addStyleTag({ content: 'html { font-size: 200% !important; }' });
  const clipped = await page.evaluate(() => {
    const viewport = document.documentElement.clientWidth;
    return [...document.body.querySelectorAll<HTMLElement>('*')]
      .filter((element) => {
        if (!element.getClientRects().length || element.closest('.table-scroll, pre')) return false;
        const rect = element.getBoundingClientRect();
        return rect.left < -1 || rect.right > viewport + 1;
      })
      .slice(0, 20)
      .map((element) => `${element.tagName.toLowerCase()}.${element.className}`);
  });
  expect(clipped).toEqual([]);
  expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
});

test('first-view performance budgets hold in the lab', async ({ page }) => {
  await page.addInitScript(() => {
    const metrics = { cls: 0, lcp: 0 };
    Object.defineProperty(window, '__projectMetrics', { value: metrics, writable: false });
    new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) metrics.lcp = entry.startTime;
    }).observe({ type: 'largest-contentful-paint', buffered: true });
    new PerformanceObserver((list) => {
      for (const entry of list.getEntries() as (PerformanceEntry & { hadRecentInput?: boolean; value?: number })[]) {
        if (!entry.hadRecentInput) metrics.cls += entry.value ?? 0;
      }
    }).observe({ type: 'layout-shift', buffered: true });
  });
  await page.goto('./');
  await page.waitForTimeout(700);
  const metrics = await page.evaluate(() => (window as typeof window & { __projectMetrics: { cls: number; lcp: number } }).__projectMetrics);
  console.log(`Project lab metrics: LCP=${metrics.lcp.toFixed(1)}ms CLS=${metrics.cls.toFixed(4)}`);
  expect(metrics.lcp).toBeLessThanOrEqual(2500);
  expect(metrics.cls).toBeLessThanOrEqual(0.1);
});

test('legacy asset URLs remain available', async ({ request }) => {
  for (const asset of legacyAssets) {
    const response = await request.get(`http://127.0.0.1:4324/on-manifold-tfg/static/images/${asset}`);
    expect(response.ok(), asset).toBe(true);
    expect(response.headers()['content-type']).toMatch(/^image\//);
  }
});

test('the complete ECCV presentation is published at the stable slide URL', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 720 });
  const response = await page.goto('http://127.0.0.1:4324/on-manifold-tfg/slides/');
  expect(response?.ok()).toBe(true);
  await expect(page).toHaveTitle('Not All Prediction Targets Keep Training-Free Diffusion Guidance on the Manifold');
  await expect(page.locator('.reveal .slides > section')).toHaveCount(18);
  await expect(page.locator('.reveal .slides > section').first()).toHaveAttribute('data-source-section', '01');
  await expect(page.locator('.reveal .slides > section[data-source-section="19"]')).toHaveCount(1);
  await expect(page.locator('.title-logo-maumai')).toHaveCount(1);
  await expect(page.locator('.title-logo-seoultech img')).toHaveCount(1);
  await expect(page.locator('.title-logo-eccv')).toHaveCount(1);
  await expect(page.locator('.slide-number')).toContainText('1 / 18');
  await expect(page.locator('img').first()).toHaveJSProperty('complete', true);
  const dimensions = await page.evaluate(() => ({ width: document.documentElement.clientWidth, scrollWidth: document.documentElement.scrollWidth }));
  expect(dimensions.scrollWidth).toBeLessThanOrEqual(dimensions.width);

  await page.goto('http://127.0.0.1:4324/on-manifold-tfg/slides/?preview=outcomes#/section-11');
  await expect(page.locator('.slide-number')).toContainText('12 / 18');
  const outcomeSlide = page.locator('.reveal .slides > section[data-source-section="paper-transition"]');
  await expect(outcomeSlide.locator('.guidance-sequence-header > p')).toHaveText('Guidance outcomes');
  await expect(outcomeSlide.locator('.guidance-sequence-header h2')).toHaveText('Do prediction targets decide how guidance fails?');
  expect(await outcomeSlide.locator('.guidance-sequence-header h2').evaluate((element) => getComputedStyle(element).fontSize)).toBe('48px');
  expect(await outcomeSlide.locator('.guidance-sequence-header h2').evaluate((element) => element.getClientRects().length)).toBe(1);
  expect(await outcomeSlide.locator('.guidance-sequence-stage').evaluate((element) => element.getBoundingClientRect().width)).toBeLessThan(850);
  const frames = page.locator('.guidance-sequence-frame');
  await expect(frames).toHaveCount(4);
  expect(await frames.evaluateAll((images) => images.every((image) => (image as HTMLImageElement).complete && (image as HTMLImageElement).naturalWidth === 2400 && (image as HTMLImageElement).naturalHeight === 1350))).toBe(true);
  const frameRects = await frames.evaluateAll((images) => images.map((image) => {
    const rect = image.getBoundingClientRect();
    return [rect.left, rect.top, rect.width, rect.height].map((value) => Math.round(value * 100) / 100);
  }));
  expect(new Set(frameRects.map((rect) => JSON.stringify(rect))).size).toBe(1);
  const activeFrame = () => frames.evaluateAll((images) => {
    const visible = images.filter((image) => {
      const style = getComputedStyle(image);
      return style.visibility !== 'hidden' && Number(style.opacity) > 0.95;
    });
    return visible[visible.length - 1]?.getAttribute('src');
  });
  const captions = page.locator('.guidance-sequence-caption');
  await expect(captions).toHaveCount(4);
  const visibleCaptions = () => captions.evaluateAll((items) => items.filter((item) => {
    const style = getComputedStyle(item);
    return style.visibility !== 'hidden' && Number(style.opacity) > 0.99;
  }).map((item) => item.textContent?.replace(/\s+/g, ' ').trim()));
  await page.waitForTimeout(700);
  expect(await activeFrame()).toContain('guidance-sequence-05-success.png');
  expect(await visibleCaptions()).toHaveLength(1);
  for (const [expected, captionCount] of [
    ['guidance-sequence-06-catastrophic.png', 2],
    ['guidance-sequence-07-graceful.png', 3],
    ['guidance-sequence-04-combined.png', 4],
  ] as const) {
    await page.keyboard.press('ArrowRight');
    await page.waitForTimeout(700);
    expect(await activeFrame()).toContain(expected);
    expect(await visibleCaptions()).toHaveLength(captionCount);
  }
  const finalCaptionText = (await visibleCaptions()).join(' ');
  expect(finalCaptionText).toContain('only a reddish artifact, not a recognizable parrot');
  expect(finalCaptionText).toContain('a high-quality blue macaw from the parrot parent class');
  expect(finalCaptionText).toContain('realistic under the data distribution');
  expect(await captions.evaluateAll((items) => items.every((item) => item.scrollWidth <= item.clientWidth))).toBe(true);
  expect(await captions.first().evaluate((item) => getComputedStyle(item).fontSize)).toBe('22px');
  expect(await captions.evaluateAll((items) => items.map((item) => getComputedStyle(item).borderLeftColor))).toEqual([
    'rgb(63, 125, 32)',
    'rgb(160, 0, 0)',
    'rgb(147, 51, 234)',
    'rgb(13, 71, 161)',
  ]);
  const captionGap = await page.evaluate(() => document.querySelector('.slide-number')!.getBoundingClientRect().top
    - document.querySelector('.guidance-sequence-captions')!.getBoundingClientRect().bottom);
  expect(captionGap).toBeGreaterThan(10);

  await page.goto('http://127.0.0.1:4324/on-manifold-tfg/slides/?preview=benchmark#/section-12');
  await expect(page.locator('.slide-number')).toContainText('13 / 18');
  const benchmarkSlide = page.locator('.reveal .slides > section[data-source-section="task-benchmark"]');
  await expect(benchmarkSlide).toHaveCount(1);
  await expect(benchmarkSlide.locator(':scope > section')).toHaveCount(0);
  await expect(benchmarkSlide.locator('.task-benchmark-header h2')).toHaveText('Steer within the pretrained bird domain.');
  await expect(benchmarkSlide.getByText('143 fine-grained species', { exact: true })).toBeVisible();
  await expect(benchmarkSlide.getByText('Wholly unseen domain transfer', { exact: true })).toBeVisible();
  const releaseBand = page.locator('.task-benchmark-release');
  expect(await releaseBand.evaluate((element) => Number(getComputedStyle(element).opacity))).toBeLessThan(0.01);
  await page.keyboard.press('ArrowRight');
  await page.waitForTimeout(700);
  expect(await releaseBand.evaluate((element) => Number(getComputedStyle(element).opacity))).toBeGreaterThan(0.99);
  await expect(releaseBand).toContainText('143');
  await expect(releaseBand).toContainText('30');
  await expect(releaseBand).toContainText('9,152');
  await expect(releaseBand).toContainText('Bird benchmark');
  await expect(releaseBand).toContainText('Butterfly · 34 species');
  await expect(releaseBand).toContainText('FID statistics');

  await page.goto('http://127.0.0.1:4324/on-manifold-tfg/slides/?preview=takeaway#/section-16');
  await expect(page.locator('.slide-number')).toContainText('17 / 18');
  for (let index = 0; index < 3; index += 1) {
    await page.keyboard.press('ArrowRight');
    await page.waitForTimeout(700);
  }
  const takeaway = page.locator('.reveal .slides > section[data-source-section="20"]');
  await expect(takeaway).toContainText('Prediction target governs how TFG fails');
  await expect(takeaway).toContainText("TFG's weak link is the clean-image estimate");
  await expect(takeaway).toContainText('classifier Validity misses');
  await expect(takeaway).not.toContainText('Training-free guidance steers a frozen diffusion model');
});

test('project custom 404 is useful and noindexed', async ({ page }) => {
  const response = await page.goto('./missing-page/');
  expect(response?.status()).toBe(404);
  await expect(page.getByRole('heading', { level: 1 })).toContainText('left the manifold');
  await expect(page.locator('meta[name="robots"]')).toHaveAttribute('content', 'noindex,follow');
  await expect(page.locator('meta[name="citation_title"], meta[property="og:image"], script[type="application/ld+json"]')).toHaveCount(0);
});
