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
  await expect(page.locator('meta[name="citation_version"]')).toHaveAttribute('content', 'Accepted camera-ready source; public PDF is arXiv v1');
  expect(await page.locator('script[type="application/ld+json"]').textContent()).toContain('ScholarlyArticle');
  await expect(page.getByRole('link', { name: /^Paper/ }).first()).toHaveAttribute('href', 'https://arxiv.org/pdf/2607.00647');
  await expect(page.getByRole('link', { name: /^Code/ }).first()).toHaveAttribute('href', 'https://github.com/ManLuML/on-manifold-tfg');
  await expect(page.getByRole('link', { name: /^Data/ }).first()).toHaveAttribute('href', '#resources');
  await expect(page.getByRole('link', { name: /^Cite/ }).first()).toHaveAttribute('href', '#bibtex');
  await expect(page.locator('.affiliations li').nth(0)).toContainText('Maum AI');
  await expect(page.locator('.affiliations li').nth(1)).toContainText('Seoul National University of Science and Technology');
  await expect(page.getByText(/Republic of Korea/)).toHaveCount(0);
  await expect(page.locator('img[src*="affiliations"], img[src*="maum"], img[src*="seoultech"]')).toHaveCount(0);
  await expect(page.getByText('ECCV 2026', { exact: true }).first()).toBeVisible();
  await expect(page.getByRole('list', { name: 'Prediction targets ranked from most to least robust for training-free guidance' })).toContainText('x-prediction');
  await expect(page.getByRole('heading', { name: 'arXiv preprint' })).toBeVisible();
  await expect(page.locator('#abstract, #method, #results, #bibtex')).toHaveCount(4);
  await expect(page.getByText('93.3%', { exact: true }).first()).toBeVisible();
  await expect(page.getByText('21.5%', { exact: true }).first()).toBeVisible();
  await expect(page.getByText('0.5%', { exact: true }).first()).toBeVisible();
  await expect(page.getByText('9,152 generated images')).toBeVisible();
  expect(externalRuntimeRequests).toEqual([]);
});

test('both explanatory interactions are keyboard complete', async ({ page }) => {
  await page.goto('./#results');
  const epsilonTab = page.getByRole('tab', { name: 'ε ε-prediction' });
  await epsilonTab.focus();
  await epsilonTab.press('Enter');
  await expect(epsilonTab).toHaveAttribute('aria-selected', 'true');
  await expect(page.getByRole('tabpanel', { name: 'ε ε-prediction' })).toBeVisible();
  await epsilonTab.press('Home');
  await expect(page.getByRole('tab', { name: 'x x-prediction' })).toHaveAttribute('aria-selected', 'true');

  const slider = page.getByRole('slider', { name: 'Flow-matching time (0 = noise, 1 = clean)' });
  await slider.focus();
  await slider.press('Home');
  await expect(slider).toHaveValue('0.02');
  await expect(page.locator('[data-time-output]')).toHaveText('t = 0.02');
  await expect(page.locator('[data-epsilon-value]')).toHaveText('49.00');
  await slider.press('End');
  await expect(slider).toHaveValue('1');
  await expect(page.locator('[data-epsilon-value]')).toHaveText('0.00');
});

test('formula and manifold figure visual baselines remain balanced', async ({ page }) => {
  await page.goto('./#method');
  await expect(page.locator('.coefficient-grid')).toHaveScreenshot('project-formulas-desktop.png');
  await expect(page.locator('.paper-manifold-figure')).toHaveScreenshot('project-manifold-figure-desktop.png');
});

test('static fallback is complete without JavaScript', async ({ browser }) => {
  const context = await browser.newContext({ javaScriptEnabled: false });
  const page = await context.newPage();
  await page.goto('http://127.0.0.1:4324/on-manifold-tfg/');
  await expect(page.locator('[role="tabpanel"]')).toHaveCount(3);
  for (const panel of await page.locator('[role="tabpanel"]').all()) await expect(panel).toBeVisible();
  await expect(page.getByRole('table', { name: 'Recovery-error multiplier before each target’s base error' })).toBeVisible();
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
  expect(await page.locator('html').evaluate((element) => getComputedStyle(element).scrollBehavior)).toBe('auto');
});

test('320px reflow keeps text affiliations and content within the viewport', async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 568 });
  await page.goto('./');
  await expect(page.locator('.affiliations')).toBeVisible();
  const dimensions = await page.evaluate(() => ({ width: document.documentElement.clientWidth, scrollWidth: document.documentElement.scrollWidth }));
  expect(dimensions.scrollWidth).toBeLessThanOrEqual(dimensions.width);
  expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
  await expect(page).toHaveScreenshot('project-320.png');
  await expect(page.locator('.affiliations')).toHaveScreenshot('project-affiliations-320.png');
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
  const interactionLatency = await page.getByRole('slider', { name: 'Flow-matching time (0 = noise, 1 = clean)' }).evaluate((element) => new Promise<number>((resolve) => {
    const input = element as HTMLInputElement;
    const start = performance.now();
    input.value = '0.02';
    input.dispatchEvent(new Event('input', { bubbles: true }));
    requestAnimationFrame(() => resolve(performance.now() - start));
  }));
  console.log(`Project lab metrics: LCP=${metrics.lcp.toFixed(1)}ms CLS=${metrics.cls.toFixed(4)} interaction=${interactionLatency.toFixed(1)}ms`);
  expect(metrics.lcp).toBeLessThanOrEqual(2500);
  expect(metrics.cls).toBeLessThanOrEqual(0.1);
  expect(interactionLatency).toBeLessThanOrEqual(200);
});

test('legacy asset URLs remain available', async ({ request }) => {
  for (const asset of legacyAssets) {
    const response = await request.get(`http://127.0.0.1:4324/on-manifold-tfg/static/images/${asset}`);
    expect(response.ok(), asset).toBe(true);
    expect(response.headers()['content-type']).toMatch(/^image\//);
  }
});

test('the already-published slide URL remains safely recoverable', async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 568 });
  const response = await page.goto('http://127.0.0.1:4324/on-manifold-tfg/slides/');
  expect(response?.ok()).toBe(true);
  await expect(page.getByRole('heading', { level: 1 })).toContainText('published slide source remains preserved');
  await expect(page.getByRole('link', { name: /preserved source on GitHub/ })).toHaveAttribute('href', /legacy-pages-pre-astro-2026-09-01/);
  await expect(page.locator('img')).toHaveCount(0);
  const dimensions = await page.evaluate(() => ({ width: document.documentElement.clientWidth, scrollWidth: document.documentElement.scrollWidth }));
  expect(dimensions.scrollWidth).toBeLessThanOrEqual(dimensions.width);
  expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
});

test('project custom 404 is useful and noindexed', async ({ page }) => {
  const response = await page.goto('./missing-page/');
  expect(response?.status()).toBe(404);
  await expect(page.getByRole('heading', { level: 1 })).toContainText('left the manifold');
  await expect(page.locator('meta[name="robots"]')).toHaveAttribute('content', 'noindex,follow');
  await expect(page.locator('meta[name="citation_title"], meta[property="og:image"], script[type="application/ld+json"]')).toHaveCount(0);
});
