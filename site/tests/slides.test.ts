import { access, readFile, readdir } from 'node:fs/promises';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

async function filesUnder(directory: string): Promise<string[]> {
  const entries = await readdir(directory, { withFileTypes: true });
  const nested = await Promise.all(entries.map((entry) => {
    const target = path.join(directory, entry.name);
    return entry.isDirectory() ? filesUnder(target) : [target];
  }));
  return nested.flat();
}

describe('ECCV slide deck', () => {
  it('publishes the complete project-owned presentation bundle', async () => {
    expect((await filesUnder('public/slides')).length).toBeGreaterThan(120);
    const html = await readFile('public/slides/index.html', 'utf8');
    expect(html).toContain('Not All Prediction Targets Keep Training-Free Diffusion Guidance on the Manifold');
    expect(html).toContain('Do prediction targets decide how guidance fails?');
    expect(html).toContain('data-source-section="19"');
    expect(html).toContain("slideNumber: 'c/t'");
    expect(html.match(/data-source-section=/g)).toHaveLength(17);
    const sequence = [
      'guidance-sequence-05-success.png',
      'guidance-sequence-06-catastrophic.png',
      'guidance-sequence-07-graceful.png',
      'guidance-sequence-04-combined.png',
    ];
    expect(sequence.map((file) => html.indexOf(file))).toEqual([...sequence.map((file) => html.indexOf(file))].sort((a, b) => a - b));
    expect(sequence.every((file) => html.includes(file))).toBe(true);
    expect(html).toContain('rewrite=google-slides-sequence-5-6-7-4');
    expect(html).toMatch(/guidance-sequence-06-catastrophic\.png[^>]+current-visible[^>]+data-fragment-index="0"/);
    expect(html).toMatch(/guidance-sequence-07-graceful\.png[^>]+current-visible[^>]+data-fragment-index="1"/);
    expect(html).toMatch(/guidance-sequence-04-combined\.png[^>]+fade-in[^>]+data-fragment-index="2"/);
    expect(html).not.toContain('<aside class="notes"');
    expect(html).not.toMatch(/\sdata-notes=/);
    for (const file of [
      'on-manifold-tfg.css',
      'assets/eccv-navbar-logo.svg',
      'assets/outcome-success-photo.png',
      'assets/outcome-catastrophic-photo.png',
      'assets/outcome-graceful-photo.png',
      'assets/guidance-sequence-05-success.png',
      'assets/guidance-sequence-06-catastrophic.png',
      'assets/guidance-sequence-07-graceful.png',
      'assets/guidance-sequence-04-combined.png',
      'fonts/Pretendard-Regular.woff2',
      'on-manifold-tfg_files/libs/revealjs/dist/reveal.js',
    ]) await access(path.join('public/slides', file));
  });

  it('keeps the public deck English-only', async () => {
    const html = await readFile('public/slides/index.html', 'utf8');
    expect(html).not.toMatch(/[\u3131-\u318E\uAC00-\uD7A3]/);
  });
});
