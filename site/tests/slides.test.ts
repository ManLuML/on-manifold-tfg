import { readFile, readdir } from 'node:fs/promises';
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

describe('legacy slide compatibility route', () => {
  it('keeps the URL recoverable without republishing excluded assets', async () => {
    expect(await filesUnder('public/slides')).toHaveLength(1);
    const html = await readFile('public/slides/index.html', 'utf8');
    expect(html).toContain('legacy-pages-pre-astro-2026-09-01');
    expect(html).toContain('c3f04b42c4857ac21b62515879bf6e9dd6e6f2a5');
    expect(html).not.toMatch(/<img|maumai-logo|seoultech-brand-guide|ddpm-fig|jit-fig/i);
  });

  it('keeps the existing public deck English-only', async () => {
    const html = await readFile('public/slides/index.html', 'utf8');
    expect(html).not.toMatch(/[\u3131-\u318E\uAC00-\uD7A3]/);
  });
});
