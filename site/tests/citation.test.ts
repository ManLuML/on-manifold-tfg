import { readFile } from 'node:fs/promises';
import { describe, expect, it } from 'vitest';
import { citationBibtex } from '../src/content/citation';

describe('provisional ECCV citation', () => {
  it('uses only verified conference metadata', () => {
    expect(citationBibtex).toContain('@inproceedings{lee2026onmanifold,');
    expect(citationBibtex).toContain('author    = {Lee, Yunsung and Lee, Hyeongmin}');
    expect(citationBibtex).toContain('booktitle = {Computer Vision -- ECCV 2026}');
    expect(citationBibtex).toContain('year      = {2026}');
    expect(citationBibtex).toContain('note      = {To appear}');
    expect(citationBibtex).not.toMatch(/\b(?:doi|pages|volume|publisher|editor|isbn|month|address)\s*=/i);
    expect((citationBibtex.match(/\{/g) ?? []).length).toBe((citationBibtex.match(/\}/g) ?? []).length);
  });

  it('keeps the repository README synchronized', async () => {
    const readme = await readFile('../README.md', 'utf8');
    expect(readme).toContain(citationBibtex);
  });
});
