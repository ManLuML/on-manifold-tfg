import { access, readFile, readdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import path from 'node:path';
import sharp from 'sharp';
import { describe, expect, it } from 'vitest';

describe('public asset separation', () => {
  it('keeps the root project page affiliation artwork text-only', async () => {
    await expect(access('public/media/affiliations/maum-ai.png')).rejects.toBeDefined();
    await expect(access('public/media/affiliations/seoultech.png')).rejects.toBeDefined();
  });

  it('excludes affiliation-logo files from the project site while allowing them in the deck', async () => {
    const walk = async (directory: string): Promise<string[]> => (await Promise.all((await readdir(directory, { withFileTypes: true })).map((entry) => {
      const file = path.join(directory, entry.name);
      return entry.isDirectory() ? walk(file) : [file];
    }))).flat();
    const forbidden = new Set([
      '2465f21973871d90cbd62df1d446bbc937c22edb0d62514bac54068b194a71b0',
      'e45169a99acd5244d7342fecf0bbd7ec1635f027c68af59035860b29d4aca19e',
      '2f6a475f567157852e6ad7ee800ce4157d048101ea1a2e7607e8dee6640e3ed1',
      'efe18143560eab3d84ba5e0967832f2098407adceb6a6da572c6eb7eecdc6cb7',
    ]);
    for (const file of await walk('public')) {
      const relative = file.split(path.sep).join('/');
      if (relative.startsWith('public/slides/')) continue;
      const buffer = await readFile(file);
      expect(file).not.toMatch(/maumai|maum-ai|seoultech|media\/affiliations/i);
      expect(forbidden.has(createHash('sha256').update(buffer).digest('hex'))).toBe(false);
      if (/\.(?:html|css|js|json|svg|txt|xml)$/.test(file)) {
        expect(buffer.toString('utf8')).not.toMatch(/(?:src|href|url\()[^\n>)]*(?:maumai-logo|maum-ai\.png|seoultech\.png|seoultech-brand-guide|media\/affiliations)/i);
      }
    }
  });

  it('ships separate 1200×630 social media artwork', async () => {
    const metadata = await sharp('public/og/project-card.png').metadata();
    expect([metadata.width, metadata.height]).toEqual([1200, 630]);
  });

  it('ships matched mobile crops for the scroll-native failure sequence', async () => {
    for (const target of ['jit', 'sit', 'dit']) {
      const metadata = await sharp(`public/media/figures/${target}-failure-mobile-640.jpg`).metadata();
      expect([metadata.width, metadata.height]).toEqual([640, 684]);
    }
  });
});
