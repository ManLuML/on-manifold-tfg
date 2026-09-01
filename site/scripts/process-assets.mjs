import { createHash } from 'node:crypto';
import { copyFile, mkdir, readFile } from 'node:fs/promises';
import path from 'node:path';
import sharp from 'sharp';

const root = process.cwd();
const sourceDir = path.join(root, 'assets/source/paper');
const publicDir = path.join(root, 'public');
const figuresDir = path.join(publicDir, 'media/figures');
const ogDir = path.join(publicDir, 'og');
const legacyDir = path.join(publicDir, 'static/images');

await Promise.all([
  mkdir(figuresDir, { recursive: true }),
  mkdir(ogDir, { recursive: true }),
  mkdir(legacyDir, { recursive: true }),
]);

const figures = [
  { slug: 'teaser', input: 'teaser.png', widths: [720, 1200, 1800], fallback: 'png' },
  { slug: 'figure-1a', input: 'figure-1a.png', widths: [720, 960], fallback: 'png' },
  { slug: 'crossed-lines-grid', input: 'crossed-lines-grid.png', widths: [720, 1400], fallback: 'png' },
  { slug: 'onmanifold-vs-dim', input: 'onmanifold-vs-dim.png', widths: [720], fallback: 'png' },
  { slug: 'rho-fid-vs-validity', input: 'rho_fid_vs_validity.png', widths: [720, 1200], fallback: 'png' },
  { slug: 'rho-fid-vs-child-fid', input: 'rho_fid_vs_child_fid.png', widths: [720, 1200], fallback: 'png' },
  { slug: 'jit-failure-grid', input: 'vis_off_jit.jpg', widths: [720, 1400, 2200], fallback: 'jpg' },
  { slug: 'sit-failure-grid', input: 'vis_off_sit.jpg', widths: [720, 1400, 2200], fallback: 'jpg' },
  { slug: 'dit-failure-grid', input: 'vis_off_dit.jpg', widths: [720, 1400, 2200], fallback: 'jpg' },
  { slug: 'lgd', input: 'bird_lgd_a.png', widths: [640, 960], fallback: 'png' },
  { slug: 'freedom', input: 'bird_lgd_b.png', widths: [640, 960], fallback: 'png' },
  { slug: 'butterfly', input: 'butterfly_pareto.png', widths: [640, 960], fallback: 'png' },
  { slug: 'style', input: 'style_gram_vs_validity.png', widths: [640, 960], fallback: 'png' },
  { slug: 'precision-recall', input: 'rho_precision_vs_recall.png', widths: [640, 960], fallback: 'png' },
];

const generated = [];

for (const figure of figures) {
  const input = path.join(sourceDir, figure.input);
  for (const width of figure.widths) {
    const base = path.join(figuresDir, `${figure.slug}-${width}`);
    const raster = sharp(input).resize({ width, withoutEnlargement: true });
    const avifOutput = `${base}.avif`;
    const webpOutput = `${base}.webp`;
    const fallbackOutput = `${base}.${figure.fallback}`;
    await Promise.all([
      raster.clone().avif({ quality: 57, effort: 6 }).toFile(avifOutput),
      raster.clone().webp({ quality: 84 }).toFile(webpOutput),
      figure.fallback === 'jpg'
        ? raster.clone().jpeg({ quality: 88, mozjpeg: true }).toFile(fallbackOutput)
        : raster.clone().png({ compressionLevel: 9 }).toFile(fallbackOutput),
    ]);
    generated.push(avifOutput, webpOutput, fallbackOutput);
  }
}

const visual = await sharp(path.join(sourceDir, 'figure-1a.png'))
  .resize({ width: 420, height: 500, fit: 'contain', background: '#F7FAFC' })
  .png()
  .toBuffer();

const cardBackground = Buffer.from(`
<svg width="1200" height="630" viewBox="0 0 1200 630" xmlns="http://www.w3.org/2000/svg">
  <rect width="1200" height="630" fill="#F7FAFC"/>
  <rect x="0" y="0" width="24" height="630" fill="#0D47A1"/>
  <text x="66" y="90" fill="#0D47A1" font-family="Arial, Helvetica, sans-serif" font-size="28" font-weight="800" letter-spacing="2">ECCV 2026</text>
  <text x="66" y="170" fill="#0A0E1A" font-family="Arial, Helvetica, sans-serif" font-size="39" font-weight="800">Not All Prediction Targets Keep</text>
  <text x="66" y="222" fill="#0A0E1A" font-family="Arial, Helvetica, sans-serif" font-size="39" font-weight="800">Training-Free Diffusion Guidance</text>
  <text x="66" y="274" fill="#0A0E1A" font-family="Arial, Helvetica, sans-serif" font-size="39" font-weight="800">on the Manifold</text>
  <text x="66" y="355" fill="#40506A" font-family="Arial, Helvetica, sans-serif" font-size="27">Yunsung Lee · Hyeongmin Lee</text>
  <text x="66" y="410" fill="#0D47A1" font-family="Arial, Helvetica, sans-serif" font-size="23" font-weight="700">TFG robustness: x first · v second · ε third</text>
  <text x="66" y="454" fill="#40506A" font-family="Arial, Helvetica, sans-serif" font-size="23">Which failures stay on the data manifold?</text>
  <rect x="66" y="518" width="360" height="4" rx="2" fill="#0891B2"/>
</svg>`);

const projectCard = path.join(ogDir, 'project-card.png');
await sharp(cardBackground)
  .composite([{ input: visual, left: 760, top: 65 }])
  .png({ compressionLevel: 9 })
  .toFile(projectCard);
generated.push(projectCard);

await Promise.all([
  copyFile(path.join(publicDir, 'brand/manluml-mascot.svg'), path.join(legacyDir, 'favicon.svg')),
  copyFile(projectCard, path.join(legacyDir, 'og_card.png')),
  copyFile(path.join(sourceDir, 'teaser.png'), path.join(legacyDir, 'teaser.png')),
  sharp(path.join(sourceDir, 'teaser.png')).resize({ width: 1800 }).webp({ quality: 86 }).toFile(path.join(legacyDir, 'teaser.webp')),
  copyFile(path.join(sourceDir, 'rho_fid_vs_validity.png'), path.join(legacyDir, 'rho_fid_vs_validity.png')),
  sharp(path.join(sourceDir, 'rho_fid_vs_validity.png')).webp({ quality: 86 }).toFile(path.join(legacyDir, 'rho_fid_vs_validity.webp')),
  copyFile(path.join(sourceDir, 'rho_fid_vs_child_fid.png'), path.join(legacyDir, 'rho_fid_vs_child_fid.png')),
  sharp(path.join(sourceDir, 'rho_fid_vs_child_fid.png')).webp({ quality: 86 }).toFile(path.join(legacyDir, 'rho_fid_vs_child_fid.webp')),
  copyFile(path.join(sourceDir, 'style_gram_vs_validity.png'), path.join(legacyDir, 'style_gram_vs_validity.png')),
  sharp(path.join(sourceDir, 'style_gram_vs_validity.png')).webp({ quality: 86 }).toFile(path.join(legacyDir, 'style_gram_vs_validity.webp')),
  ...['vis_on_jit', 'vis_off_jit', 'vis_on_dit', 'vis_off_dit'].flatMap((name) => [
    copyFile(path.join(sourceDir, `${name}.jpg`), path.join(legacyDir, `${name}.jpg`)),
    sharp(path.join(sourceDir, `${name}.jpg`)).webp({ quality: 84 }).toFile(path.join(legacyDir, `${name}.webp`)),
  ]),
]);

for (const file of generated) {
  const buffer = await readFile(file);
  const hash = createHash('sha256').update(buffer).digest('hex');
  const metadata = await sharp(buffer).metadata();
  console.log(JSON.stringify({
    file: path.relative(root, file),
    sha256: hash,
    width: metadata.width,
    height: metadata.height,
    format: metadata.format,
  }));
}
