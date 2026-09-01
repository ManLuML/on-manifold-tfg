import { access, readFile, readdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import path from 'node:path';
import sharp from 'sharp';

const dist = path.resolve('dist');
const html = await readFile(path.join(dist, 'index.html'), 'utf8');
const description = 'Prediction target determines whether training-free guidance fails gracefully or catastrophically. Explore the equations, controls, results, resources, and limitations.';
const pageTitle = 'Not All Prediction Targets Keep Training-Free Diffusion Guidance on the Manifold — ECCV 2026';
const required = [
  '<html lang="en">',
  '<link rel="canonical" href="https://manluml.github.io/on-manifold-tfg/">',
  `<meta name="description" content="${description}">`,
  '<meta property="og:type" content="article">',
  `<meta property="og:title" content="${pageTitle}">`,
  `<meta property="og:description" content="${description}">`,
  '<meta property="og:url" content="https://manluml.github.io/on-manifold-tfg/">',
  '<meta property="og:image" content="https://manluml.github.io/on-manifold-tfg/og/project-card.png">',
  '<meta property="og:image:width" content="1200">',
  '<meta property="og:image:height" content="630">',
  '<meta name="twitter:card" content="summary_large_image">',
  `<meta name="twitter:title" content="${pageTitle}">`,
  `<meta name="twitter:description" content="${description}">`,
  '<meta name="twitter:image" content="https://manluml.github.io/on-manifold-tfg/og/project-card.png">',
  'citation_pdf_url',
  'citation_version',
  'https://arxiv.org/pdf/2607.00647',
  '"@type":"ScholarlyArticle"',
  'Maum AI',
  'Seoul National University of Science and Technology',
  'id="abstract"',
  'id="method"',
  'id="results"',
  'id="bibtex"',
  'TFG robustness ranking',
  'most robust',
  'arXiv preprint',
  'media/figures/figure-1a-960.png',
  '93.3%', '21.5%', '0.5%', '32.9', '34.7', '38.1', '9,152',
];
for (const value of required) if (!html.includes(value)) throw new Error(`Missing built assertion: ${value}`);
if (/[\u3131-\u318E\uAC00-\uD7A3]/.test(html)) throw new Error('Built HTML must remain English-only.');
if (/<(?:script|img)[^>]+src="https?:\/\//.test(html) || /<link[^>]+rel="stylesheet"[^>]+href="https?:\/\//.test(html)) {
  throw new Error('Runtime CDN dependency found.');
}
if (/Republic of Korea|media\/affiliations|maum-ai\.png|seoultech\.png/i.test(html)) throw new Error('Affiliations must remain simplified and text-only.');
if (/x\s*&lt;\s*v\s*&lt;\s*ε/i.test(html) || /Camera-ready preprint/i.test(html)) throw new Error('Ambiguous ranking or public-preprint conflation found.');
const card = await sharp(path.join(dist, 'og/project-card.png')).metadata();
if (card.width !== 1200 || card.height !== 630) throw new Error('Project social card must be 1200×630.');
const compatibility = [
  'favicon.svg', 'og_card.png', 'rho_fid_vs_child_fid.png', 'rho_fid_vs_child_fid.webp',
  'rho_fid_vs_validity.png', 'rho_fid_vs_validity.webp', 'style_gram_vs_validity.png',
  'style_gram_vs_validity.webp', 'teaser.png', 'teaser.webp', 'vis_off_dit.jpg',
  'vis_off_dit.webp', 'vis_off_jit.jpg', 'vis_off_jit.webp', 'vis_on_dit.jpg',
  'vis_on_dit.webp', 'vis_on_jit.jpg', 'vis_on_jit.webp',
];
for (const file of compatibility) await access(path.join(dist, 'static/images', file));
await access(path.join(dist, 'slides/index.html'));
const slidesHtml = await readFile(path.join(dist, 'slides/index.html'), 'utf8');
if (!slidesHtml.includes('legacy-pages-pre-astro-2026-09-01') || /<img|maumai-logo|seoultech-brand-guide|ddpm-fig|jit-fig/i.test(slidesHtml)) {
  throw new Error('Legacy slide route must remain recoverable without republishing excluded assets.');
}
for (const file of ['404.html', 'robots.txt', 'sitemap-index.xml', '.nojekyll']) await access(path.join(dist, file));

async function filesUnder(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  const nested = await Promise.all(entries.map((entry) => {
    const file = path.join(directory, entry.name);
    return entry.isDirectory() ? filesUnder(file) : [file];
  }));
  return nested.flat();
}

const builtFiles = await filesUnder(dist);
const forbiddenLogoHashes = new Set([
  '2465f21973871d90cbd62df1d446bbc937c22edb0d62514bac54068b194a71b0',
  'e45169a99acd5244d7342fecf0bbd7ec1635f027c68af59035860b29d4aca19e',
  '2f6a475f567157852e6ad7ee800ce4157d048101ea1a2e7607e8dee6640e3ed1',
  'efe18143560eab3d84ba5e0967832f2098407adceb6a6da572c6eb7eecdc6cb7',
]);
for (const file of builtFiles) {
  const relative = path.relative(dist, file);
  const buffer = await readFile(file);
  if (/maumai|maum-ai|seoultech|media\/affiliations/i.test(relative)
    || forbiddenLogoHashes.has(createHash('sha256').update(buffer).digest('hex'))) {
    throw new Error(`Forbidden affiliation-logo artifact found: ${relative}`);
  }
  if (/\.(?:html|css|js|json|svg|txt|xml)$/.test(file)
    && /(?:src|href|url\()[^\n>)]*(?:maumai-logo|maum-ai\.png|seoultech\.png|seoultech-brand-guide|media\/affiliations)/i.test(buffer.toString('utf8'))) {
    throw new Error(`Forbidden affiliation-logo reference found: ${relative}`);
  }
}

const assetDocuments = [html, await readFile(path.join(dist, '404.html'), 'utf8'), slidesHtml];
for (const file of builtFiles.filter((file) => file.endsWith('.css'))) assetDocuments.push(await readFile(file, 'utf8'));
const internalAssets = new Set();
for (const document of assetDocuments) {
  for (const match of document.matchAll(/(?:src|href)="(\/on-manifold-tfg\/[^"#]*)"/g)) {
    internalAssets.add(match[1].split(/[?#]/, 1)[0].slice('/on-manifold-tfg/'.length));
  }
  for (const match of document.matchAll(/srcset="([^"]+)"/g)) {
    for (const candidate of match[1].split(',')) {
      const pathname = candidate.trim().split(/\s+/, 1)[0];
      if (pathname.startsWith('/on-manifold-tfg/')) internalAssets.add(pathname.slice('/on-manifold-tfg/'.length));
    }
  }
  for (const match of document.matchAll(/url\((\/on-manifold-tfg\/[^)]+)\)/g)) {
    internalAssets.add(match[1].split(/[?#]/, 1)[0].slice('/on-manifold-tfg/'.length));
  }
}
for (const relative of internalAssets) {
  const target = relative === '' || relative.endsWith('/') ? path.join(dist, relative, 'index.html') : path.join(dist, relative);
  await access(target);
}
console.log('Verified project distribution metadata, claims, compatibility, assets, language, and runtime isolation.');
