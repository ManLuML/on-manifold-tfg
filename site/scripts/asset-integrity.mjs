import { createHash } from 'node:crypto';
import { readFile, readdir, writeFile } from 'node:fs/promises';
import path from 'node:path';

const manifestPath = 'assets/generated-assets.json';
const targets = [
  'scripts/process-assets.mjs',
  'package-lock.json',
  'assets/source',
  'public/brand',
  'public/licenses',
  'public/media/figures',
  'public/og',
  'public/static/images',
  'public/slides',
];

async function filesUnder(target) {
  const entries = await readdir(target, { withFileTypes: true });
  const nested = await Promise.all(entries.map((entry) => {
    const file = path.join(target, entry.name);
    return entry.isDirectory() ? filesUnder(file) : [file];
  }));
  return nested.flat();
}

const files = (await Promise.all(targets.map(async (target) => {
  try {
    const entries = await readdir(target, { withFileTypes: true });
    return entries.length >= 0 ? filesUnder(target) : [];
  } catch {
    return [target];
  }
}))).flat().sort();

const current = {};
for (const file of files) current[file] = createHash('sha256').update(await readFile(file)).digest('hex');

if (process.argv.includes('--record')) {
  await writeFile(manifestPath, `${JSON.stringify({ version: 1, files: current }, null, 2)}\n`);
  console.log(`Recorded ${files.length} deterministic asset inputs and outputs.`);
} else {
  const recorded = JSON.parse(await readFile(manifestPath, 'utf8')).files;
  if (JSON.stringify(current) !== JSON.stringify(recorded)) {
    const changed = [...new Set([...Object.keys(current), ...Object.keys(recorded)])].filter((file) => current[file] !== recorded[file]);
    throw new Error(`Generated asset integrity mismatch. Run npm run assets and review:\n${changed.join('\n')}`);
  }
  console.log(`Verified ${files.length} deterministic asset inputs and outputs.`);
}
