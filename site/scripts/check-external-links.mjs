import { readFile } from 'node:fs/promises';

const html = `${await readFile('dist/index.html', 'utf8')}\n${await readFile('dist/slides/index.html', 'utf8')}`;
const urls = [...new Set([...html.matchAll(/href="(https:\/\/[^"#]+(?:#[^"]*)?)"/g)].map((match) => match[1]))]
  .filter((url) => !url.startsWith('https://manluml.github.io/'));

const definitiveFailures = [];
for (const url of urls) {
  let status = 0;
  let error;
  for (let attempt = 0; attempt < 2; attempt += 1) {
    try {
      let response = await fetch(url, { method: 'HEAD', redirect: 'follow', signal: AbortSignal.timeout(15_000), headers: { 'user-agent': 'ManLuML link verifier' } });
      if (response.status === 405) response = await fetch(url, { redirect: 'follow', signal: AbortSignal.timeout(15_000), headers: { 'user-agent': 'ManLuML link verifier', range: 'bytes=0-0' } });
      status = response.status;
      if (status !== 429 && status < 500) break;
    } catch (caught) {
      error = caught;
    }
  }
  if (status === 404 || status === 410) definitiveFailures.push(`${status} ${url}`);
  else if (!status || status === 403 || status === 429 || status >= 500) console.warn(`Transient or policy-limited external check: ${status || error?.name || 'error'} ${url}`);
  else console.log(`${status} ${url}`);
}
if (definitiveFailures.length) throw new Error(`Definitively broken external links:\n${definitiveFailures.join('\n')}`);
