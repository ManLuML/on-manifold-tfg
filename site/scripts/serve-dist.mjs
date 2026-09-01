import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import path from 'node:path';

const args = new Map(process.argv.slice(2).reduce((pairs, value, index, values) => {
  if (value.startsWith('--')) pairs.push([value.slice(2), values[index + 1]]);
  return pairs;
}, []));
const port = Number(args.get('port') ?? 4324);
const base = (args.get('base') ?? '/on-manifold-tfg').replace(/\/$/, '') || '/';
const dist = path.resolve('dist');
const mime = {
  '.avif': 'image/avif',
  '.css': 'text/css; charset=utf-8',
  '.html': 'text/html; charset=utf-8',
  '.ico': 'image/x-icon',
  '.jpg': 'image/jpeg',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.txt': 'text/plain; charset=utf-8',
  '.webp': 'image/webp',
  '.woff2': 'font/woff2',
  '.xml': 'application/xml; charset=utf-8',
};

const server = createServer(async (request, response) => {
  const pathname = decodeURIComponent(new URL(request.url ?? '/', 'http://localhost').pathname);
  if (base !== '/' && pathname !== base && !pathname.startsWith(`${base}/`)) {
    response.writeHead(404);
    response.end('Not found');
    return;
  }
  const relative = base === '/' ? pathname : pathname.slice(base.length) || '/';
  const safe = path.normalize(relative).replace(/^(\.\.(\/|\\|$))+/, '').replace(/^[/\\]+/, '');
  let target = path.join(dist, safe);
  try {
    const details = await stat(target);
    if (details.isDirectory()) target = path.join(target, 'index.html');
    const body = await readFile(target);
    response.writeHead(200, { 'content-type': mime[path.extname(target)] ?? 'application/octet-stream' });
    response.end(body);
  } catch {
    try {
      const body = await readFile(path.join(dist, '404.html'));
      response.writeHead(404, { 'content-type': 'text/html; charset=utf-8' });
      response.end(body);
    } catch {
      response.writeHead(404);
      response.end('Not found');
    }
  }
});

server.listen(port, '127.0.0.1', () => console.log(`Serving ${dist} at http://127.0.0.1:${port}${base === '/' ? '/' : `${base}/`}`));
