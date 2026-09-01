const rootUrl = 'https://manluml.github.io/';
const projectUrl = 'https://manluml.github.io/on-manifold-tfg/';

async function fetchProduction(url) {
  let lastError;
  for (let attempt = 1; attempt <= 12; attempt += 1) {
    try {
      const response = await fetch(url, {
        redirect: 'follow',
        signal: AbortSignal.timeout(15_000),
        headers: {
          'cache-control': 'no-cache',
          'user-agent': 'ManLuML production verifier',
        },
      });
      const html = await response.text();
      if (response.ok) return { response, html };
      lastError = new Error(`${response.status} ${url}`);
    } catch (error) {
      lastError = error;
    }
    if (attempt < 12) await new Promise((resolve) => setTimeout(resolve, 5_000));
  }
  throw lastError;
}

const [{ response: rootResponse, html: rootHtml }, { response: projectResponse, html: projectHtml }] = await Promise.all([
  fetchProduction(rootUrl),
  fetchProduction(projectUrl),
]);

if (!rootHtml.includes(`href="${projectUrl}"`)) {
  throw new Error('Production root does not link to the project page.');
}
if (!projectHtml.includes(`<link rel="canonical" href="${projectUrl}"`)) {
  throw new Error('Production project canonical is missing or incorrect.');
}
if (!projectHtml.includes(`href="${rootUrl}"`)) {
  throw new Error('Production project does not link back to the organization home.');
}

console.log(`${rootResponse.status} ${rootUrl}`);
console.log(`${projectResponse.status} ${projectUrl}`);
console.log('Verified production canonicals and bidirectional cross-site navigation.');
