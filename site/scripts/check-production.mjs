const rootUrl = 'https://manluml.github.io/';
const projectUrl = 'https://manluml.github.io/on-manifold-tfg/';
const releaseKey = process.env.GITHUB_SHA ?? String(Date.now());

async function fetchProduction(url, accepts = () => true) {
  let lastError;
  for (let attempt = 1; attempt <= 12; attempt += 1) {
    try {
      const requestUrl = new URL(url);
      requestUrl.searchParams.set('release', releaseKey);
      const response = await fetch(requestUrl, {
        redirect: 'follow',
        signal: AbortSignal.timeout(15_000),
        headers: {
          'cache-control': 'no-cache',
          'user-agent': 'ManLuML production verifier',
        },
      });
      const html = await response.text();
      if (response.ok && accepts(html)) return { response, html };
      lastError = new Error(response.ok ? `Production marker not visible yet: ${url}` : `${response.status} ${url}`);
    } catch (error) {
      lastError = error;
    }
    if (attempt < 12) await new Promise((resolve) => setTimeout(resolve, 5_000));
  }
  throw lastError;
}

const [{ response: rootResponse, html: rootHtml }, { response: projectResponse, html: projectHtml }] = await Promise.all([
  fetchProduction(rootUrl),
  fetchProduction(projectUrl, (html) => html.includes('Can guidance hit the target and keep the image realistic?')
    && html.includes('jit-failure-mobile-640')
    && html.includes('Computer Vision -- ECCV 2026')
    && !/role="tab"|role="tabpanel"|type="range"|Scope and limitations|camera-ready/i.test(html)),
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
console.log('Verified the revised production marker, canonicals, and bidirectional cross-site navigation.');
