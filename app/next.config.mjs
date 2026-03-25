/** @type {import('next').NextConfig} */
const nextConfig = {
  // Required for GitHub Pages: export as a fully static site
  output: 'export',

  // GitHub Pages serves the site at /<repo-name>/
  basePath: '/color-predictor',

  // Required when using `output: 'export'` with Next.js image optimization
  images: {
    unoptimized: true,
  },
};

export default nextConfig;
