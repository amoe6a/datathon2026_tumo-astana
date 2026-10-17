# Frontend reference and attribution record

The interface uses local HTML, CSS and JavaScript; it does not require React,
Tailwind, a CDN, web fonts or a Node.js production server.

## Illustrative photographs in the three-step section

Downloaded on 17 October 2026 from Unsplash's free library. Each original
photo page identifies the image as free to use under the standard
[Unsplash License](https://unsplash.com/license). These are illustrative
website assets, separate from all event footage, evidence and ML data.
The images are not part of any training or evaluation dataset.

| Local file | Photographer | Original source |
|---|---|---|
| `images/steps/recording.webp` | Kevin Kandlbinder | [Security camera](https://unsplash.com/photos/a-security-camera-mounted-to-the-side-of-a-building-1hUVnQ64-SQ) |
| `images/steps/timeline.webp` | Jakob Owens | [Video-editing monitor](https://unsplash.com/photos/black-flat-screen-computer-monitor-abdBDZT9LJk) |
| `images/steps/documents.webp` | Scott Graham | [Writing on paper](https://unsplash.com/photos/man-writing-on-paper-OQMZwNd3ThU) |

The local WebP files use Unsplash's image-delivery width/quality options.
Cropping and visual tint are applied by CSS; no runtime image host is used.
Photographer/source credits also appear below the cards. The standard
Unsplash License permits downloading, modifying and using free-library
images, including commercial use, and excludes resale of unmodified images
and replication of a competing image service. This notice does not imply
endorsement of Forte Review by the photographers, Unsplash or any software
visible in the photographs.

## Illustrative photograph in the sources section

`images/sources/law-books.jpg` is a photograph of legal reference books by
Krists Luhaers: [original photograph](https://unsplash.com/photos/law-books-are-lined-up-on-a-shelf-ioRcKUIiIdw).
Downloaded on 17 October 2026. The original page identifies it as free to use
under the standard [Unsplash License](https://unsplash.com/license).
The JPEG is served locally, with cropping and muted colors applied in CSS.
The page includes an illustration label and photographer credit.

This is a decorative illustration of reference material, not a photograph
of the applicable Kazakhstan regulation, a source citation, bank footage,
evidence or an ML dataset item. Actual regulatory sources remain in the
authenticated catalog and result cards. No endorsement is implied.

## Brainwave visual reference

Repository: https://github.com/adrianhajdin/brainwave
Inspected revision: 73eaea5c1ce9543a0bb76858ec80e39d0201461f

Inspected `src/components/Hero.jsx`, `src/components/design/Hero.jsx`,
`src/components/Section.jsx`, `src/components/Benefits.jsx`,
`src/assets/svg/SectionSvg.jsx`, `src/index.css`, and the asset tree.

Reference ideas: centered oversized hero, concentric orbital geometry,
gradient-framed central workspace, layered plinth, section rails and corner
markers, numbered asymmetric feature cards. These are independently implemented
in `index.html` and `style.css`. No Brainwave source or image assets are copied:
no root license was found in the inspected tree and the GitHub license endpoint
returned 404. Its robot, photographs, logos and font files are not included.

## AI SaaS Landing adaptation

Repository: https://github.com/HardikRathod123/ai-saas-landing-page-template
Branch: master
Inspected revision: 488341b21ec4eb4e9fbee2970e6017b19f22c949

Inspected `LICENSE`, `src/sections/Hero.tsx`, `src/sections/Header.tsx`,
`src/sections/Features.tsx`, `src/app/globals.css`, the asset tree,
`src/assets/images/underline.svg`, and `src/assets/images/loader.svg`.

The layered padding-box/conic-gradient border from `.border-gradient` in
`src/app/globals.css` is adapted in `.workspace-shell` with project-specific
colors. Hero.tsx's isolated radial background, concentric orbit arrangement and
central framed content inform the independently built CSS layout. Header.tsx
informs the compact wordmark/navigation arrangement. React state, Motion
dependencies, stock robot image, logos, testimonial portraits and SVG assets
are not transferred. No externally hosted resources are required.

The following notice applies to the adapted MIT source:

MIT License

Copyright (c) 2024 Rathod Hardik

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
