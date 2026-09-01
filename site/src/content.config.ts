import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

const url = z.url();

const projects = defineCollection({
  loader: glob({ pattern: '**/*.{yaml,yml}', base: './src/content/projects' }),
  schema: z.object({
    title: z.string().min(1),
    shortTitle: z.string().min(1),
    venue: z.literal('ECCV 2026'),
    year: z.literal(2026),
    abstract: z.string().min(1),
    datePublished: z.coerce.date().transform((date) => date.toISOString().slice(0, 10)),
    dateModified: z.coerce.date().transform((date) => date.toISOString().slice(0, 10)),
    authors: z.array(z.object({
      name: z.string().min(1),
      marker: z.string().min(1),
      profileUrl: url,
      email: z.email(),
      affiliation: z.string().min(1),
    })).length(2),
    links: z.object({
      paper: url,
      paperAbstract: url,
      code: url,
      datasets: z.array(z.object({ label: z.string().min(1), url })).length(3),
    }),
  }),
});

const pages = defineCollection({
  loader: glob({ pattern: 'page.mdx', base: './src/content' }),
  schema: z.object({ title: z.string().min(1), reviewedAgainst: z.string().min(7) }),
});

export const collections = { projects, pages };
