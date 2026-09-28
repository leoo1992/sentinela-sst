import type { MetadataRoute } from 'next';

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: 'Sentinela SST',
    short_name: 'Sentinela SST',
    description: 'Visão computacional em tempo real para Segurança e Saúde no Trabalho.',
    start_url: '/',
    display: 'standalone',
    background_color: '#07100f',
    theme_color: '#07100f',
  };
}
