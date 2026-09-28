import type { Metadata, Viewport } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Sentinela SST | Visão Computacional',
  description: 'Visão computacional em tempo real para inspeção visual de EPI, segurança em altura, ergonomia e levantamento de cargas.',
  applicationName: 'Sentinela SST',
};

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  viewportFit: 'cover',
  themeColor: '#07100f',
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="pt-BR">
      <body>{children}</body>
    </html>
  );
}
