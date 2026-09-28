'use client';

import VisionCamera from './VisionCamera';

export default function SentinelaApp() {
  return (
    <main className="appShell">
      <header className="topbar">
        <div className="brand">
          <span className="brandShield" aria-hidden="true">
            <span className="brandCheck">✓</span>
          </span>
          <div>
            <p className="brandKicker">INSPEÇÃO VISUAL • SST</p>
            <h1>Sentinela SST</h1>
          </div>
        </div>
        <div className="privacyPill">
          <span className="privacyDot" />
          análise de EPI
        </div>
      </header>

      <section className="heroIntro">
        <div>
          <p className="sectionKicker">INSPEÇÃO DE EPI</p>
          <h2>Identificação visual de equipamentos de proteção</h2>
          <p>
            Envie uma foto ou use a câmera para verificar capacete, óculos de proteção,
            vestimenta refletiva, luvas e calçado.
          </p>
        </div>
      </section>

      <VisionCamera />

      <footer className="footerNote">
        <span>Sentinela SST</span>
        <p>
          Ferramenta de apoio visual para inspeções de Segurança do Trabalho.
          O resultado deve ser confirmado pelo profissional responsável quando a imagem
          estiver parcial, distante, desfocada ou com algum item oculto.
        </p>
      </footer>
    </main>
  );
}
