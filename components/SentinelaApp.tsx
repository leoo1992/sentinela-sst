'use client';

import { useState } from 'react';
import VisionCamera from './VisionCamera';
import type { ModuleId } from '@/lib/types';

const modules: Array<{ id: ModuleId; label: string; eyebrow: string; description: string }> = [
  { id: 'epi', label: 'Inspeção de EPI', eyebrow: 'EPI', description: 'Verifique visualmente os principais equipamentos de proteção do trabalhador.' },
  { id: 'altura', label: 'Segurança em Altura', eyebrow: 'ALTURA', description: 'Acompanhe a aproximação do trabalhador de uma área de risco e a presença de capacete.' },
  { id: 'ergonomia', label: 'Análise Ergonômica', eyebrow: 'ERGO', description: 'Observe postura, inclinação do tronco, pescoço, joelhos e possíveis assimetrias.' },
  { id: 'cargas', label: 'Levantamento de Cargas', eyebrow: 'CARGAS', description: 'Acompanhe a postura durante o levantamento e identifique movimentos que exigem atenção.' },
];

export default function SentinelaApp() {
  const [moduleId, setModuleId] = useState<ModuleId>('epi');
  const active = modules.find((module) => module.id === moduleId) ?? modules[0];

  return (
    <main className="appShell">
      <header className="topbar">
        <div className="brand">
          <span className="brandShield" aria-hidden="true"><span className="brandCheck">✓</span></span>
          <div><p className="brandKicker">MONITORAMENTO PREVENTIVO • SST</p><h1>Sentinela SST</h1></div>
        </div>
        <div className="privacyPill"><span className="privacyDot" />câmera local • sem armazenamento</div>
      </header>

      <section className="heroIntro">
        <div>
          <p className="sectionKicker">MÓDULO ATIVO</p>
          <h2>{active.label}</h2>
          <p>{active.description}</p>
        </div>
      </section>

      <section className="modulePanel" aria-label="Seleção do módulo">
        <label htmlFor="module-mobile" className="mobileSelectLabel">Selecione o módulo</label>
        <select id="module-mobile" className="moduleSelectMobile" value={moduleId} onChange={(event) => setModuleId(event.target.value as ModuleId)}>
          {modules.map((module) => <option key={module.id} value={module.id}>{module.label}</option>)}
        </select>
        <div className="moduleTabs">
          {modules.map((module) => (
            <button key={module.id} type="button" className={module.id === moduleId ? 'moduleTab active' : 'moduleTab'} onClick={() => setModuleId(module.id)} aria-pressed={module.id === moduleId}>
              <span>{module.eyebrow}</span><strong>{module.label}</strong>
            </button>
          ))}
        </div>
      </section>

      <VisionCamera moduleId={moduleId} />

      <footer className="footerNote">
        <span>Sentinela SST</span>
        <p>Ferramenta de apoio visual para atividades de Segurança do Trabalho. Os alertas auxiliam a observação e não substituem inspeções, avaliações formais ou procedimentos aplicáveis.</p>
      </footer>
    </main>
  );
}
