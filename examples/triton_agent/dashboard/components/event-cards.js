/**
 * event-cards.js — Per-event-type card renderers.
 *
 * List API returns: id, event_id, event_type, source, timestamp_str, received_at, summary, visible
 * Full payload (with `raw`) is loaded lazily in the drawer.
 *
 * Summary format (from events.py):
 *   ActionEvent:       "[Action:tool_name] thought text"
 *   ObservationEvent:  "[Observation:tool_name] result text"
 *   HeartbeatEvent:    "[Heartbeat] phase=X iteration=N running=True"
 *   StartupEvent:      "[Startup] description"
 *   SystemPromptEvent: "[SystemPromptEvent]"
 *   EvaluateEvent:     "[Evaluate] ..."
 *   FinishEvent:       "[Finish] ..."
 */

window.EventCards = (() => {

  // ── SVG Icons ─────────────────────────────────────────────────────────
  const ICONS = {
    ActionEvent:            `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/></svg>`,
    ObservationEvent:       `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>`,
    MessageEvent:           `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>`,
    SystemPromptEvent:      `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>`,
    StartupEvent:           `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>`,
    HeartbeatEvent:         `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>`,
    EvaluateEvent:          `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>`,
    FinishEvent:            `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 12l5 5L20 7"/></svg>`,
    AgentErrorEvent:        `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>`,
    LLMCompletionLogEvent:  `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a10 10 0 1 0 10 10"/><path d="M12 6v6l4 2"/><circle cx="18" cy="6" r="4"/></svg>`,
    PauseEvent:             `<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>`,
    ConversationStateUpdateEvent: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 2H3v16h5l3 3 3-3h7V2z"/><line x1="9" y1="9" x2="15" y2="9"/><line x1="9" y1="13" x2="12" y2="13"/></svg>`,
    default:                `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="16"/><line x1="8" y1="12" x2="16" y2="12"/></svg>`,
  };

  function getIcon(type) { return ICONS[type] || ICONS.default; }

  // ── Summary parser ────────────────────────────────────────────────────
  // Parses "[Prefix:tool_name] rest text" or "[Prefix] rest text"
  function parseSummary(summary) {
    if (!summary) return { prefix: '', tool: '', text: '' };
    const m = summary.match(/^\[([^\]]+)\]\s*(.*)/s);
    if (!m) return { prefix: '', tool: '', text: summary };
    const bracketContent = m[1];
    const text = m[2] || '';
    const colonIdx = bracketContent.indexOf(':');
    if (colonIdx >= 0) {
      return { prefix: bracketContent.slice(0, colonIdx), tool: bracketContent.slice(colonIdx + 1), text };
    }
    return { prefix: bracketContent, tool: '', text };
  }

  // Parse "key=val key=val ..." from heartbeat summary
  function parseKV(str) {
    const out = {};
    for (const m of str.matchAll(/(\w+)=(\S+)/g)) out[m[1]] = m[2];
    return out;
  }

  // ── Time formatting ───────────────────────────────────────────────────
  function fmtRelTime(ts) {
    if (!ts) return '';
    const date = typeof ts === 'number' ? new Date(ts * 1000) : new Date(ts);
    const diff = Math.floor((Date.now() - date.getTime()) / 1000);
    if (isNaN(diff)) return '';
    if (diff < 60) return `${diff}s ago`;
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return date.toLocaleDateString();
  }

  function fmtAbsTime(ts) {
    if (!ts) return '';
    const date = typeof ts === 'number' ? new Date(ts * 1000) : new Date(ts);
    if (isNaN(date)) {
      // Try parsing ISO string directly
      const d2 = new Date(ts);
      if (!isNaN(d2)) return d2.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
      return String(ts);
    }
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  }

  // _trunc removed: all text is now displayed in full without truncation
  function _esc(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

  // ── Body renderers — work from summary field (list API) ───────────────
  const _renderers = {
    ActionEvent(ev) {
      const { tool, text } = parseSummary(ev.summary);
      // Try extracting action detail from raw if available (drawer context)
      const raw = ev.raw || {};
      const action = raw.action || {};
      const actionCmd = action.command ? `${action.command}${action.path ? ' ' + action.path : ''}` : '';
      const thought = (raw.thought && raw.thought[0] && raw.thought[0].text) || text;
      return `
        <div class="ev-header-detail">
          ${tool ? `<span class="ev-tool-badge">${_esc(tool)}</span>` : ''}
          ${actionCmd ? `<span class="ev-action-cmd">${_esc(actionCmd)}</span>` : ''}
        </div>
        <div class="ev-thought">${_esc(thought)}</div>`;
    },
    ObservationEvent(ev) {
      const { tool, text } = parseSummary(ev.summary);
      const raw = ev.raw || {};
      const obs = raw.observation || {};
      const content = (obs.content && obs.content[0] && obs.content[0].text) || text;
      const isError = obs.is_error || false;
      return `
        <div class="ev-header-detail">
          ${tool ? `<span class="ev-tool-badge">${_esc(tool)}</span>` : ''}
          ${isError ? `<span class="ev-error-badge">ERROR</span>` : ''}
        </div>
        <div class="ev-obs-content${isError ? ' ev-summary--error' : ''}">${_esc(content)}</div>`;
    },
    HeartbeatEvent(ev) {
      const { text } = parseSummary(ev.summary);
      const kv = parseKV(text);
      return `
        <div class="ev-hb-row">
          ${kv.phase ? `<span class="ev-kv-chip phase-${kv.phase}">${kv.phase?.replace(/_/g,' ')}</span>` : ''}
          ${kv.iteration !== undefined ? `<span class="ev-kv-chip">iter ${kv.iteration}</span>` : ''}
          ${kv.running !== undefined ? `<span class="ev-kv-chip">${kv.running === 'True' ? '▶ running' : '⏹ stopped'}</span>` : ''}
        </div>`;
    },
    StartupEvent(ev) {
      const { text } = parseSummary(ev.summary);
      const model = ev.llm_model || '';
      const task = ev.task_preview || ev.task_instruction || '';
      return `
        <div class="ev-thought">${_esc(text)}</div>
        ${model ? `<div class="ev-header-detail"><span class="ev-tool-badge">${_esc(model)}</span></div>` : ''}
        ${task ? `<div class="ev-obs-content">${_esc(task)}</div>` : ''}`;
    },
    SystemPromptEvent(ev) {
      const raw = ev.raw || {};
      const tools = (raw.tools || []).map(t => t.title || t.name || '').filter(Boolean);
      const sysText = raw.system_prompt && raw.system_prompt.text ? raw.system_prompt.text : '';
      const dynText = raw.dynamic_context && raw.dynamic_context.text ? raw.dynamic_context.text : '';
      return `
        <div class="ev-thought">${sysText ? _esc(sysText) : 'System prompt'}</div>
        ${tools.length ? `<div class="ev-header-detail">${tools.map(t => `<span class="ev-tool-badge">${_esc(t)}</span>`).join('')}</div>` : ''}
        ${dynText ? `<div class="ev-obs-content">${_esc(dynText)}</div>` : ''}`;
    },
    EvaluateEvent(ev) {
      const { text } = parseSummary(ev.summary);
      const cost = ev.accumulated_cost != null ? `$${Number(ev.accumulated_cost).toFixed(4)}` : '';
      const calls = ev.total_llm_calls != null ? `${ev.total_llm_calls} LLM calls` : '';
      const iters = ev.iterations != null ? `${ev.iterations} iterations` : '';
      const status = ev.status || '';
      return `
        <div class="ev-header-detail">
          ${status ? `<span class="ev-tool-badge">${_esc(status)}</span>` : ''}
          ${cost ? `<span class="ev-kv-chip">${cost}</span>` : ''}
          ${calls ? `<span class="ev-kv-chip">${calls}</span>` : ''}
          ${iters ? `<span class="ev-kv-chip">${iters}</span>` : ''}
        </div>
        ${text ? `<div class="ev-thought">${_esc(text)}</div>` : ''}`;
    },
    FinishEvent(ev) {
      const { text } = parseSummary(ev.summary);
      const ok = ev.exit_code == null || ev.exit_code === 0;
      return `
        <div class="ev-header-detail">
          <span class="ev-tool-badge ev-exit-${ok ? 'ok' : 'err'}">exit ${ev.exit_code ?? 0}</span>
        </div>
        <div class="ev-thought${ok ? '' : ' ev-summary--error'}">${_esc(text || ev.reason || '')}</div>`;
    },
    AgentErrorEvent(ev) {
      const { text } = parseSummary(ev.summary);
      const raw = ev.raw || {};
      const err = raw.error || text || ev.summary || '';
      return `<div class="ev-obs-content ev-summary--error">${_esc(err)}</div>`;
    },
    LLMCompletionLogEvent(ev) {
      const { text } = parseSummary(ev.summary);
      const raw = ev.raw || {};
      const tokens = raw.usage ? `${raw.usage.total_tokens ?? '?'} tok` : '';
      const cost = raw.cost ? `$${Number(raw.cost).toFixed(5)}` : '';
      const model = raw.model || '';
      return `
        <div class="ev-header-detail">
          ${model ? `<span class="ev-tool-badge">${_esc(model.split('/').pop())}</span>` : ''}
          ${tokens ? `<span class="ev-kv-chip">${tokens}</span>` : ''}
          ${cost ? `<span class="ev-kv-chip">${cost}</span>` : ''}
        </div>
        ${text ? `<div class="ev-thought">${_esc(text)}</div>` : ''}`;
    },
    ConversationStateUpdateEvent(ev) {
      const { text } = parseSummary(ev.summary);
      return `<div class="ev-header-detail"><span class="ev-kv-chip">${_esc(text)}</span></div>`;
    },
    default(ev) {
      const { prefix, tool, text } = parseSummary(ev.summary);
      return `
        <div class="ev-header-detail">
          ${prefix ? `<span class="ev-tool-badge">${_esc(prefix)}</span>` : ''}
          ${tool ? `<span class="ev-kv-chip">${_esc(tool)}</span>` : ''}
        </div>
        ${text ? `<div class="ev-thought">${_esc(text)}</div>` : ''}`;
    },
  };

  // ── Public: render card DOM element ──────────────────────────────────
  function render(ev, opts = {}) {
    const type = ev.event_type || 'unknown';
    const renderer = _renderers[type] || _renderers.default;
    const isHidden = ev.visible === 0 || ev.visible === false;
    const isExit = type === 'FinishEvent' && ev.exit_code && ev.exit_code !== 0;

    const card = document.createElement('div');
    card.className = `ev-card ${type}${isExit ? ' exit-error' : ''}${isHidden ? ' ev-hidden' : ''}`;
    card.dataset.id = ev.id || '';
    card.dataset.type = type;
    card.dataset.visible = isHidden ? '0' : '1';

    // Prefer event timestamp_str (ISO from container), fall back to received_at (server unix)
    const ts = ev.timestamp_str || ev.received_at;
    const absTime = fmtAbsTime(ts);
    const relTime = fmtRelTime(ev.received_at || ts);

    const eyeIcon = isHidden
      ? `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"/><line x1="1" y1="1" x2="23" y2="23"/></svg>`
      : `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>`;

    card.innerHTML = `
      <div class="ev-icon">${getIcon(type)}</div>
      <div class="ev-body">
        <div class="ev-top-row">
          <span class="ev-type">${type.replace('Event','')}</span>
          <span class="ev-source">${_esc(ev.source || '')}</span>
          <span class="ev-time" title="${absTime}">${relTime || absTime}</span>
          ${ev.id ? `<span class="ev-id">#${ev.id}</span>` : ''}
        </div>
        ${renderer(ev)}
      </div>
      <div class="ev-actions">
        <button class="ev-vis-btn${isHidden ? ' hidden-state' : ''}" data-ev-id="${ev.id}" title="${isHidden ? 'Show' : 'Hide'}">${eyeIcon}</button>
      </div>`;

    card.addEventListener('click', (e) => {
      if (e.target.closest('.ev-vis-btn')) return;
      opts.onOpen && opts.onOpen(ev);
    });
    card.querySelector('.ev-vis-btn')?.addEventListener('click', (e) => {
      e.stopPropagation();
      opts.onToggleVisible && opts.onToggleVisible(ev, card);
    });
    return card;
  }

  // ── Time gap divider ──────────────────────────────────────────────────
  function renderGap(seconds) {
    const el = document.createElement('div');
    el.className = 'time-gap';
    const label = seconds < 60 ? `${seconds.toFixed(1)}s` : seconds < 3600 ? `${Math.floor(seconds/60)}m ${Math.floor(seconds%60)}s` : `${Math.floor(seconds/3600)}h`;
    el.innerHTML = `<div class="time-gap-line"></div><span class="time-gap-label">${label} gap</span><div class="time-gap-line"></div>`;
    return el;
  }

  // ── Collapsed group ───────────────────────────────────────────────────
  function renderGroup(type, count, onExpand) {
    const el = document.createElement('div');
    el.className = 'ev-group';
    el.innerHTML = `${getIcon(type)}<span class="ev-group-count">${count}×</span><span>${type.replace('Event','')} collapsed</span><span style="margin-left:auto;font-size:10px;opacity:.5">click to expand</span>`;
    el.addEventListener('click', onExpand);
    return el;
  }

  return { render, renderGap, renderGroup, getIcon, fmtRelTime, fmtAbsTime, parseSummary };
})();

// ── Extra inline styles for card internals ────────────────────────────────
const _s = document.createElement('style');
_s.textContent = `
.ev-top-row { display:flex; align-items:center; gap:6px; margin-bottom:4px; flex-wrap:wrap; }
.ev-type { font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:.04em; color:var(--ev-color,var(--gray)); flex-shrink:0; }
.ev-source { font-size:10px; color:var(--text-muted); }
.ev-time { font-size:10px; color:var(--text-muted); font-family:var(--font-mono); margin-left:auto; }
.ev-id { font-size:10px; color:var(--text-muted); font-family:var(--font-mono); }
.ev-header-detail { display:flex; flex-wrap:wrap; gap:4px; margin-bottom:3px; align-items:center; }
.ev-tool-badge { font-size:10px; font-weight:600; padding:1px 6px; background:color-mix(in srgb,var(--ev-color,var(--gray)) 15%,transparent); color:var(--ev-color,var(--gray)); border:1px solid color-mix(in srgb,var(--ev-color,var(--gray)) 30%,transparent); border-radius:3px; font-family:var(--font-mono); }
.ev-kv-chip { font-size:10px; padding:1px 6px; background:var(--bg-base); border:1px solid var(--border); border-radius:10px; color:var(--text-secondary); font-family:var(--font-mono); white-space:normal; word-break:break-word; }
.ev-action-cmd { font-size:10px; font-family:var(--font-mono); color:var(--cyan); background:var(--bg-base); padding:1px 6px; border-radius:3px; white-space:pre-wrap; word-break:break-word; }
.ev-thought { font-size:12px; color:var(--text-secondary); line-height:1.5; white-space:pre-wrap; word-break:break-word; }
.ev-obs-content { font-size:11px; color:var(--text-muted); font-family:var(--font-mono); white-space:pre-wrap; word-break:break-word; margin-top:2px; }
.ev-hb-row { display:flex; flex-wrap:wrap; gap:4px; }
.ev-kv-chip.phase-waiting_for_reply { background:var(--purple-dim); color:var(--purple); border-color:rgba(139,92,246,.3); }
.ev-kv-chip.phase-executing_command { background:var(--cyan-dim); color:var(--cyan); border-color:rgba(6,182,212,.3); }
.ev-kv-chip.phase-initializing { background:var(--blue-dim); color:var(--blue); border-color:rgba(59,130,246,.3); }
.ev-kv-chip.phase-finished { background:var(--green-dim); color:var(--green); border-color:rgba(16,185,129,.3); }
.ev-summary--error { color:var(--red) !important; }
.ev-error-badge { font-size:10px; font-weight:700; padding:1px 6px; background:var(--red-dim); color:var(--red); border:1px solid rgba(239,68,68,.3); border-radius:3px; }
.ev-exit-ok { background:var(--green-dim)!important; color:var(--green)!important; border-color:rgba(16,185,129,.3)!important; }
.ev-exit-err { background:var(--red-dim)!important; color:var(--red)!important; border-color:rgba(239,68,68,.3)!important; }
/* HeartbeatEvent compact */
.ev-card.HeartbeatEvent { padding:5px 10px; }
.ev-card.HeartbeatEvent .ev-icon { width:20px; height:20px; }
`;
document.head.appendChild(_s);
