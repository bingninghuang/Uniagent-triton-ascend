/**
 * data-source.js — Extensible plugin registry for data providers.
 *
 * Each data source can contribute KPI cards, custom event types, and
 * custom tab panels. Built-in 'openhands' source is always registered.
 * Future RL / Agent sources can register without touching existing code.
 *
 * Usage:
 *   DataSources.register('my_source', {
 *     label: 'My Source',
 *     eventTypes: ['MyEvent'],
 *     fetchKPI: async (sessionId) => ({ ... }),
 *     renderKPICards: (container, kpiData) => { ... },
 *   });
 */

window.DataSources = (() => {
  const _registry = new Map();

  function register(name, descriptor) {
    _registry.set(name, {
      label: descriptor.label || name,
      eventTypes: descriptor.eventTypes || [],
      fetchKPI: descriptor.fetchKPI || null,
      renderKPICards: descriptor.renderKPICards || null,
      renderTabPanel: descriptor.renderTabPanel || null,
    });
  }

  function get(name) { return _registry.get(name); }
  function listAll() { return [..._registry.keys()]; }

  // Built-in OpenHands source
  register('openhands', {
    label: 'OpenHands Observability',
    eventTypes: [
      'ActionEvent', 'ObservationEvent', 'MessageEvent',
      'PauseEvent', 'AgentErrorEvent', 'ConversationStateUpdateEvent',
      'LLMCompletionLogEvent', 'StartupEvent', 'HeartbeatEvent',
      'EvaluateEvent', 'FinishEvent',
    ],
  });

  // ── Future RL source template (interface defined, not implemented) ──
  // register('rl_training', {
  //   label: 'RL Training Metrics',
  //   eventTypes: ['RewardEvent', 'PolicyUpdateEvent', 'EpisodeEndEvent'],
  //   fetchKPI: async (sessionId) => {
  //     return fetch(`/api/v1/sessions/${sessionId}/rl/metrics`).then(r => r.json());
  //   },
  //   renderKPICards: (container, data) => {
  //     // Renders reward curves, episode stats, etc.
  //   },
  // });

  return { register, get, listAll };
})();
