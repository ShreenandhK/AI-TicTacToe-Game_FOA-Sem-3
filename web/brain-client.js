/* brain-client.js: asks the Python brain (server.py) for the AI's move.
   TTTBrain.analyze(game, opts) returns a Promise of the same object the
   prototype's TTTEngine.analyze() returned. If the server can't be reached
   (e.g. the page was opened as a file), it falls back to the JS engine and
   marks the result engine: 'js-fallback'. */
(function () {
  const revive = (k, v) => (v === '+inf' ? Infinity : v === '-inf' ? -Infinity : v);
  let warned = false;
  async function analyze(game, opts) {
    try {
      const res = await fetch('/api/analyze', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ game, opts })
      });
      const text = await res.text();
      if (!res.ok) throw new Error(text);
      return JSON.parse(text, revive);
    } catch (e) {
      if (!warned) { console.warn('Python brain unavailable, using the JS fallback engine. Start it with: py server.py', e); warned = true; }
      return Object.assign(window.TTTEngine.analyze(game, opts), { engine: 'js-fallback' });
    }
  }
  window.TTTBrain = { analyze };
})();
