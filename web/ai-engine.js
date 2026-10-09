/* ai-engine.js — the replaceable AI for AI Tic-Tac-Toe (Group 5).
   The UI only calls TTTEngine.analyze(game, opts) and renders the returned object.
   Swap analyze() for the real implementation; keep the return shape. Cells are 0-8 (shown 1-9). */
(function () {
  const LINES = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]];
  const LINE_NAMES = ['top row 1-2-3','middle row 4-5-6','bottom row 7-8-9','left column 1-4-7','middle column 2-5-8','right column 3-6-9','diagonal 1-5-9','diagonal 3-5-7'];
  const ORDER = [4,0,2,6,8,1,3,5,7];
  const CELL_NAMES = ['top-left corner','top edge','top-right corner','left edge','centre','right edge','bottom-left corner','bottom edge','bottom-right corner'];
  const other = p => (p === 'X' ? 'O' : 'X');
  class Abort extends Error {}

  function lineOf(c) { for (const L of LINES) if (c.includes(L[0]) && c.includes(L[1]) && c.includes(L[2])) return L; return null; }
  function lineName(L) { const i = LINES.findIndex(x => x[0] === L[0] && x[1] === L[1] && x[2] === L[2]); return LINE_NAMES[i] || ''; }
  function keyOf(p) { return p.X.join('') + '|' + p.O.join('') + '|' + p.toMove; }
  function empties(p) { const o = p.X.concat(p.O); return ORDER.filter(c => o.indexOf(c) < 0); }
  function play(p, cell, mode) {
    const s = p.toMove; const mine = p[s].concat([cell]); let vanished = null;
    if (mode === 'infinite' && mine.length > 3) vanished = mine.shift();
    return { pos: { X: s === 'X' ? mine : p.X, O: s === 'O' ? mine : p.O, toMove: other(s) }, vanished, line: lineOf(mine), mover: s };
  }
  function winVal(d, mode) { return mode === 'classic' ? 10 - d : 10 - Math.ceil(d / 2); }
  function heur(p, ai) {
    const a = p[ai], h = p[other(ai)]; let s = 0;
    for (const L of LINES) { let ca = 0, ch = 0; for (const c of L) { if (a.includes(c)) ca++; else if (h.includes(c)) ch++; } if (ch === 0) s += ca * ca; if (ca === 0) s -= ch * ch; }
    return Math.max(-3, Math.min(3, Math.round(s / 4)));
  }
  function child(p, cell, ctx, ply) {
    const r = play(p, cell, ctx.mode); let term = null, rep = false;
    if (r.line) term = (r.mover === ctx.ai ? 1 : -1) * winVal(ply + 1, ctx.mode);
    else if (ctx.mode === 'infinite') { if ((ctx.seen.get(keyOf(r.pos)) || 0) >= 2) { term = (r.mover === ctx.ai ? -1 : 1) * winVal(ply + 1, ctx.mode); rep = true; } }
    else if (r.pos.X.length + r.pos.O.length === 9) term = 0;
    return { r, term, rep };
  }
  // Minimax with alpha-beta. Records the first two plies (rec) for the search-tree view.
  function ab(p, ply, alpha, beta, ctx, rec) {
    ctx.nodes++;
    if (ctx.deadline && (ctx.nodes & 1023) === 0 && performance.now() > ctx.deadline) throw new Abort();
    if (ply >= ctx.depth) return heur(p, ctx.ai);
    const maxing = p.toMove === ctx.ai, moves = empties(p); let best = maxing ? -Infinity : Infinity;
    for (let i = 0; i < moves.length; i++) {
      const cell = moves[i], e = child(p, cell, ctx, ply); let node = null;
      if (rec) { node = { move: cell, player: maxing ? 'MAX' : 'MIN', alpha, beta, value: null, pruned: false, cutoff: false, terminal: e.term !== null, repetition: e.rep, vanished: e.r.vanished, visitOrder: ++ctx.visit, doneOrder: 0, subtree: 0, children: ply < 1 ? [] : null }; rec.push(node); }
      const n0 = ctx.nodes; let v;
      if (e.term !== null) { ctx.nodes++; v = e.term; }
      else {
        let k = null; if (ctx.mode === 'infinite') { k = keyOf(e.r.pos); ctx.seen.set(k, (ctx.seen.get(k) || 0) + 1); }
        if (node && ply + 1 >= ctx.depth) node.cutoff = true;
        v = ab(e.r.pos, ply + 1, alpha, beta, ctx, node ? node.children : null);
        if (k) ctx.seen.set(k, ctx.seen.get(k) - 1);
      }
      if (node) { node.value = v; node.subtree = ctx.nodes - n0; node.doneOrder = ++ctx.visit; }
      if (maxing) { if (v > best) best = v; if (best > alpha) alpha = best; } else { if (v < best) best = v; if (best < beta) beta = best; }
      if (alpha >= beta) {
        ctx.cuts++;
        if (rec) { const at = ++ctx.visit; for (let j = i + 1; j < moves.length; j++) rec.push({ move: moves[j], player: maxing ? 'MAX' : 'MIN', alpha, beta, value: null, pruned: true, cutoff: false, terminal: false, visitOrder: at, doneOrder: at, subtree: 0, children: null }); }
        break;
      }
    }
    return best;
  }
  const mmMemo = new Map();
  // Plain minimax node count. Classic: exact. Infinite: full-width estimate (ignores early wins).
  function mmCount(p, mode, d) {
    if (mode === 'classic') {
      const k = keyOf(p) + '@' + d; if (mmMemo.has(k)) return mmMemo.get(k);
      let n = 1;
      if (d > 0) for (const c of empties(p)) { const r = play(p, c, mode); if (r.line || r.pos.X.length + r.pos.O.length === 9 || d === 1) n += 1; else n += mmCount(r.pos, mode, d - 1); }
      mmMemo.set(k, n); return n;
    }
    let nm = p[p.toMove].length, no = p[other(p.toMove)].length, n = 1, w = 1;
    for (let i = 0; i < d; i++) { w *= 9 - nm - no; n += w; nm = Math.min(3, nm + 1); [nm, no] = [no, nm]; }
    return n;
  }
  function losingCells(pos, history, mode) {
    if (mode !== 'infinite') return [];
    return empties(pos).filter(c => { const r = play(pos, c, mode); return !r.line && ((history || {})[keyOf(r.pos)] || 0) >= 2; });
  }
  function reason(p, cell, ai, mode) {
    const h = other(ai), r = play(p, cell, mode);
    if (r.line) return 'Completes the ' + lineName(r.line);
    const rb = play({ X: p.X, O: p.O, toMove: h }, cell, mode);
    if (rb.line) return 'Blocks your ' + lineName(rb.line);
    const ap = { X: r.pos.X, O: r.pos.O, toMove: ai };
    const threats = empties(ap).filter(c => play(ap, c, mode).line);
    if (threats.length >= 2) return 'Creates a fork: two ways to win';
    if (threats.length === 1) return 'Threatens to win on cell ' + (threats[0] + 1);
    if (mode === 'infinite' && r.vanished != null) return 'Repositions while cell ' + (r.vanished + 1) + ' vanishes';
    return 'Takes the ' + CELL_NAMES[cell];
  }

  /* game: { mode, board:{X,O} (oldest first), toMove, history:{positionKey:count} }
     opts: { ai:'X'|'O', difficulty:'unbeatable'|'normal', maxDepth, slipChance, forceMove } */
  function analyze(game, opts) {
    opts = opts || {};
    const mode = game.mode, ai = opts.ai, diff = opts.difficulty || 'unbeatable';
    const pos = { X: game.board.X.slice(), O: game.board.O.slice(), toMove: game.toMove };
    const hist = new Map(Object.entries(game.history || {}));
    const fullDepth = mode === 'classic' ? 9 : (opts.maxDepth || 10);
    const depthLimit = diff === 'normal' ? 2 : fullDepth;
    const t0 = performance.now(), hard = t0 + 700;
    let last = null; const deepening = [];
    for (let d = mode === 'infinite' ? 1 : depthLimit; d <= depthLimit; d++) {
      const rec = [], ts = performance.now();
      const ctx = { mode, ai, depth: d, seen: new Map(hist), nodes: 0, visit: 0, cuts: 0, deadline: d > 1 ? hard : 0 };
      try { const v = ab(pos, 0, -Infinity, Infinity, ctx, rec); last = { v, ctx, rec, d }; deepening.push({ depth: d, nodes: ctx.nodes, ms: +(performance.now() - ts).toFixed(1) }); }
      catch (e) { if (e instanceof Abort) break; throw e; }
      if (mode === 'infinite' && Math.abs(last.v) >= 5) break;
      if (performance.now() - t0 > 260) break;
    }
    const timeMs = Math.max(0.1, performance.now() - t0);
    const scoreDepth = mode === 'classic' ? 9 : diff === 'normal' ? Math.min(6, fullDepth) : last.d;
    const sdl = performance.now() + 450, moves = empties(pos), aiMarks = pos[ai];
    const vanishes = mode === 'infinite' && aiMarks.length === 3 ? aiMarks[0] : null;
    const candidates = moves.map(cell => {
      const sctx = { mode, ai, depth: scoreDepth, seen: new Map(hist), nodes: 0, visit: 0, cuts: 0, deadline: sdl };
      const e = child(pos, cell, sctx, 0); let score;
      if (e.term !== null) score = e.term;
      else {
        try { if (mode === 'infinite') { const k = keyOf(e.r.pos); sctx.seen.set(k, (sctx.seen.get(k) || 0) + 1); } score = ab(e.r.pos, 1, -Infinity, Infinity, sctx, null); }
        catch (err) { const n = last.rec.find(x => x.move === cell); score = n && n.value != null ? n.value : 0; }
      }
      return { cell, score, exact: mode === 'classic' || Math.abs(score) >= 5, vanishes };
    });
    let optimal = candidates[0]; for (const c of candidates) if (c.score > optimal.score) optimal = c;
    const rootKids = last.rec.filter(n => !n.pruned); let abBest = rootKids[0];
    for (const n of rootKids) if (n.value > abBest.value) abBest = n;
    let chosen = abBest.move;
    if (opts.forceMove != null) chosen = opts.forceMove;
    else if (diff === 'normal') {
      const rng = opts.rng || Math.random;
      const worse = candidates.filter(c => c.score < optimal.score).sort((x, y) => y.score - x.score);
      if (worse.length && rng() < (opts.slipChance != null ? opts.slipChance : 0.3)) chosen = (rng() < 0.7 ? worse[0] : worse[Math.floor(rng() * worse.length)]).cell;
    }
    const cc = candidates.find(c => c.cell === chosen), value = cc.score, exact = cc.exact;
    const forcedWinIn = exact && value > 0 ? (mode === 'classic' ? 10 - value : 2 * (10 - value) - 1) : null;
    const forcedLossIn = exact && value < 0 ? (mode === 'classic' ? 10 + value : 2 * (10 + value)) : null;
    const pv = [chosen]; const cn = last.rec.find(n => n.move === chosen);
    if (cn && !cn.terminal && cn.children) {
      const kids = cn.children.filter(k => !k.pruned);
      if (kids.length) { let b = kids[0]; for (const k of kids) if (k.value < b.value) b = k; pv.push(b.move); }
    }
    const nodesAB = last.ctx.nodes, nodesMM = Math.max(mmCount(pos, mode, last.d), nodesAB + 1);
    return {
      mode, board: { X: pos.X, O: pos.O }, toMove: pos.toMove, ai,
      vanishesNext: mode === 'infinite' ? { X: pos.X.length === 3 ? pos.X[0] : null, O: pos.O.length === 3 ? pos.O[0] : null } : null,
      chosenMove: chosen, optimalMove: optimal.cell, optimalValue: optimal.score, value, slip: value < optimal.score,
      reason: reason(pos, chosen, ai, mode), exact, forcedWinIn, forcedLossIn, candidates,
      nodes: { minimax: nodesMM, alphaBeta: nodesAB }, depthLimit, depthReached: last.d,
      timeMs: +timeMs.toFixed(1), minimaxTimeMs: +(timeMs * nodesMM / nodesAB).toFixed(1),
      principalVariation: pv, deepening,
      repetition: { count: hist.get(keyOf(pos)) || 1, losingCells: losingCells(pos, game.history, mode) },
      tree: { move: null, player: 'MAX', value: last.v, alpha: -Infinity, beta: Infinity, pruned: false, cutoff: false, visitOrder: 0, doneOrder: last.ctx.visit + 1, children: last.rec },
      maxStep: last.ctx.visit + 1, cuts: last.ctx.cuts
    };
  }
  window.TTTEngine = { LINES, CELL_NAMES, keyOf, empties, play, lineOf, lineName, analyze, losingCells };
})();
