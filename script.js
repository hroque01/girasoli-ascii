'use strict';

/*
 * Girasoli — un bouquet 3D fatto di caratteri.
 *
 * Nessuna libreria. La scena nasce da superfici parametriche (disco dei semi, petali,
 * coppa, brattee, steli, foglie, carta, nastro). A ogni frame:
 *   1. i vertici vengono proiettati sulla griglia dei caratteri;
 *   2. i triangoli vengono rasterizzati con uno z-buffer: una cella = un carattere;
 *   3. ogni cella visibile viene illuminata e diventa un carattere + un colore;
 *   4. i caratteri si disegnano in bianco da un atlante e si colorano tutti insieme
 *      (composizione "source-in" con un'immagine grande una cella per pixel).
 */
(() => {
  // ── utilità ─────────────────────────────────────────────────────────────
  const TAU = Math.PI * 2;
  const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
  const lerp = (a, b, t) => a + (b - a) * t;
  const smooth = (a, b, x) => {
    const t = clamp((x - a) / (b - a), 0, 1);
    return t * t * (3 - 2 * t);
  };
  const wrapPi = (a) => Math.atan2(Math.sin(a), Math.cos(a));

  const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const len = (a) => Math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
  const norm = (a) => {
    const l = len(a) || 1;
    return [a[0] / l, a[1] / l, a[2] / l];
  };
  const mix3 = (a, b, t) => [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];

  // generatore pseudo-casuale con seme fisso: il bouquet è sempre lo stesso
  function mulberry32(seed) {
    let s = seed >>> 0;
    return () => {
      s = (s + 0x6d2b79f5) >>> 0;
      let t = s;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const random = mulberry32(20261123);
  const rand = (a, b) => a + (b - a) * random();

  // ── semi: spirale di Vogel (angolo aureo) ───────────────────────────────
  const GOLDEN = Math.PI * (3 - Math.sqrt(5));
  const SEED_GRID = 48;

  function makeSeeds(N) {
    const G = SEED_GRID;
    const xs = new Float32Array(N);
    const ys = new Float32Array(N);
    const cells = new Int32Array(N);
    const start = new Int32Array(G * G + 1);
    const cellOf = (x, y) =>
      clamp(Math.floor((y + 1) * 0.5 * G), 0, G - 1) * G + clamp(Math.floor((x + 1) * 0.5 * G), 0, G - 1);
    for (let n = 0; n < N; n++) {
      const r = Math.sqrt((n + 0.5) / N);
      const th = n * GOLDEN;
      xs[n] = r * Math.cos(th);
      ys[n] = r * Math.sin(th);
      cells[n] = cellOf(xs[n], ys[n]);
      start[cells[n] + 1]++;
    }
    for (let c = 0; c < G * G; c++) start[c + 1] += start[c];
    const cursor = start.slice(0, G * G);
    const items = new Int32Array(N);
    for (let n = 0; n < N; n++) items[cursor[cells[n]]++] = n;
    return { N, xs, ys, start, items, spacing: Math.sqrt(Math.PI / N) };
  }

  // distanza dal seme più vicino, in coordinate del disco (raggio 1)
  function seedDistance(S, x, y) {
    const G = SEED_GRID;
    const cx = clamp(Math.floor((x + 1) * 0.5 * G), 0, G - 1);
    const cy = clamp(Math.floor((y + 1) * 0.5 * G), 0, G - 1);
    let best = 1e9;
    for (let gy = Math.max(0, cy - 1); gy <= Math.min(G - 1, cy + 1); gy++) {
      for (let gx = Math.max(0, cx - 1); gx <= Math.min(G - 1, cx + 1); gx++) {
        const c = gy * G + gx;
        for (let q = S.start[c], e = S.start[c + 1]; q < e; q++) {
          const n = S.items[q];
          const dx = S.xs[n] - x;
          const dy = S.ys[n] - y;
          const d = dx * dx + dy * dy;
          if (d < best) best = d;
        }
      }
    }
    return Math.sqrt(best);
  }

  // ── costruzione della mesh ──────────────────────────────────────────────
  const M = { DISK: 0, PETAL: 1, BOWL: 2, BRACT: 3, STEM: 4, CUT: 5, LEAF: 6, PAPER: 7, TISSUE: 8, RIBBON: 9 };
  //                DISK  PETAL BOWL  BRACT STEM  CUT   LEAF  PAPER TISSUE RIBBON
  const MAT_KS = [0.08, 0.16, 0.05, 0.06, 0.12, 0.0, 0.2, 0.04, 0.05, 0.5];
  const MAT_SH = [16, 22, 10, 10, 24, 1, 20, 8, 8, 34];
  const NONE = 255; // oggetto non cliccabile

  const vP = [], vN = [], vA = [], vO = [], vW = [], vPh = [];
  const vS = [], vD = [], vT = []; // entrata: posizione di partenza, ritardo, durata
  const tI = [], tM = [], tO = [];

  const Y_BIND = -0.32; // quota della legatura
  const swayWeight = (y) => {
    const t = clamp((y - Y_BIND) / 2.8, 0, 1.25);
    return t * t;
  };

  function normalAt(fn, u, v) {
    const e = 1e-3;
    for (const s of [0, 0.01, -0.01, 0.04, -0.04]) {
      const uu = clamp(u + s, 0, 1);
      const du = sub(fn(Math.min(1, uu + e), v), fn(Math.max(0, uu - e), v));
      const dv = sub(fn(uu, v + e), fn(uu, v - e));
      const n = cross(du, dv);
      const l = len(n);
      if (l > 1e-14) return [n[0] / l, n[1] / l, n[2] / l];
    }
    return [0, 1, 0];
  }

  // superficie parametrica fn(u, v) con u, v in [0, 1], griglia nu × nv
  function surface(nu, nv, fn, o) {
    const base = vP.length / 3;
    const obj = o.obj === undefined ? NONE : o.obj;
    for (let i = 0; i <= nu; i++) {
      const u = i / nu;
      for (let j = 0; j <= nv; j++) {
        const v = j / nv;
        const p = fn(u, v);
        const n = normalAt(fn, u, v);
        const at = o.attr ? o.attr(u, v) : [u, v];
        vP.push(p[0], p[1], p[2]);
        vN.push(n[0], n[1], n[2]);
        vA.push(at[0], at[1]);
        vO.push(o.ao ? o.ao(u, v, p) : 1);
        vW.push(o.sway === false ? 0 : swayWeight(p[1]));
        vPh.push(o.phase || 0);
        const from = o.from ? o.from(u, v, p) : p;
        vS.push(from[0], from[1], from[2]);
        vD.push(o.delay ? o.delay(u, v, p) : 0);
        vT.push(o.dur || 0.6);
      }
    }
    for (let i = 0; i < nu; i++) {
      for (let j = 0; j < nv; j++) {
        const a = base + i * (nv + 1) + j;
        const b = a + nv + 1;
        tI.push(a, b, b + 1, a, b + 1, a + 1);
        tM.push(o.mat, o.mat);
        tO.push(obj, obj);
      }
    }
  }

  // lama curva (petalo, brattea, foglia): parte da `base`, avanza lungo `dir`
  // incurvandosi verso `up`; `shape(u)` dà la mezza larghezza relativa.
  function blade(o) {
    const { base, dir, up, side, length, width, e0, e1, shape } = o;
    const cup = o.cup || 0;
    const bend = o.bend || 0;
    const serr = o.serr || 0;
    const k = e1 / length;
    return (u, v) => {
      const s = u * length;
      const e = e0 + k * s;
      let rho, z;
      if (Math.abs(k) < 1e-6) {
        rho = s * Math.cos(e0);
        z = s * Math.sin(e0);
      } else {
        rho = (Math.sin(e) - Math.sin(e0)) / k;
        z = (Math.cos(e0) - Math.cos(e)) / k;
      }
      const across = v * 2 - 1;
      let w = width * shape(clamp(u, 0, 1));
      if (serr) w *= 1 + serr * Math.sin(u * 46);
      const lat = across * w + bend * s * s;
      const lift = cup * across * across * w;
      const se = Math.sin(e);
      const ce = Math.cos(e);
      return [
        base[0] + dir[0] * rho + up[0] * z + side[0] * lat + (ce * up[0] - se * dir[0]) * lift,
        base[1] + dir[1] * rho + up[1] * z + side[1] * lat + (ce * up[1] - se * dir[1]) * lift,
        base[2] + dir[2] * rho + up[2] * z + side[2] * lat + (ce * up[2] - se * dir[2]) * lift,
      ];
    };
  }

  const petalShape = (u) =>
    (0.22 + 0.78 * Math.sin(Math.PI * Math.min(1, Math.pow(u, 0.72)))) * Math.pow(Math.max(0, 1 - u), 0.38);
  const bractShape = (u) => Math.pow(Math.max(0, 1 - u), 0.85) * (0.75 + 0.25 * Math.sin(Math.PI * u));
  const leafShape = (u) =>
    (0.06 + 0.94 * Math.sin(Math.PI * Math.pow(u, 0.62)) * smooth(0.02, 0.18, u)) * Math.pow(Math.max(0, 1 - u), 0.22);

  function bezier(p0, p1, p2, p3, t) {
    const s = 1 - t;
    const a = s * s * s, b = 3 * s * s * t, c = 3 * s * t * t, d = t * t * t;
    return [
      a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0],
      a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1],
      a * p0[2] + b * p1[2] + c * p2[2] + d * p3[2],
    ];
  }

  // tubo lungo una curva, con riferimenti a trasporto parallelo (niente torsioni)
  function makeTube(curve, K) {
    const P = [], T = [], N = [];
    for (let i = 0; i <= K; i++) P.push(curve(i / K));
    for (let i = 0; i <= K; i++) T.push(norm(sub(P[Math.min(K, i + 1)], P[Math.max(0, i - 1)])));
    N.push(norm(cross(T[0], Math.abs(T[0][1]) < 0.9 ? [0, 1, 0] : [1, 0, 0])));
    for (let i = 1; i <= K; i++) {
      const prev = N[i - 1];
      N.push(norm(sub(prev, mul(T[i], dot(prev, T[i])))));
    }
    return (t) => {
      const f = clamp(t, 0, 1) * K;
      const i = Math.min(K - 1, Math.floor(f));
      const s = f - i;
      const p = mix3(P[i], P[i + 1], s);
      const tt = norm(mix3(T[i], T[i + 1], s));
      let n = mix3(N[i], N[i + 1], s);
      n = norm(sub(n, mul(tt, dot(n, tt))));
      return { p, n, b: cross(tt, n) };
    };
  }

  // ── carta del bouquet ───────────────────────────────────────────────────
  const PAPER_Y0 = -0.62;
  const paperTopY = (phi) => 1.22 + 1.32 * (1 - Math.cos(phi)) / 2 + 0.07 * Math.sin(5 * phi + 1);
  const paperTopR = (phi) => 1.42 + 0.34 * (1 - Math.cos(phi)) / 2;

  function paperPoint(u, phi, inner) {
    const top = paperTopY(phi) + (inner ? 0.1 : 0);
    const y = PAPER_Y0 + (top - PAPER_Y0) * u;
    const r0 = inner ? 0.17 : 0.2;
    const rTop = paperTopR(phi) * (inner ? 0.86 : 1);
    let r = r0 + (rTop - r0) * Math.pow(u, 1.35);
    r *= 1 + (inner ? 0.04 : 0.075) * u * Math.sin((inner ? 9 : 7) * phi + (inner ? 2.2 : 0.4));
    return [r * Math.sin(phi), y, r * Math.cos(phi)];
  }

  function paperRadiusAt(y, phi) {
    const u = clamp((y - PAPER_Y0) / (paperTopY(phi) - PAPER_Y0), 0, 1);
    const p = paperPoint(u, phi, false);
    return Math.hypot(p[0], p[2]);
  }

  // ── girasoli ────────────────────────────────────────────────────────────
  const flowers = [];

  function buildHead(fl, petalLen) {
    const { C, F, Rd } = fl;
    let U = cross(F, [0, 1, 0]);
    U = len(U) < 1e-3 ? [1, 0, 0] : norm(U);
    const V = cross(F, U);
    const at = (x, y, z) => [
      C[0] + U[0] * x + V[0] * y + F[0] * z,
      C[1] + U[1] * x + V[1] * y + F[1] * z,
      C[2] + U[2] * x + V[2] * y + F[2] * z,
    ];
    const radial = (th) => add(mul(U, Math.cos(th)), mul(V, Math.sin(th)));
    const tangent = (th) => add(mul(U, -Math.sin(th)), mul(V, Math.cos(th)));
    const obj = fl.index;
    const phase = fl.phase;
    const rot = rand(0, TAU);
    const D = Rd * 0.42;
    const t0 = fl.t0;
    const shrink = (k) => (u, v, p) => add(C, mul(sub(p, C), k));

    // disco dei semi, appena bombato
    surface(16, 72, (u, v) => {
      const th = v * TAU;
      const r = u * Rd;
      return at(r * Math.cos(th), r * Math.sin(th), 0.07 * Rd * (1 - u * u));
    }, {
      mat: M.DISK, obj, phase, attr: (u, v) => [u * Math.cos(v * TAU + rot), u * Math.sin(v * TAU + rot)],
      from: shrink(0.12), delay: () => t0 + 0.15, dur: 0.7,
    });

    // coppa verde sul retro
    surface(8, 72, (u, v) => {
      const th = v * TAU;
      const r = u * Rd;
      return at(r * Math.cos(th), r * Math.sin(th), -D * Math.pow(Math.max(0, 1 - u * u), 0.55));
    }, { mat: M.BOWL, obj, phase, from: shrink(0.15), delay: () => t0, dur: 0.6 });

    // petali: 34 esterni + 21 interni (due numeri di Fibonacci)
    const layers = [
      { n: 34, len: petalLen, wid: petalLen * 0.15, e0: 0.2, e1: -0.42, cup: 0.4, rb: Rd * 0.93, z0: -0.012, off: 0, ao0: 0.55, lag: 0.1 },
      { n: 21, len: petalLen * 0.84, wid: petalLen * 0.17, e0: 0.48, e1: -0.38, cup: 0.45, rb: Rd * 0.93, z0: -0.008, off: 0.5, ao0: 0.6, lag: 0 },
    ];
    for (const L of layers) {
      for (let k = 0; k < L.n; k++) {
        const th = ((k + L.off + rand(-0.18, 0.18)) * TAU) / L.n;
        const geo = {
          base: at(L.rb * Math.cos(th), L.rb * Math.sin(th), L.z0),
          dir: radial(th),
          up: F,
          side: tangent(th),
          length: L.len * rand(0.88, 1.08),
          width: L.wid * rand(0.85, 1.12),
          e0: L.e0 + rand(-0.12, 0.12),
          e1: L.e1 + rand(-0.15, 0.15),
          cup: L.cup,
          bend: rand(-0.25, 0.25),
          shape: petalShape,
        };
        // bocciolo: lo stesso petalo chiuso in avanti, da cui parte la fioritura
        const bud = blade({ ...geo, length: geo.length * 0.6, width: geo.width * 0.7, e0: 1.3, e1: 0.2, cup: 0.7, bend: 0 });
        const delay = t0 + 0.35 + L.lag + k * 0.004;
        surface(8, 4, blade(geo), {
          mat: M.PETAL, obj, phase, attr: (u, v) => [u, v * 2 - 1], ao: (u) => lerp(L.ao0, 1, smooth(0, 0.45, u)),
          from: (u, v) => bud(u, v), delay: () => delay, dur: 1.3,
        });
      }
      fl.petals += L.n;
    }

    // brattee verdi dietro i petali
    const rows = [
      { n: 26, rho: 0.98, z: -0.02 * Rd, len: Rd * 0.62, wid: Rd * 0.12, e0: -0.55, e1: 0.75, off: 0 },
      { n: 21, rho: 0.78, z: -D * Math.pow(1 - 0.78 * 0.78, 0.55), len: Rd * 0.55, wid: Rd * 0.13, e0: -1.05, e1: 0.6, off: 0.4 },
    ];
    for (const R of rows) {
      for (let k = 0; k < R.n; k++) {
        const th = ((k + rand(-0.2, 0.2)) * TAU) / R.n + R.off;
        surface(5, 2, blade({
          base: at(R.rho * Rd * Math.cos(th), R.rho * Rd * Math.sin(th), R.z),
          dir: radial(th),
          up: F,
          side: tangent(th),
          length: R.len * rand(0.85, 1.1),
          width: R.wid,
          e0: R.e0 + rand(-0.15, 0.15),
          e1: R.e1,
          cup: 0.3,
          bend: rand(-0.3, 0.3),
          shape: bractShape,
        }), { mat: M.BRACT, obj, phase, attr: (u, v) => [u, v * 2 - 1], from: shrink(0.3), delay: () => t0, dur: 0.7 });
      }
    }
  }

  function buildStem(fl, E, Bo, Hb) {
    const F = fl.F;
    const low = [E, mix3(E, Bo, 0.4), sub(Bo, [0, 0.5, 0]), Bo];
    const high = [Bo, add(Bo, [0, 0.85, 0]), sub(Hb, mul(F, 0.75)), Hb];
    const J = 0.45;
    const curve = (t) => (t < J ? bezier(...low, t / J) : bezier(...high, (t - J) / (1 - J)));
    const tube = makeTube(curve, 120);
    const ring = (f, r, th) => [
      f.p[0] + (f.n[0] * Math.cos(th) + f.b[0] * Math.sin(th)) * r,
      f.p[1] + (f.n[1] * Math.cos(th) + f.b[1] * Math.sin(th)) * r,
      f.p[2] + (f.n[2] * Math.cos(th) + f.b[2] * Math.sin(th)) * r,
    ];
    surface(44, 8, (u, v) => ring(tube(u), lerp(0.04, 0.052, clamp(u, 0, 1)), v * TAU), {
      mat: M.STEM,
      obj: fl.index,
      phase: fl.phase,
      delay: (u) => fl.grow + u * 1.05,
      dur: 0.22,
      ao: (u, v, p) => (p[1] < Y_BIND + 0.05 ? 0.85 : lerp(0.45, 1, smooth(Y_BIND + 0.3, Y_BIND + 2.4, p[1]))),
    });
    // taglio alla base dello stelo
    const f0 = tube(0);
    surface(2, 8, (u, v) => ring(f0, 0.04 * u, v * TAU), { mat: M.CUT, obj: fl.index, phase: fl.phase, delay: () => fl.grow, dur: 0.22 });
  }

  // disposizione: una cupola inclinata verso chi guarda
  const AXIS = norm([0, Math.cos(0.35), Math.sin(0.35)]);
  const E1 = [1, 0, 0];
  const E2 = cross(AXIS, E1);
  const ORIGIN = [0, 0.45, -0.3];
  const SPHERE = 2.2;
  const SEED_COUNTS = [987, 840, 910, 780, 950, 860, 820];

  const headSpecs = [{ pol: 0, az: 0, Rd: 0.37, pl: 0.47 }];
  for (let k = 0; k < 6; k++) {
    headSpecs.push({
      pol: rand(0.74, 0.82),
      az: 0.26 + (k * TAU) / 6 + rand(-0.1, 0.1),
      Rd: rand(0.3, 0.35),
      pl: rand(0.38, 0.45),
    });
  }

  headSpecs.forEach((h, i) => {
    const ringDir = add(mul(E1, Math.cos(h.az)), mul(E2, Math.sin(h.az)));
    const d = norm(add(mul(AXIS, Math.cos(h.pol)), mul(ringDir, Math.sin(h.pol))));
    const C = add(ORIGIN, mul(d, SPHERE));
    const F = norm(add(d, [0, 0.15, 0.55])); // girati verso chi guarda
    // "su" e "destra" della scritta sul disco, per chi guarda il fiore di fronte
    const TU = norm(sub([0, 1, 0], mul(F, F[1])));
    const fl = {
      index: i, C, F, Rd: h.Rd, R: h.Rd + h.pl, seeds: makeSeeds(SEED_COUNTS[i]), petals: 0, phase: rand(0, TAU),
      TU, TR: cross(TU, F), grow: 0.05 + 0.04 * i, t0: 1.25 + 0.12 * i,
    };
    flowers.push(fl);
    buildHead(fl, h.pl);

    const off = i === 0 ? [0, 0, 0] : mul(norm([ringDir[0], 0, ringDir[2]]), 0.075);
    const Bo = add([0, Y_BIND, 0], off);
    const E = [Bo[0] * 2.6 + rand(-0.03, 0.03), -1.95 + rand(-0.05, 0.05), Bo[2] * 2.6 + rand(-0.03, 0.03)];
    const Hb = sub(C, mul(F, h.Rd * 0.42 * 0.9));
    buildStem(fl, E, Bo, Hb);
  });

  // foglie che sporgono dal bordo della carta
  for (const phi of [-1.8, -0.85, 0.85, 1.8, Math.PI - 0.3]) {
    const a = phi + rand(-0.12, 0.12);
    const dir = [Math.sin(a), 0, Math.cos(a)];
    const up = [0, 1, 0];
    const r = paperTopR(phi) * 0.6;
    const leafBase = [Math.sin(phi) * r, paperTopY(phi) - 0.24, Math.cos(phi) * r];
    const leafDelay = 2.0 + (phi + 2) * 0.1;
    surface(18, 8, blade({
      base: leafBase,
      dir,
      up,
      side: cross(up, dir),
      length: rand(0.85, 1.0),
      width: rand(0.27, 0.32),
      e0: 1.0,
      e1: -2.1,
      cup: 0.32,
      bend: rand(-0.2, 0.2),
      serr: 0.04,
      shape: leafShape,
    }), {
      mat: M.LEAF, phase: rand(0, TAU), attr: (u, v) => [u, v * 2 - 1], ao: (u) => lerp(0.5, 1, smooth(0, 0.45, u)),
      from: (u, v, p) => add(leafBase, mul(sub(p, leafBase), 0.05)), delay: (u) => leafDelay + u * 0.35, dur: 0.7,
    });
  }

  // carta kraft fuori, velina chiara dentro; all'entrata si apre dal retro verso il davanti
  const unwrap = (u, v, p) => [p[0] * 0.3, p[1], p[2] * 0.3];
  const wrapDelay = (t) => (u, v) => t + (0.55 * (1 + Math.cos(v * TAU))) / 2 + 0.25 * u;
  surface(22, 96, (u, v) => paperPoint(u, v * TAU, false), {
    mat: M.PAPER, sway: false, ao: (u) => lerp(0.85, 1.1, smooth(0, 0.7, u)), from: unwrap, delay: wrapDelay(0.35), dur: 0.9,
  });
  surface(20, 96, (u, v) => paperPoint(u, v * TAU + 0.3, true), {
    mat: M.TISSUE, sway: false, ao: (u) => lerp(0.45, 1, smooth(0.3, 0.95, u)), from: unwrap, delay: wrapDelay(0.55), dur: 0.9,
  });

  // nastro con fiocco
  const Y_RIB = Y_BIND + 0.08;
  surface(3, 96, (u, v) => {
    const phi = v * TAU;
    const y = Y_RIB + (u - 0.5) * 0.13;
    const r = paperRadiusAt(y, phi) + 0.014;
    return [r * Math.sin(phi), y, r * Math.cos(phi)];
  }, {
    mat: M.RIBBON, sway: false,
    from: (u, v, p) => [p[0] * 0.85, p[1], p[2] * 0.85], delay: (u, v) => 2.55 + 0.25 * v, dur: 0.5,
  });

  function strip(curve, widthDir, width, nu, t0) {
    surface(nu, 2, (u, v) => {
      const s = clamp(u, 0, 1);
      const p = curve(s);
      const w = widthDir(s);
      return [p[0] + w[0] * (v - 0.5) * width, p[1] + w[1] * (v - 0.5) * width, p[2] + w[2] * (v - 0.5) * width];
    }, {
      mat: M.RIBBON, sway: false,
      from: (u, v, p) => add(KNOT, mul(sub(p, KNOT), 0.05)), delay: (u) => t0 + u * 0.3, dur: 0.5,
    });
  }
  const inPlaneWidth = (curve, normal) => (s) => {
    const d = norm(sub(curve(Math.min(1, s + 0.01)), curve(Math.max(0, s - 0.01))));
    return norm(cross(normal, d));
  };

  const KNOT = [0, Y_RIB, paperRadiusAt(Y_RIB, 0) + 0.035];
  for (const side of [-1, 1]) {
    const ax = norm([side, 0.25, 0.15]);
    const ay = norm(sub([0, 1, 0], mul(ax, ax[1])));
    const pn = norm(cross(ax, ay));
    const loop = (s) => add(KNOT, add(
      mul(ax, (0.3 * (1 - Math.cos(TAU * s))) / 2),
      add(mul(ay, 0.13 * Math.sin(TAU * s)), [0, 0, 0.05 * Math.sin(Math.PI * s)]),
    ));
    strip(loop, inPlaneWidth(loop, pn), 0.075, 32, 2.85);
    const tail = (s) => [
      KNOT[0] + side * (0.03 + 0.14 * s) + 0.02 * Math.sin(s * 7),
      KNOT[1] - 0.55 * s,
      KNOT[2] + 0.03 + 0.04 * Math.sin(Math.PI * s),
    ];
    strip(tail, inPlaneWidth(tail, [0, 0, 1]), 0.09, 16, 2.95);
  }
  surface(6, 12, (u, v) => {
    const th = u * Math.PI;
    const ph = v * TAU;
    return [
      KNOT[0] + 0.05 * Math.sin(th) * Math.cos(ph),
      KNOT[1] + 0.05 * Math.cos(th),
      KNOT[2] + 0.03 + 0.035 * Math.sin(th) * Math.sin(ph),
    ];
  }, { mat: M.RIBBON, sway: false, from: (u, v, p) => add(KNOT, mul(sub(p, KNOT), 0.05)), delay: () => 2.75, dur: 0.4 });

  // ── dati della mesh in array tipizzati ──────────────────────────────────
  const NV = vP.length / 3;
  const NT = tI.length / 3;
  const POS0 = Float32Array.from(vP);
  const NRM = Float32Array.from(vN);
  const ATT = Float32Array.from(vA);
  const AOC = Float32Array.from(vO);
  const SWW = Float32Array.from(vW);
  const TRI = Uint32Array.from(tI);
  const TMAT = Uint8Array.from(tM);
  const TOBJ = Uint8Array.from(tO);
  const POSA = Float32Array.from(vS);
  const DEL = Float32Array.from(vD);
  const DUR = Float32Array.from(vT);
  const PROG = new Float32Array(NV).fill(1);
  let INTRO_END = 0;
  for (let i = 0; i < NV; i++) INTRO_END = Math.max(INTRO_END, DEL[i] + DUR[i]);

  const phases = [...new Set(vPh)];
  const SWI = Uint16Array.from(vPh.map((p) => phases.indexOf(p)));
  const SWX = new Float32Array(phases.length);
  const SWZ = new Float32Array(phases.length);

  const POS = new Float32Array(NV * 3);
  const SX = new Float32Array(NV);
  const SY = new Float32Array(NV);
  const IZ = new Float32Array(NV);

  // centro e raggio della scena
  const SCENE_C = (() => {
    const lo = [1e9, 1e9, 1e9], hi = [-1e9, -1e9, -1e9];
    for (let i = 0; i < NV; i++) {
      for (let c = 0; c < 3; c++) {
        lo[c] = Math.min(lo[c], POS0[i * 3 + c]);
        hi[c] = Math.max(hi[c], POS0[i * 3 + c]);
      }
    }
    return mix3(lo, hi, 0.5);
  })();
  let SCENE_R = 0, SCENE_RX = 0, SCENE_RY = 0;
  for (let i = 0; i < NV; i++) {
    const d = sub([POS0[i * 3], POS0[i * 3 + 1], POS0[i * 3 + 2]], SCENE_C);
    SCENE_R = Math.max(SCENE_R, len(d));
    SCENE_RX = Math.max(SCENE_RX, Math.hypot(d[0], d[2]));
    SCENE_RY = Math.max(SCENE_RY, Math.abs(d[1]));
  }

  // ── caratteri ───────────────────────────────────────────────────────────
  const RAMP_MAIN = '.,-:;=+*#%@';
  const RAMP_DISK = '.,:;o*#%@';
  const RAMP_PAPER = '.,:-=+*#';
  const SCRAMBLE = '01<>/\\|{}[]?!$&';
  const GLYPHS = [...new Set((RAMP_MAIN + RAMP_DISK + RAMP_PAPER + SCRAMBLE + 'oO@').split(''))].join('');
  const toIdx = (s) => Uint8Array.from(s.split('').map((c) => GLYPHS.indexOf(c)));
  const R_MAIN = toIdx(RAMP_MAIN);
  const R_DISK = toIdx(RAMP_DISK);
  const R_PAPER = toIdx(RAMP_PAPER);
  const R_SCR = toIdx(SCRAMBLE);
  const R_SEED = Uint8Array.from(' .:oO@'.split('').map((c) => (c === ' ' ? 255 : GLYPHS.indexOf(c))));
  const R_SEED_SOFT = R_SEED.slice(0, 4); // semi più quieti sotto la scritta: niente O e @
  const G_LETTER = GLYPHS.indexOf('@');

  // ── messaggio scritto nel disco del girasole toccato ────────────────────
  const MESSAGE = ['Eleval', 'sono', 'per te'];
  const FONT5x7 = {
    ' ': ['.....', '.....', '.....', '.....', '.....', '.....', '.....'],
    E: ['#####', '#....', '#....', '####.', '#....', '#....', '#####'],
    l: ['.##..', '..#..', '..#..', '..#..', '..#..', '..#..', '.###.'],
    e: ['.....', '.....', '.###.', '#...#', '#####', '#....', '.###.'],
    v: ['.....', '.....', '#...#', '#...#', '#...#', '.#.#.', '..#..'],
    a: ['.....', '.....', '.###.', '....#', '.####', '#...#', '.####'],
    s: ['.....', '.....', '.####', '#....', '.###.', '....#', '####.'],
    o: ['.....', '.....', '.###.', '#...#', '#...#', '#...#', '.###.'],
    n: ['.....', '.....', '#.##.', '##..#', '#...#', '#...#', '#...#'],
    p: ['.....', '.....', '####.', '#...#', '####.', '#....', '#....'],
    r: ['.....', '.....', '#.##.', '##..#', '#....', '#....', '#....'],
    t: ['.#...', '.#...', '####.', '.#...', '.#...', '.#..#', '..##.'],
  };
  const MSG_GAP = 2; // righe vuote tra una riga e l'altra
  const MSG_W = Math.max(...MESSAGE.map((line) => line.length * 6 - 1));
  const MSG_H = MESSAGE.length * 7 + (MESSAGE.length - 1) * MSG_GAP;
  const MSG_SEQ = new Uint8Array(MSG_W * MSG_H).fill(255); // per ogni pixel acceso: indice del carattere (ordine di battitura)
  let msgChars = 0;
  MESSAGE.forEach((line, li) => {
    const x0 = Math.floor((MSG_W - (line.length * 6 - 1)) / 2);
    const y0 = li * (7 + MSG_GAP);
    [...line].forEach((chr, ci) => {
      (FONT5x7[chr] || FONT5x7[' ']).forEach((row, ry) => {
        for (let rx = 0; rx < 5; rx++) {
          if (row[rx] === '#') MSG_SEQ[(y0 + ry) * MSG_W + x0 + ci * 6 + rx] = msgChars;
        }
      });
      msgChars++;
    });
  });
  let MSG_PW = 0.03, MSG_PH = 0.06; // lato di un pixel della scritta, in unità del disco
  const MAT_RAMP = [R_DISK, R_MAIN, R_MAIN, R_MAIN, R_MAIN, R_MAIN, R_MAIN, R_PAPER, R_PAPER, R_MAIN];

  const FONT = 'ui-monospace, "SF Mono", Menlo, Monaco, Consolas, "Roboto Mono", "DejaVu Sans Mono", monospace';
  const MAX_CELLS = 100000; // dettaglio massimo, con un tetto per gli schermi enormi

  // ── stato ───────────────────────────────────────────────────────────────
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const FOV = (38 * Math.PI) / 180;
  const NEAR = 0.04;
  const HOME_PITCH = 0.36;

  const state = {
    focus: -1,
    hover: -1,
    auto: !reduceMotion,
    dragging: false,
    lastInput: -1e9,
    opened: false, // la schermata "tocca per aprire" è passata
    build: reduceMotion ? INTRO_END : 0, // secondi dall'apertura: il bouquet si compone
    msg: 0, // secondi dal tocco su un girasole: la scritta si batte
  };
  const view = { yaw: -0.25, pitch: HOME_PITCH, dist: 10, target: SCENE_C.slice() };
  const goal = { yaw: -0.25, pitch: HOME_PITCH, dist: 10, target: SCENE_C.slice() };
  const vel = { yaw: 0, pitch: 0 };
  let homeDist = 0;
  let swayAmp = 0;

  // ── griglia e atlante dei caratteri ─────────────────────────────────────
  const canvas = document.getElementById('scena');
  const ctx = canvas.getContext('2d');
  const colorCanvas = document.createElement('canvas');
  const colorCtx = colorCanvas.getContext('2d');
  let atlas = null;
  let dpr = 1, cw = 8, ch = 14, cols = 0, rows = 0, NC = 0, fpx = 1;
  let ZB, TB, B0, B1, GB, colorImage, CD;


  function fitDistance(r) {
    const aspect = canvas.width / canvas.height;
    const half = Math.min(FOV / 2, Math.atan(Math.tan(FOV / 2) * aspect));
    return r / Math.sin(half);
  }

  function buildAtlas(fontPx) {
    atlas = document.createElement('canvas');
    atlas.width = cw * GLYPHS.length;
    atlas.height = ch;
    const a = atlas.getContext('2d');
    a.font = `600 ${fontPx * dpr}px ${FONT}`;
    a.textAlign = 'center';
    a.textBaseline = 'middle';
    a.fillStyle = '#fff';
    for (let i = 0; i < GLYPHS.length; i++) a.fillText(GLYPHS[i], i * cw + cw / 2, ch / 2 + 0.5);
  }

  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 3);
    const W = Math.floor(innerWidth * dpr);
    const H = Math.floor(innerHeight * dpr);
    let fontPx = innerWidth < 700 ? 6 : 7; // caratteri piccoli = dettaglio massimo
    for (;;) {
      ctx.font = `600 ${fontPx * dpr}px ${FONT}`;
      cw = Math.max(2, Math.round(ctx.measureText('M').width));
      ch = Math.max(3, Math.round(fontPx * dpr * 1.16));
      cols = Math.floor(W / cw);
      rows = Math.floor(H / ch);
      if (cols * rows <= MAX_CELLS || fontPx >= 16) break;
      fontPx += 1;
    }
    // con dpr intero, dimensioni CSS intere: niente ricampionamento sfocato
    if (Number.isInteger(dpr)) {
      while ((cols * cw) % dpr) cols--;
      while ((rows * ch) % dpr) rows--;
    }

    canvas.width = cols * cw;
    canvas.height = rows * ch;
    const cssW = canvas.width / dpr;
    const cssH = canvas.height / dpr;
    canvas.style.width = `${cssW}px`;
    canvas.style.height = `${cssH}px`;
    canvas.style.left = `${Math.floor((innerWidth - cssW) / 2)}px`;
    canvas.style.top = `${Math.floor((innerHeight - cssH) / 2)}px`;

    NC = cols * rows;
    ZB = new Float32Array(NC);
    TB = new Int32Array(NC);
    B0 = new Float32Array(NC);
    B1 = new Float32Array(NC);
    GB = new Uint8Array(NC);
    colorCanvas.width = cols;
    colorCanvas.height = rows;
    colorImage = colorCtx.createImageData(cols, rows);
    CD = colorImage.data;
    buildAtlas(fontPx);
    fpx = canvas.height / 2 / Math.tan(FOV / 2);

    // la scritta deve stare nel disco: le celle sono più alte che larghe
    const aspect = ch / cw;
    MSG_PW = 0.9 / Math.hypot(MSG_W / 2, (MSG_H / 2) * aspect);
    MSG_PH = MSG_PW * aspect;

    const previous = homeDist;
    // il bouquet è più alto che largo: si inquadra sui due ingombri, non sulla sfera
    const tanV = Math.tan(FOV / 2);
    const portrait = canvas.height > canvas.width;
    homeDist = Math.max((SCENE_RY * 1.05) / tanV, (SCENE_RX * (portrait ? 0.95 : 1.12)) / (tanV * (canvas.width / canvas.height)));
    if (!previous) {
      view.dist = goal.dist = homeDist * 1.35; // parte un po' lontano e si avvicina all'apertura
    } else if (state.focus < 0) {
      goal.dist *= homeDist / previous;
      view.dist *= homeDist / previous;
    } else {
      goal.dist = focusDistance(flowers[state.focus]);
    }
  }

  // ── rendering ───────────────────────────────────────────────────────────
  function render(time) {
    // camera in orbita attorno al bersaglio
    const cp = Math.cos(view.pitch), sp = Math.sin(view.pitch);
    const cy = Math.cos(view.yaw), sy = Math.sin(view.yaw);
    const ex = view.target[0] + view.dist * cp * sy;
    const ey = view.target[1] + view.dist * sp;
    const ez = view.target[2] + view.dist * cp * cy;
    const fx = -cp * sy, fy = -sp, fz = -cp * cy; // avanti
    const rx = cy, rz = -sy; // destra (orizzontale)
    const ux = -sy * sp, uy = cp, uz = -cy * sp; // su

    // luce in alto a sinistra, verso chi guarda
    let lx = -0.45 * rx + 0.4 * ux - 0.8 * fx;
    let ly = 0.4 * uy - 0.8 * fy;
    let lz = -0.45 * rz + 0.4 * uz - 0.8 * fz;
    const ll = Math.sqrt(lx * lx + ly * ly + lz * lz);
    lx /= ll; ly /= ll; lz /= ll;

    // oscillazione nel vento, una fase per fiore
    for (let s = 0; s < phases.length; s++) {
      SWX[s] = swayAmp * Math.sin(time * 0.9 + phases[s]);
      SWZ[s] = swayAmp * 0.6 * Math.sin(time * 0.67 + phases[s] * 1.7);
    }

    // proiezione dei vertici sulla griglia
    const halfW = canvas.width / 2, halfH = canvas.height / 2;
    const icw = 1 / cw, ich = 1 / ch;
    const building = state.build < INTRO_END;
    for (let i = 0, j = 0; i < NV; i++, j += 3) {
      let px = POS0[j], py = POS0[j + 1], pz = POS0[j + 2];
      if (building) {
        let k = (state.build - DEL[i]) / DUR[i];
        k = k < 0 ? 0 : k > 1 ? 1 : k;
        PROG[i] = k;
        if (k < 1) {
          const e = 1 - (1 - k) * (1 - k) * (1 - k);
          px = POSA[j] + (px - POSA[j]) * e;
          py = POSA[j + 1] + (py - POSA[j + 1]) * e;
          pz = POSA[j + 2] + (pz - POSA[j + 2]) * e;
        }
      }
      const w = SWW[i];
      if (w > 0) {
        const s = SWI[i];
        px += w * SWX[s];
        pz += w * SWZ[s];
      }
      POS[j] = px; POS[j + 1] = py; POS[j + 2] = pz;
      const dx = px - ex, dy = py - ey, dz = pz - ez;
      const z = dx * fx + dy * fy + dz * fz;
      if (z < NEAR) { IZ[i] = -1; continue; }
      const iz = 1 / z;
      IZ[i] = iz;
      SX[i] = (halfW + (dx * rx + dz * rz) * iz * fpx) * icw;
      SY[i] = (halfH - (dx * ux + dy * uy + dz * uz) * iz * fpx) * ich;
    }

    // rasterizzazione con z-buffer (si campiona il centro di ogni cella)
    ZB.fill(0);
    TB.fill(-1);
    for (let t = 0, q = 0; t < NT; t++, q += 3) {
      const i0 = TRI[q], i1 = TRI[q + 1], i2 = TRI[q + 2];
      const z0 = IZ[i0], z1 = IZ[i1], z2 = IZ[i2];
      if (z0 < 0 || z1 < 0 || z2 < 0) continue;
      if (building && (PROG[i0] === 0 || PROG[i1] === 0 || PROG[i2] === 0)) continue; // non ancora nato
      const x0 = SX[i0], y0 = SY[i0], x1 = SX[i1], y1 = SY[i1], x2 = SX[i2], y2 = SY[i2];
      const minX = x0 < x1 ? (x0 < x2 ? x0 : x2) : x1 < x2 ? x1 : x2;
      const maxX = x0 > x1 ? (x0 > x2 ? x0 : x2) : x1 > x2 ? x1 : x2;
      const ax = Math.max(0, Math.ceil(minX - 0.5));
      const bx = Math.min(cols - 1, Math.floor(maxX - 0.5));
      if (ax > bx) continue;
      const minY = y0 < y1 ? (y0 < y2 ? y0 : y2) : y1 < y2 ? y1 : y2;
      const maxY = y0 > y1 ? (y0 > y2 ? y0 : y2) : y1 > y2 ? y1 : y2;
      const ay = Math.max(0, Math.ceil(minY - 0.5));
      const by = Math.min(rows - 1, Math.floor(maxY - 0.5));
      if (ay > by) continue;
      const area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
      if (area > -1e-12 && area < 1e-12) continue;
      const ia = 1 / area;
      for (let py = ay; py <= by; py++) {
        const cyy = py + 0.5;
        let k = py * cols + ax;
        for (let px = ax; px <= bx; px++, k++) {
          const cxx = px + 0.5;
          const w0 = ((x1 - cxx) * (y2 - cyy) - (x2 - cxx) * (y1 - cyy)) * ia;
          if (w0 < -1e-5) continue;
          const w1 = ((x2 - cxx) * (y0 - cyy) - (x0 - cxx) * (y2 - cyy)) * ia;
          if (w1 < -1e-5) continue;
          const w2 = 1 - w0 - w1;
          if (w2 < -1e-5) continue;
          const z = w0 * z0 + w1 * z1 + w2 * z2;
          if (z > ZB[k]) {
            ZB[k] = z;
            TB[k] = t;
            B0[k] = w0;
            B1[k] = w1;
          }
        }
      }
    }

    // ombreggiatura: da cella a carattere + colore
    const cellWorld = cw / fpx; // lato di una cella in unità mondo a distanza 1
    const fogR = state.focus >= 0 ? flowers[state.focus].R * 1.6 : SCENE_R;
    const fogStart = view.dist - fogR * 0.2;
    const fogSpan = fogR * 1.6;
    const hover = state.focus < 0 ? state.hover : -1;
    // messaggio: quanti caratteri sono già battuti e quanto si quietano i semi intorno
    const typed = state.focus >= 0 ? (state.msg - 0.8) * 11 : -1;
    const hush = state.focus >= 0 ? smooth(0.3, 1.2, state.msg) : 0;
    const glow = 0.92 + 0.08 * Math.sin(time * 2.2);

    for (let k = 0; k < NC; k++) {
      const t = TB[k];
      if (t < 0) { GB[k] = 255; continue; }
      const q = t * 3;
      const i0 = TRI[q], i1 = TRI[q + 1], i2 = TRI[q + 2];
      const w0 = B0[k], w1 = B1[k], w2 = 1 - w0 - w1;
      // pesi corretti per la prospettiva
      let p0 = w0 * IZ[i0], p1 = w1 * IZ[i1], p2 = w2 * IZ[i2];
      const depth = 1 / (p0 + p1 + p2);
      p0 *= depth; p1 *= depth; p2 *= depth;

      const a0 = i0 * 3, a1 = i1 * 3, a2 = i2 * 3;
      let nx = p0 * NRM[a0] + p1 * NRM[a1] + p2 * NRM[a2];
      let ny = p0 * NRM[a0 + 1] + p1 * NRM[a1 + 1] + p2 * NRM[a2 + 1];
      let nz = p0 * NRM[a0 + 2] + p1 * NRM[a1 + 2] + p2 * NRM[a2 + 2];
      const nl = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;
      nx /= nl; ny /= nl; nz /= nl;
      const wx = p0 * POS[a0] + p1 * POS[a1] + p2 * POS[a2];
      const wy = p0 * POS[a0 + 1] + p1 * POS[a1 + 1] + p2 * POS[a2 + 1];
      const wz = p0 * POS[a0 + 2] + p1 * POS[a1 + 2] + p2 * POS[a2 + 2];
      let vx = ex - wx, vy = ey - wy, vz = ez - wz;
      const vl = Math.sqrt(vx * vx + vy * vy + vz * vz) || 1;
      vx /= vl; vy /= vl; vz /= vl;

      // superfici sottili: si illumina sempre il lato che guarda la camera
      let ndv = nx * vx + ny * vy + nz * vz;
      let back = false;
      if (ndv < 0) { nx = -nx; ny = -ny; nz = -nz; ndv = -ndv; back = true; }
      const ndl = nx * lx + ny * ly + nz * lz;
      const diff = ndl > 0 ? ndl : 0;

      const mat = TMAT[t];
      const obj = TOBJ[t];
      const ao = p0 * AOC[i0] + p1 * AOC[i1] + p2 * AOC[i2];
      const c0 = i0 * 2, c1 = i1 * 2, c2 = i2 * 2;
      const ua = p0 * ATT[c0] + p1 * ATT[c1] + p2 * ATT[c2];
      const ub = p0 * ATT[c0 + 1] + p1 * ATT[c1 + 1] + p2 * ATT[c2 + 1];

      const rim = 1 - ndv;
      let lum = (0.16 + 0.72 * diff + 0.3 * ndv) * ao + 0.14 * rim * rim * rim;
      let spec = 0;
      if (diff > 0 && MAT_KS[mat] > 0) {
        const hx = lx + vx, hy = ly + vy, hz = lz + vz;
        const ndh = (nx * hx + ny * hy + nz * hz) / (Math.sqrt(hx * hx + hy * hy + hz * hz) || 1);
        if (ndh > 0) spec = MAT_KS[mat] * Math.pow(ndh, MAT_SH[mat]);
      }

      let r, g, b;
      let seedPat = -1;
      let letter = 0; // 1 = carattere in battitura, 2 = carattere scritto
      switch (mat) {
        case M.DISK: {
          const fl = flowers[obj];
          const rho = Math.sqrt(ua * ua + ub * ub);
          // quanti semi cadono in una cella: se troppi, si sfuma il motivo
          const ratio = (cellWorld * depth) / (fl.Rd * fl.seeds.spacing);
          let pat = 0.4;
          if (ratio < 1.6) {
            const d = seedDistance(fl.seeds, ua, ub) / fl.seeds.spacing;
            pat = lerp(1 - smooth(0.12, 0.55, d), 0.4, smooth(0.55, 1.6, ratio));
            if (ratio < 0.9) seedPat = pat; // semi abbastanza grandi: carattere dal disegno del seme
          }
          const ring = smooth(0.7, 0.82, rho);
          const core = 1 - smooth(0.09, 0.2, rho);
          r = 82 + 70 * pat; g = 46 + 42 * pat; b = 20 + 16 * pat;
          r = lerp(r, 160 + 95 * pat, ring); g = lerp(g, 80 + 110 * pat, ring); b = lerp(b, 22 + 30 * pat, ring);
          r = lerp(r, 120 + 40 * pat, core); g = lerp(g, 112 + 40 * pat, core); b = lerp(b, 40, core);
          lum *= (0.35 + 0.7 * pat) * (1 - 0.3 * smooth(0.94, 1, rho));
          if (obj === state.focus && hush > 0) {
            lum *= 1 - 0.45 * hush;
            if (typed > 0) {
              // coordinate del punto nel riquadro della scritta, dritta per chi guarda
              const dx = wx - fl.C[0], dy = wy - fl.C[1], dz = wz - fl.C[2];
              const tx = (dx * fl.TR[0] + dy * fl.TR[1] + dz * fl.TR[2]) / fl.Rd;
              const ty = (dx * fl.TU[0] + dy * fl.TU[1] + dz * fl.TU[2]) / fl.Rd;
              const ix = Math.floor(tx / MSG_PW + MSG_W / 2);
              const iy = Math.floor(MSG_H / 2 - ty / MSG_PH);
              if (ix >= 0 && iy >= 0 && ix < MSG_W && iy < MSG_H) {
                const seq = MSG_SEQ[iy * MSG_W + ix];
                if (seq !== 255 && typed > seq) letter = typed > seq + 1 ? 2 : 1;
              }
            }
          }
          break;
        }
        case M.PETAL: {
          if (ua < 0.35) {
            const s = ua / 0.35;
            r = lerp(206, 249, s); g = lerp(108, 184, s); b = lerp(16, 26, s);
          } else {
            const s = (ua - 0.35) / 0.65;
            r = lerp(249, 255, s); g = lerp(184, 218, s); b = lerp(26, 76, s);
          }
          if (back) { r *= 0.9; g *= 0.94; b = b * 0.5 + 34; }
          lum *= 0.96 + 0.04 * Math.cos(ub * 10);
          break;
        }
        case M.BOWL:
          r = 86; g = 122; b = 42;
          lum *= 0.88 + 0.12 * Math.cos(ub * TAU * 26);
          break;
        case M.BRACT:
          r = lerp(66, 118, ua); g = lerp(112, 150, ua); b = lerp(34, 58, ua);
          break;
        case M.STEM:
          r = 84; g = 132; b = 46;
          lum *= 0.92 + 0.08 * Math.cos(ub * TAU * 5);
          break;
        case M.CUT:
          r = 196; g = 206; b = 150;
          break;
        case M.LEAF: {
          const ab = Math.abs(ub);
          const f = ua * 7 - ab * 2.4;
          const d = Math.abs(f - Math.round(f));
          const vein = Math.max(1 - smooth(0, 0.09, ab), (1 - smooth(0.02, 0.1, d)) * smooth(0.05, 0.15, ab) * (1 - ab) * 0.7);
          r = lerp(62, 150, vein * 0.6); g = lerp(120, 190, vein * 0.6); b = lerp(38, 92, vein * 0.6);
          if (back) { r = r * 0.9 + 12; g = g * 0.95 + 10; b = b * 0.9 + 14; }
          break;
        }
        case M.PAPER:
          r = 182; g = 134; b = 88;
          lum *= 1.5; // la carta kraft è opaca e chiara: va schiarita rispetto alla luce dall'alto
          if (back) { r *= 0.92; g *= 0.9; b *= 0.88; }
          break;
        case M.TISSUE:
          r = 196; g = 178; b = 150;
          break;
        default: // nastro
          r = 38; g = 112; b = 70;
      }

      if (letter) {
        // lettere: crema dorata, sempre ben leggibili, niente nebbia
        const l = (0.82 + 0.3 * diff) * glow;
        GB[k] = letter === 2 ? G_LETTER : R_SCR[(Math.random() * R_SCR.length) | 0];
        const o = k * 4;
        CD[o] = 255 * l; CD[o + 1] = 236 * l; CD[o + 2] = 170 * l; CD[o + 3] = 255;
        continue;
      }

      // nebbia: ciò che sta dietro si spegne un po'
      lum *= 1 - 0.5 * clamp((depth - fogStart) / fogSpan, 0, 1);
      if (obj === hover) lum *= 1.25;

      const ramp = MAT_RAMP[mat];
      let tl = lum * 0.82;
      tl = tl < 0 ? 0 : tl > 0.999 ? 0.999 : tl;
      let glyph = ramp[(tl * ramp.length) | 0];
      if (seedPat >= 0) {
        const sr = obj === state.focus && hush > 0.5 ? R_SEED_SOFT : R_SEED;
        glyph = sr[Math.min(sr.length - 1, (seedPat * sr.length) | 0)];
      }
      if (building) {
        // entrata: ogni parte nasce come caratteri a caso che poi si assestano
        const pc = p0 * PROG[i0] + p1 * PROG[i1] + p2 * PROG[i2];
        if (pc < 1 && Math.random() > pc * pc) glyph = R_SCR[(Math.random() * R_SCR.length) | 0];
      }
      GB[k] = glyph;
      if (glyph === 255) continue; // spazio vuoto tra i semi

      const c = lum > 1.15 ? 1.15 : lum;
      const o = k * 4;
      CD[o] = r * c + 255 * spec; CD[o + 1] = g * c + 235 * spec; CD[o + 2] = b * c + 190 * spec;
      CD[o + 3] = 255;
    }

    // disegno: caratteri bianchi dall'atlante, poi colore per cella
    ctx.globalCompositeOperation = 'source-over';
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (let py = 0, k = 0; py < rows; py++) {
      const dy = py * ch;
      for (let px = 0; px < cols; px++, k++) {
        const g = GB[k];
        if (g !== 255) ctx.drawImage(atlas, g * cw, 0, cw, ch, px * cw, dy, cw, ch);
      }
    }
    colorCtx.putImageData(colorImage, 0, 0);
    ctx.globalCompositeOperation = 'source-in';
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(colorCanvas, 0, 0, canvas.width, canvas.height);
    ctx.globalCompositeOperation = 'source-over';
  }

  // ── camera ──────────────────────────────────────────────────────────────
  const clampDist = (d) => clamp(d, 0.28, homeDist * 2.6);

  function step(dt, now) {
    if (!state.dragging) {
      goal.yaw += vel.yaw * dt;
      goal.pitch += vel.pitch * dt;
      const damp = Math.exp(-dt * 3.2);
      vel.yaw *= damp;
      vel.pitch *= damp;
      // mentre il bouquet nasce gira appena, così si vede crescere di fronte
      const spin = state.build < INTRO_END ? 0.04 : 0.2;
      if (state.auto && state.focus < 0 && now - state.lastInput > 2500) goal.yaw += spin * dt;
    }
    goal.pitch = clamp(goal.pitch, -1.48, 1.48);
    const ka = 1 - Math.exp(-dt * (state.dragging ? 18 : 7));
    const kt = 1 - Math.exp(-dt * 5);
    view.yaw += (goal.yaw - view.yaw) * ka;
    view.pitch += (goal.pitch - view.pitch) * ka;
    view.dist = Math.exp(lerp(Math.log(view.dist), Math.log(goal.dist), kt));
    for (let i = 0; i < 3; i++) view.target[i] += (goal.target[i] - view.target[i]) * kt;
    const swayGoal = reduceMotion ? 0 : state.focus >= 0 ? 0.004 : 0.035;
    swayAmp += (swayGoal - swayAmp) * kt;
    if (state.opened && state.build < INTRO_END) {
      state.build += dt;
      if (state.build > 2.6) document.body.classList.add('pronto');
    }
    if (state.focus >= 0) state.msg += dt;
  }

  // abbastanza vicino da dare a ogni pixel della scritta un po' più di una cella
  function focusDistance(fl) {
    const legible = (MSG_PW * fl.Rd * fpx) / (1.3 * cw);
    return Math.min(fitDistance(fl.R * 1.25), legible);
  }

  function focusFlower(i) {
    const fl = flowers[i];
    if (!fl) return;
    state.focus = i;
    state.hover = -1;
    state.msg = 0;
    goal.target = fl.C.slice();
    goal.dist = focusDistance(fl);
    // di fronte al disco, così la scritta si legge dritta
    const yawTo = Math.atan2(fl.F[0], fl.F[2]);
    goal.yaw = view.yaw + wrapPi(yawTo - view.yaw);
    goal.pitch = clamp(Math.asin(clamp(fl.F[1], -1, 1)), -1.48, 1.48);
    vel.yaw = vel.pitch = 0;
    updateCard();
  }

  function goHome() {
    state.focus = -1;
    goal.target = SCENE_C.slice();
    goal.dist = homeDist;
    goal.pitch = HOME_PITCH;
    vel.yaw = vel.pitch = 0;
    updateCard();
  }

  // ── interfaccia ─────────────────────────────────────────────────────────
  const btnHome = document.getElementById('btn-bouquet');
  const btnAuto = document.getElementById('btn-ruota');
  const hint = document.getElementById('aiuto');
  let hintTimer = 0;

  if (window.matchMedia('(pointer: coarse)').matches) {
    hint.textContent = 'trascina per girarli · tocca un girasole';
  }

  function updateCard() {
    btnHome.hidden = state.focus < 0;
  }

  function setAuto(on) {
    state.auto = on;
    btnAuto.setAttribute('aria-pressed', String(on));
  }

  function touched() {
    state.lastInput = performance.now();
    if (!hintTimer) hintTimer = setTimeout(() => hint.classList.add('nascosto'), 5000);
  }

  btnHome.addEventListener('click', goHome);
  btnAuto.addEventListener('click', () => setAuto(!state.auto));
  setAuto(state.auto);

  // girasole sotto il puntatore (con un po' di tolleranza tra i petali)
  function flowerAt(clientX, clientY) {
    const rect = canvas.getBoundingClientRect();
    const cx = Math.floor(((clientX - rect.left) * dpr) / cw);
    const cy = Math.floor(((clientY - rect.top) * dpr) / ch);
    let best = -1, bestD = 1e9;
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -2; dx <= 2; dx++) {
        const x = cx + dx, y = cy + dy;
        if (x < 0 || y < 0 || x >= cols || y >= rows) continue;
        const t = TB[y * cols + x];
        if (t < 0 || TOBJ[t] >= flowers.length) continue;
        const d = dx * dx + dy * dy * 4;
        if (d < bestD) { bestD = d; best = TOBJ[t]; }
      }
    }
    return best;
  }

  // ── musica ──────────────────────────────────────────────────────────────
  // Se nella cartella c'è un mp3, scrivi qui il nome (es. 'canzone.mp3'): funziona ovunque, anche offline.
  // Se resta vuoto si usa il video di YouTube, che parte solo con il sito online (https).
  const SONG_FILE = '';
  const SONG_YOUTUBE = 'Z0GcAu7LAu4'; // V – For Us
  const btnMusic = document.getElementById('btn-musica');
  const music = { on: true, playing: false, audio: null, yt: null, ytReady: false };

  if (SONG_FILE) {
    music.audio = new Audio(SONG_FILE);
    music.audio.loop = true;
    music.audio.preload = 'auto';
  } else if (location.protocol.startsWith('http')) {
    window.onYouTubeIframeAPIReady = () => {
      music.yt = new window.YT.Player('musica-yt', {
        width: 200,
        height: 200,
        videoId: SONG_YOUTUBE,
        playerVars: { controls: 0, loop: 1, playlist: SONG_YOUTUBE, playsinline: 1, rel: 0, origin: location.origin },
        events: {
          onReady: () => {
            music.ytReady = true;
            if (music.playing) playSong();
          },
          onError: () => { btnMusic.hidden = true; },
        },
      });
    };
    const api = document.createElement('script');
    api.src = 'https://www.youtube.com/iframe_api';
    api.onerror = () => { btnMusic.hidden = true; };
    document.head.appendChild(api);
  } else {
    btnMusic.hidden = true; // aperto come file: YouTube non parte, il bottone non servirebbe
  }

  function playSong() {
    music.playing = true;
    if (!music.on) return;
    if (music.audio) music.audio.play().catch(() => {});
    else if (music.ytReady) {
      music.yt.unMute();
      music.yt.playVideo();
    }
  }
  function pauseSong() {
    if (music.audio) music.audio.pause();
    else if (music.ytReady) music.yt.pauseVideo();
  }
  function setMusic(on) {
    music.on = on;
    btnMusic.setAttribute('aria-pressed', String(on));
    if (on && music.playing) playSong();
    else pauseSong();
  }
  btnMusic.addEventListener('click', () => setMusic(!music.on));
  // il telefono in tasca o un'altra app aperta: la musica si ferma e poi riprende
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) pauseSong();
    else if (music.on && music.playing) playSong();
  });

  // ── schermata d'apertura: il primo tocco avvia musica e bouquet ─────────
  const gate = document.getElementById('apri');
  function openGift() {
    if (state.opened) return;
    state.opened = true;
    goal.dist = homeDist;
    gate.classList.add('via');
    setTimeout(() => { gate.hidden = true; }, 1000);
    playSong();
  }
  gate.addEventListener('click', openGift);
  gate.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      openGift();
    }
  });
  gate.focus();

  // ── input: trascinamento, pizzico, rotella, tastiera ────────────────────
  const ROT = 0.0065;
  const pointers = new Map();
  let press = null;
  let pinch = 0;
  let lastMove = 0;

  const pinchDistance = () => {
    const [a, b] = [...pointers.values()];
    return Math.hypot(a.x - b.x, a.y - b.y);
  };

  canvas.addEventListener('pointerdown', (e) => {
    canvas.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pointers.size === 1) {
      press = { x: e.clientX, y: e.clientY, moved: false };
      state.dragging = true;
      vel.yaw = vel.pitch = 0;
      canvas.classList.add('trascina');
    } else if (pointers.size === 2) {
      pinch = pinchDistance();
      if (press) press.moved = true;
    }
    touched();
  });

  canvas.addEventListener('pointermove', (e) => {
    const p = pointers.get(e.pointerId);
    if (!p) {
      if (e.pointerType === 'mouse') {
        const f = flowerAt(e.clientX, e.clientY);
        state.hover = state.focus < 0 ? f : -1;
        canvas.classList.toggle('sopra-fiore', f >= 0 && f !== state.focus);
      }
      return;
    }
    const dx = e.clientX - p.x;
    const dy = e.clientY - p.y;
    p.x = e.clientX;
    p.y = e.clientY;
    if (pointers.size === 1) {
      goal.yaw -= dx * ROT;
      goal.pitch += dy * ROT;
      const now = performance.now();
      const dt = Math.max(8, now - lastMove) / 1000;
      lastMove = now;
      vel.yaw = lerp(vel.yaw, (-dx * ROT) / dt, 0.5);
      vel.pitch = lerp(vel.pitch, (dy * ROT) / dt, 0.5);
      if (press && Math.hypot(e.clientX - press.x, e.clientY - press.y) > 6) press.moved = true;
    } else if (pointers.size === 2) {
      const d = pinchDistance();
      if (pinch > 0 && d > 0) goal.dist = clampDist((goal.dist * pinch) / d);
      pinch = d;
    }
    touched();
  });

  const release = (e) => {
    if (!pointers.has(e.pointerId)) return;
    pointers.delete(e.pointerId);
    if (pointers.size === 1) pinch = 0;
    if (pointers.size > 0) return;
    state.dragging = false;
    canvas.classList.remove('trascina');
    if (performance.now() - lastMove > 90) vel.yaw = vel.pitch = 0;
    if (press && !press.moved && e.type === 'pointerup') {
      const i = flowerAt(e.clientX, e.clientY);
      if (i >= 0 && i !== state.focus) focusFlower(i);
    }
    press = null;
    touched();
  };
  canvas.addEventListener('pointerup', release);
  canvas.addEventListener('pointercancel', release);
  canvas.addEventListener('pointerleave', () => { if (!pointers.size) state.hover = -1; });

  canvas.addEventListener('dblclick', (e) => {
    if (flowerAt(e.clientX, e.clientY) < 0) goHome();
  });

  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    let dy = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY;
    if (e.ctrlKey) dy *= 3; // pizzico sul trackpad
    goal.dist = clampDist(goal.dist * Math.exp(dy * 0.0011));
    touched();
  }, { passive: false });

  window.addEventListener('keydown', (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (!state.opened) return;
    if (e.target instanceof HTMLButtonElement && (e.key === ' ' || e.key === 'Enter')) return;
    let handled = true;
    switch (e.key) {
      case 'ArrowLeft': goal.yaw += 0.15; break;
      case 'ArrowRight': goal.yaw -= 0.15; break;
      case 'ArrowUp': goal.pitch += 0.1; break;
      case 'ArrowDown': goal.pitch -= 0.1; break;
      case '+': case '=': goal.dist = clampDist(goal.dist / 1.15); break;
      case '-': case '_': goal.dist = clampDist(goal.dist * 1.15); break;
      case 'Escape': case 'r': case 'R': goHome(); break;
      case ' ': setAuto(!state.auto); break;
      default:
        if (/^[1-9]$/.test(e.key) && Number(e.key) <= flowers.length) focusFlower(Number(e.key) - 1);
        else handled = false;
    }
    if (handled) {
      e.preventDefault();
      touched();
    }
  });

  window.addEventListener('resize', resize);

  // ── avvio ───────────────────────────────────────────────────────────────
  resize();
  if (state.build >= INTRO_END) document.body.classList.add('pronto');
  let last = performance.now();
  function frame(now) {
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    step(dt, now);
    render(now / 1000);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();
