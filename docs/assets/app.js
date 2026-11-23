/* MMAI front-end: hash router, ratings table, team pages, bracket, in-browser matchup net. */
(() => {
  "use strict";
  const $ = (s, el = document) => el.querySelector(s);
  const app = $("#app");
  const cache = {};
  const ROUND_NAMES = ["Opening Round", "Round of 64", "Round of 32", "Sweet 16", "Elite 8", "Final Four", "Championship"];

  // ---------------------------------------------------------------- utils
  async function get(name) {
    if (cache[name]) return cache[name];
    const r = await fetch(`data/${name}`, { cache: "no-cache" });
    if (!r.ok) throw new Error(`${name}: ${r.status}`);
    return (cache[name] = await r.json());
  }
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toFixed(d));
  const sgn = (v, d = 1) => (v === null || v === undefined ? "—" : (v > 0 ? "+" : "") + Number(v).toFixed(d));
  const pct = (p, d = 0) => {
    if (p === null || p === undefined) return "—";
    const lim = d === 0 ? 0.99 : 0.999;
    if (p > lim) return ">" + (100 * lim).toFixed(d) + "%";
    if (p < 1 - lim && p > 0) return "<" + (100 * (1 - lim)).toFixed(d) + "%";
    return (100 * p).toFixed(d) + "%";
  };
  const slugify = (s) => String(s).toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  const logo = (t, cls = "") => (t && t.logo ? `<img class="logo ${cls}" src="${esc(t.logo)}" alt="" loading="lazy" onerror="this.style.visibility='hidden'">` : `<span class="logo ${cls}"></span>`);
  const teamLink = (name, t) => `<a href="#/team/${slugify(name)}" class="teamcell">${logo(t)}<span class="teamname">${esc(name)}</span></a>`;
  const locWord = (l) => (l > 0 ? "vs" : l < 0 ? "@" : "vs (N)");

  let TEAMS = {}; // name -> rating row
  async function teams() {
    if (Object.keys(TEAMS).length) return TEAMS;
    const r = await get("ratings.json");
    r.teams.forEach((t) => (TEAMS[t.team] = t));
    return TEAMS;
  }

  // percentile helpers (for heat colouring)
  function pctRank(values, v) {
    if (v === null || v === undefined) return null;
    let below = 0;
    for (const x of values) if (x < v) below++;
    return below / Math.max(values.length - 1, 1);
  }
  function heat(p, invert = false) {
    if (p === null || p === undefined) return "";
    if (invert) p = 1 - p;
    const a = Math.abs(p - 0.5) * 2; // 0..1
    const col = p >= 0.5 ? "var(--pos)" : "var(--neg)";
    return `background: color-mix(in srgb, ${col} ${Math.round(a * 26)}%, transparent)`;
  }

  // ---------------------------------------------------------------- theme
  const root = document.documentElement;
  try { const t = localStorage.getItem("mmai-theme"); if (t) root.dataset.theme = t; } catch (e) {}
  $("#theme").addEventListener("click", () => {
    const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("mmai-theme", root.dataset.theme); } catch (e) {}
  });

  // ---------------------------------------------------------------- router
  const routes = { ratings: viewRatings, team: viewTeam, bracket: viewBracket, matchup: viewMatchup, games: viewGames, method: viewMethod };
  async function route() {
    const h = location.hash.replace(/^#\/?/, "") || "ratings";
    const [path, query] = h.split("?");
    const [name, arg] = path.split("/");
    const params = new URLSearchParams(query || "");
    document.querySelectorAll("#nav a").forEach((a) => a.classList.toggle("active", a.dataset.route === (name === "team" ? "ratings" : name)));
    const fn = routes[name] || viewRatings;
    app.innerHTML = `<div class="loading">Loading…</div>`;
    try {
      await fn(arg ? decodeURIComponent(arg) : null, params);
    } catch (e) {
      console.error(e);
      app.innerHTML = `<div class="card pad">Could not load data (${esc(e.message)}). The daily pipeline may not have run yet.</div>`;
    }
    window.scrollTo({ top: 0 });
  }
  window.addEventListener("hashchange", route);

  get("meta.json").then((m) => ($("#updated").textContent = `Updated ${m.updated}`)).catch(() => {});

  // ================================================================ RATINGS
  const VIEWS = {
    core: [
      ["power", "Power", 1, false, "Average predicted margin vs every D-I team on a neutral floor (network round-robin)"],
      ["eff_o", "AdjO", 1, false, "Adjusted offensive efficiency (pts / 100 poss)"],
      ["eff_d", "AdjD", 1, true, "Adjusted defensive efficiency (lower is better)"],
      ["adjem", "AdjEM", 1, false, "AdjO − AdjD"],
      ["adjt", "Tempo", 1, false, "Adjusted possessions / 40 min"],
      ["elo", "Elo", 0, false, "Margin-aware Elo"],
      ["roster", "Roster", 1, false, "Player-based strength: Σ minutes share × player value"],
      ["injury_pts", "Inj", 1, false, "Availability adjustment in points (injuries / absences)"],
      ["sos", "SOS", 1, false, "Average power of opponents faced"],
    ],
    offense: [
      ["efg_o", "eFG%", 1, false], ["tov_o", "TO%", 1, true], ["orb_o", "OR%", 1, false], ["ftr_o", "FTR", 1, false],
      ["tpar_o", "3PA%", 1, false], ["tp_o", "3P%", 1, false], ["twop_o", "2P%", 1, false], ["ast_o", "Ast%", 1, false],
    ],
    defense: [
      ["efg_d", "eFG%", 1, true], ["tov_d", "TO%", 1, false], ["orb_d", "OR%", 1, true], ["ftr_d", "FTR", 1, true],
      ["tpar_d", "3PA%", 1, true], ["tp_d", "3P%", 1, true], ["blk_d", "Blk%", 1, false], ["stl_d", "Stl%", 1, false],
    ],
    roster: [
      ["roster", "Roster", 1, false], ["star", "Best player", 1, false], ["height", "Height", 1, false],
      ["experience", "Exp.", 2, false], ["continuity", "Continuity", 2, false], ["bench", "Bench min", 2, false],
      ["form", "Form", 1, false],
    ],
  };
  let ratingsState = { sort: "rank", dir: 1, view: "core", conf: "", q: "" };

  async function viewRatings() {
    const r = await get("ratings.json");
    await teams();
    const confs = [...new Set(r.teams.map((t) => t.conf).filter(Boolean))].sort();
    app.innerHTML = `
      <div class="page-head">
        <div>
          <h1>Power Ratings</h1>
          <div class="sub">${r.season - 1}–${String(r.season).slice(2)} season · ${r.teams.length} teams · ratings come from a matchup-aware neural network trained on ${"every D-I game since 2008"}, updated ${esc(r.updated)}.</div>
        </div>
      </div>
      <div class="controls">
        <input type="search" id="q" placeholder="Search teams…" value="${esc(ratingsState.q)}" style="width:220px">
        <select id="conf"><option value="">All conferences</option>${confs.map((c) => `<option ${c === ratingsState.conf ? "selected" : ""}>${esc(c)}</option>`).join("")}</select>
        <div class="seg" id="view">${["core", "offense", "defense", "roster"].map((v) => `<button data-v="${v}" class="${v === ratingsState.view ? "on" : ""}">${v[0].toUpperCase() + v.slice(1)}</button>`).join("")}</div>
      </div>
      <div class="card table-wrap"><table id="tbl"></table></div>`;
    $("#q").addEventListener("input", (e) => { ratingsState.q = e.target.value; drawRatings(r); });
    $("#conf").addEventListener("change", (e) => { ratingsState.conf = e.target.value; drawRatings(r); });
    $("#view").addEventListener("click", (e) => {
      const b = e.target.closest("button"); if (!b) return;
      ratingsState.view = b.dataset.v; [...$("#view").children].forEach((x) => x.classList.toggle("on", x === b)); drawRatings(r);
    });
    drawRatings(r);
  }

  function drawRatings(r) {
    const cols = VIEWS[ratingsState.view];
    const all = r.teams;
    const dist = {};
    cols.forEach(([k]) => (dist[k] = all.map((t) => t[k]).filter((v) => v !== null && v !== undefined)));
    let rows = all.filter((t) => (!ratingsState.conf || t.conf === ratingsState.conf) && (!ratingsState.q || t.team.toLowerCase().includes(ratingsState.q.toLowerCase())));
    const k = ratingsState.sort, d = ratingsState.dir;
    rows = rows.slice().sort((a, b) => {
      const x = a[k], y = b[k];
      if (typeof x === "string" || typeof y === "string") return d * String(x ?? "").localeCompare(String(y ?? ""));
      return d * ((x ?? -1e9) - (y ?? -1e9));
    });
    const th = (key, label, cls = "", title = "") => `<th data-k="${key}" class="${cls} ${k === key ? "sorted" : ""}" title="${esc(title)}">${label}${k === key ? (d > 0 ? " ↑" : " ↓") : ""}</th>`;
    const head = `<thead><tr>${th("rank", "#", "rankcol")}${th("team", "Team", "l")}${th("conf", "Conf", "l hide-sm")}${th("wins", "W-L")}${cols.map(([key, lab, , , title]) => th(key, lab, "", title || "")).join("")}</tr></thead>`;
    const body = rows.map((t) => {
      let mv = "";
      if (t.adjem_rank_week_ago && t.adjem_rank) {
        const ch = t.adjem_rank_week_ago - t.adjem_rank;
        if (ch >= 3) mv = `<span class="up" title="AdjEM rank change, 7 days">▲${ch}</span>`;
        else if (ch <= -3) mv = `<span class="down" title="AdjEM rank change, 7 days">▼${-ch}</span>`;
      }
      const cells = cols.map(([key, , dg, inv]) => {
        const v = t[key];
        const p = pctRank(dist[key], v);
        const rk = t[key + "_rank"] ? `<span class="rk">${t[key + "_rank"]}</span>` : "";
        const txt = key === "injury_pts" ? (Math.abs(v) < 0.05 ? "<span class='faint'>—</span>" : `<span class="${v < 0 ? "loss" : "win"}">${sgn(v)}</span>`)
          : key === "power" ? `<span class="big-num">${sgn(v)}</span>` : fmt(v, dg);
        const style = key === "power" || key === "injury_pts" ? "" : heat(p, inv);
        return `<td><span class="heat" style="${style}">${txt}</span>${rk}</td>`;
      }).join("");
      return `<tr><td class="rankcol">${t.rank} ${mv}</td><td class="l">${teamLink(t.team, t)}</td><td class="l hide-sm muted">${esc(t.conf || "")}</td><td class="muted">${t.wins}-${t.losses}</td>${cells}</tr>`;
    }).join("");
    const tbl = $("#tbl");
    tbl.innerHTML = head + `<tbody>${body}</tbody>`;
    tbl.querySelectorAll("th").forEach((h) => h.addEventListener("click", () => {
      const key = h.dataset.k;
      if (ratingsState.sort === key) ratingsState.dir *= -1;
      else { ratingsState.sort = key; ratingsState.dir = ["rank", "team", "conf", "eff_d"].includes(key) || (VIEWS[ratingsState.view].find((c) => c[0] === key) || [])[3] ? 1 : -1; }
      drawRatings(r);
    }));
  }

  // ================================================================ TEAM
  async function viewTeam(slug) {
    const T = await teams();
    const name = Object.keys(T).find((n) => slugify(n) === slug);
    if (!name) { app.innerHTML = `<div class="card pad">Team not found.</div>`; return; }
    const t = T[name];
    const d = await get(`teams/${slug}.json`);
    const all = Object.values(T);
    const color = t.color ? `#${t.color}` : "var(--accent)";
    const profile = [
      ["Shooting (eFG%)", "efg_o", false, "efg_d", true], ["Turnovers", "tov_o", true, "tov_d", false],
      ["Off. rebounding", "orb_o", false, "orb_d", true], ["Free throw rate", "ftr_o", false, "ftr_d", true],
      ["3PA rate", "tpar_o", null, "tpar_d", null], ["3P%", "tp_o", false, "tp_d", true], ["2P%", "twop_o", false, "twop_d", true],
      ["Assist rate", "ast_o", false, "ast_d", true], ["Blocks", "blk_o", true, "blk_d", false], ["Steals", "stl_o", true, "stl_d", false],
    ];
    const bar = (key, inv) => {
      const vals = all.map((x) => x[key]).filter((v) => v != null);
      let p = pctRank(vals, t[key]);
      if (p === null) return `<div></div><div></div>`;
      const shown = inv === null ? p : inv ? 1 - p : p;
      const col = inv === null ? "var(--blue)" : shown >= 0.5 ? "var(--pos)" : "var(--neg)";
      return `<div class="pct-track" title="${Math.round(shown * 100)}th percentile"><span style="width:${Math.max(3, shown * 100)}%;background:${col}"></span></div><div class="small muted" style="text-align:right">${fmt(t[key])}</div>`;
    };
    const statsHtml = [
      ["Power rank", `#${t.rank}`, `${sgn(t.power)} vs avg`], ["Adj. offense", fmt(t.eff_o), `#${t.eff_o_rank}`],
      ["Adj. defense", fmt(t.eff_d), `#${t.eff_d_rank}`], ["Tempo", fmt(t.adjt), `#${t.adjt_rank}`],
      ["Elo", fmt(t.elo, 0), `#${t.elo_rank}`], ["Roster strength", fmt(t.roster), `#${t.roster_rank}`],
      ["Win vs avg team", pct(t.win_vs_avg), "neutral floor"], ["Win vs top-25", pct(t.win_vs_top25), "neutral floor"],
    ].map(([k, v, r]) => `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div><div class="r">${r}</div></div>`).join("");

    const sched = d.schedule.map((g) => {
      const opp = T[g.opp];
      const res = g.played ? `<span class="${g.pts > g.opp_pts ? "win" : "loss"}">${g.pts > g.opp_pts ? "W" : "L"} ${g.pts}-${g.opp_pts}</span>` : `<span class="muted">${g.p != null ? pct(g.p) : ""}</span>`;
      const pre = g.played && g.p != null ? `<span class="faint small">pre ${pct(g.p)}</span>` : g.margin != null && !g.played ? `<span class="faint small">${sgn(g.margin)}</span>` : "";
      return `<tr><td class="l muted">${g.date.slice(5)}</td><td class="l muted">${locWord(g.loc)}</td><td class="l">${opp ? teamLink(g.opp, opp) : esc(g.opp)}</td><td>${res}</td><td>${pre}</td></tr>`;
    }).join("");
    const roster = d.roster.map((p) => {
      const st = p.status ? `<span class="chip ${p.avail < 0.3 ? "neg" : "acc"}">${esc(p.status)}</span>` : p.avail < 0.7 ? `<span class="chip">likely out (missed recent games)</span>` : "";
      const w = Math.min(56, Math.max(2, (p.value + 6) * 3.5));
      return `<tr><td class="l"><span class="teamname">${esc(p.player)}</span> ${st}</td><td class="muted">${esc(p.cls || "")}</td><td class="muted hide-sm">${p.inches ? `${Math.floor(p.inches / 12)}-${p.inches % 12}` : ""}</td><td>${fmt(p.mpg)}</td><td><span class="mini" style="width:${w}px;background:${p.value >= 0 ? "var(--pos)" : "var(--neg)"}"></span> ${sgn(p.value)}</td><td class="hide-sm">${sgn(p.o)}</td><td class="hide-sm">${sgn(p.d)}</td></tr>`;
    }).join("");
    app.innerHTML = `
      <div class="team-hero" style="--team:${color}">
        ${logo(t, "lg")}
        <div>
          <h1>${esc(name)}</h1>
          <div class="badges">
            <span class="chip acc">#${t.rank} MMAI</span><span class="chip">${esc(t.conf || "")}</span>
            <span class="chip">${t.wins}-${t.losses}</span>
            ${Math.abs(t.injury_pts) >= 0.1 ? `<span class="chip ${t.injury_pts < 0 ? "neg" : "pos"}">injury adj ${sgn(t.injury_pts)} pts</span>` : ""}
          </div>
        </div>
        <div style="margin-left:auto" class="hide-sm"><a class="chip" href="#/matchup?a=${encodeURIComponent(name)}">Compare →</a></div>
      </div>
      <div class="stats" style="margin-bottom:16px">${statsHtml}</div>
      <div class="grid cols-2">
        <div class="card pad"><h3>Rating trajectory (AdjEM)</h3><div class="chart" id="chart"></div></div>
        <div class="card pad">
          <h3>Style profile <span class="faint" style="text-transform:none;letter-spacing:0">— national percentile, offense | defense</span></h3>
          ${profile.map(([lab, ko, io, kd, id]) => `<div class="profile-row" style="grid-template-columns:110px 1fr 40px 1fr 40px"><div class="lab">${lab}</div>${bar(ko, io)}${bar(kd, id)}</div>`).join("")}
        </div>
      </div>
      <div class="grid cols-2" style="margin-top:16px">
        <div class="card"><div class="pad" style="padding:16px 20px 4px"><h3>Roster · player values</h3><div class="small muted">Value = points per 100 possessions above an average D-I player (blend of last season and this season's box-score impact).</div></div>
          <div class="table-wrap"><table><thead><tr><th class="l">Player</th><th>Cl</th><th class="hide-sm">Ht</th><th>MPG</th><th>Value</th><th class="hide-sm">Off</th><th class="hide-sm">Def</th></tr></thead><tbody>${roster}</tbody></table></div></div>
        <div class="card"><div class="pad" style="padding:16px 20px 4px"><h3>Schedule</h3></div>
          <div class="table-wrap"><table><tbody>${sched}</tbody></table></div></div>
      </div>`;
    drawLine($("#chart"), d.history.map((h) => ({ x: h.date, y: h.adjem, r: h.rank })), color);
  }

  function drawLine(el, pts, color) {
    pts = pts.filter((p) => p.y != null);
    if (pts.length < 2) { el.innerHTML = `<div class="muted small">Not enough games yet.</div>`; return; }
    const W = 560, H = 220, P = { l: 34, r: 12, t: 12, b: 24 };
    const ys = pts.map((p) => p.y);
    let lo = Math.min(...ys), hi = Math.max(...ys);
    const pad = Math.max(2, (hi - lo) * 0.15); lo -= pad; hi += pad;
    const x = (i) => P.l + (i / (pts.length - 1)) * (W - P.l - P.r);
    const y = (v) => P.t + (1 - (v - lo) / (hi - lo)) * (H - P.t - P.b);
    const path = pts.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p.y).toFixed(1)}`).join("");
    const area = path + `L${x(pts.length - 1)},${H - P.b}L${x(0)},${H - P.b}Z`;
    const ticks = 4;
    let grid = "";
    for (let i = 0; i <= ticks; i++) {
      const v = lo + ((hi - lo) * i) / ticks;
      grid += `<line class="axis" x1="${P.l}" x2="${W - P.r}" y1="${y(v)}" y2="${y(v)}" stroke-dasharray="2 3"/><text x="${P.l - 6}" y="${y(v) + 3}" text-anchor="end">${v.toFixed(0)}</text>`;
    }
    const labels = [0, Math.floor(pts.length / 2), pts.length - 1].map((i) => `<text x="${x(i)}" y="${H - 6}" text-anchor="middle">${pts[i].x.slice(5)}</text>`).join("");
    const last = pts[pts.length - 1];
    el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Rating history">
      <defs><linearGradient id="g1" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="${color}" stop-opacity=".25"/><stop offset="1" stop-color="${color}" stop-opacity="0"/></linearGradient></defs>
      ${grid}<path d="${area}" fill="url(#g1)"/><path d="${path}" fill="none" stroke="${color}" stroke-width="2.2" stroke-linejoin="round"/>
      <circle cx="${x(pts.length - 1)}" cy="${y(last.y)}" r="4" fill="${color}"/>${labels}
      <text x="${x(pts.length - 1) - 6}" y="${y(last.y) - 9}" text-anchor="end" style="fill:var(--text);font-weight:600">${last.y.toFixed(1)} · #${last.r}</text></svg>`;
  }

  // ================================================================ BRACKET
  let bracketMode = "likely";
  async function viewBracket() {
    const b = await get("bracket.json");
    const T = await teams();
    const tree = b[bracketMode];
    const regionsByName = {};
    (function collect(n) {
      if (!n) return;
      if (n.round === 4 && n.region) regionsByName[n.region] = n;
      (n.children || []).forEach(collect);
    })(tree);
    const ff = tree.children;
    const champ = tree.pick ? T[tree.pick] : null;
    const adv = b.advancement.slice(0, 80);
    const opening = [];
    (function findOpen(n) { if (!n) return; if (n.round === 0 && n.children) opening.push(n); (n.children || []).forEach(findOpen); })(tree);

    app.innerHTML = `
      <div class="page-head">
        <div><h1>${b.projected ? "Projected" : ""} ${b.season} NCAA Tournament</h1>
          <div class="sub">${b.field_size}-team field. Every game is priced by the matchup network with each team's travel distance to the actual venue. Advancement odds are computed exactly by propagating win probabilities through the bracket.</div></div>
        <div class="seg" id="mode"><button data-m="likely" class="${bracketMode === "likely" ? "on" : ""}">Most likely</button><button data-m="pool" class="${bracketMode === "pool" ? "on" : ""}">Pool pick</button></div>
      </div>
      ${b.projected ? `<div class="notice">${esc(b.note)} The real bracket replaces this automatically after Selection Sunday.</div>` : ""}
      <div class="grid cols-3" style="margin-bottom:22px">
        <div class="card pad champ-card"><h3>Champion pick</h3>${champ ? logo(champ, "lg") : ""}<div style="font-size:22px;font-weight:800;margin-top:6px">${esc(tree.pick || "—")}</div>
          <div class="muted">${tree.pick ? pct((b.advancement.find((a) => a.team === tree.pick) || {}).champ, 1) + " to win it all" : ""}</div></div>
        <div class="card pad" style="grid-column: span 2"><h3>Title odds</h3>${adv.slice(0, 8).map((a) => `
          <div style="display:grid;grid-template-columns:200px 1fr 54px;gap:10px;align-items:center;padding:4px 0">
            ${teamLink(a.team, T[a.team])}<div class="pbar"><span style="width:${(100 * a.champ / adv[0].champ).toFixed(1)}%"></span></div><div style="text-align:right;font-weight:600">${pct(a.champ, 1)}</div></div>`).join("")}</div>
      </div>
      <h2>Final Four · ${esc(ff[0] && ff[0].site || "")}</h2>
      <div class="ff" style="margin-bottom:26px">${ffCard(ff[0], T)}${matchCard(tree, T, "Championship")}${ffCard(ff[1], T)}</div>
      ${opening.length ? `<h2>Opening Round</h2><div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:10px;margin-bottom:26px">${opening.map((n) => matchCard(n, T)).join("")}</div>` : ""}
      ${b.regions.map((r) => regionHtml(regionsByName[r], r, T)).join("")}
      <h2 style="margin-top:10px">Advancement odds</h2>
      <div class="card table-wrap"><table><thead><tr><th class="l">Team</th><th>Seed</th><th class="l hide-sm">Region</th><th>R32</th><th>S16</th><th>E8</th><th>F4</th><th>Final</th><th>Champ</th></tr></thead>
        <tbody>${b.advancement.map((a) => `<tr><td class="l">${teamLink(a.team, T[a.team])}</td><td class="muted">${a.seed ?? ""}</td><td class="l muted hide-sm">${esc(a.region || "")}</td>
          ${["r32", "s16", "e8", "f4", "final", "champ"].map((k) => `<td><span class="heat" style="${heat(Math.min(1, a[k] * (k === "champ" ? 4 : k === "final" ? 3 : k === "f4" ? 2 : 1)))}">${a[k] < 0.001 ? "<span class='faint'>&lt;0.1%</span>" : pct(a[k], 1)}</span></td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
    $("#mode").addEventListener("click", (e) => { const x = e.target.closest("button"); if (!x) return; bracketMode = x.dataset.m; viewBracket(); });
  }

  function entrants(n) {
    // the two teams contesting node n under the current picks
    return (n.children || []).map((c) => c.pick || c.team);
  }
  function matchCard(n, T, title = "") {
    if (!n) return "";
    const [a, b] = entrants(n);
    const top = Object.fromEntries(n.top || []);
    const seedOf = (c) => {
      let s = null;
      (function f(x) { if (!x || s) return; if (x.team === c) s = x.seed; (x.children || []).forEach(f); })(n);
      return s;
    };
    const row = (t) => t ? `<div class="row ${n.pick === t ? "w" : ""}"><span class="seed">${seedOf(t) ?? ""}</span>${logo(T[t])}<span class="ellip">${esc(t)}</span><span class="p">${top[t] != null ? pct(top[t]) : ""}</span></div>` : `<div class="row"><span></span><span></span><span class="faint">TBD</span><span></span></div>`;
    return `<div class="mu">${title ? `<div class="site" style="border-top:0;border-bottom:1px solid var(--border)">${title}</div>` : ""}${row(a)}${row(b)}${n.site ? `<div class="site">${esc(n.site)}${n.pick_prob ? ` · pick ${pct(n.pick_prob)}` : ""}</div>` : ""}</div>`;
  }
  function ffCard(n, T) { return matchCard(n, T, "National semifinal"); }
  function regionHtml(root, name, T) {
    if (!root) return "";
    const byRound = { 1: [], 2: [], 3: [], 4: [] };
    (function walk(n) {
      if (!n || n.round === 0 && !n.children) return;
      if (n.round >= 1) byRound[n.round].push(n);
      (n.children || []).forEach(walk);
    })(root);
    // order: in-order traversal keeps bracket order
    const ordered = { 1: [], 2: [], 3: [], 4: [] };
    (function walk(n) { if (!n || !n.children) return; walk(n.children[0]); if (n.round >= 1) ordered[n.round].push(n); walk(n.children[1]); })(root);
    const winner = root.pick;
    return `<div class="region"><div class="region-title"><h2 style="margin:0">${esc(name)}</h2><span class="muted small">${esc(ordered[3][0] && ordered[3][0].site || "")}</span></div>
      <div class="bracket-scroll"><div class="rounds rounds-head">${[1, 2, 3, 4].map((r) => `<h4>${ROUND_NAMES[r]}</h4>`).join("")}<h4>Regional champion</h4></div>
      <div class="rounds">
        ${[1, 2, 3, 4].map((r) => `<div class="round-col">${ordered[r].map((n) => matchCard(n, T)).join("")}</div>`).join("")}
        <div class="round-col"><div class="mu"><div class="row w"><span></span>${logo(T[winner])}<span class="ellip">${esc(winner || "")}</span><span></span></div></div></div>
      </div></div></div>`;
  }

  // ================================================================ MATCHUP (in-browser network)
  let MODEL = null, VEC = null;
  const gelu = (x) => 0.5 * x * (1 + Math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
  const softplus = (x) => (x > 20 ? x : Math.log1p(Math.exp(x)));
  function dense(W, b, x, act) {
    const out = new Array(W.length);
    for (let i = 0; i < W.length; i++) {
      let s = b[i]; const w = W[i];
      for (let j = 0; j < x.length; j++) s += w[j] * x[j];
      out[i] = act ? act(s) : s;
    }
    return out;
  }
  function erf(x) { // Abramowitz-Stegun
    const t = 1 / (1 + 0.3275911 * Math.abs(x));
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return x >= 0 ? y : -y;
  }
  const Phi = (z) => 0.5 * (1 + erf(z / Math.SQRT2));
  function netForward(net, xa, xb, ctx) {
    const enc = (x) => dense(net.enc1[0], net.enc1[1], dense(net.enc0[0], net.enc0[1], x, gelu), gelu);
    const ea = enc(xa), eb = enc(xb);
    const prod = ea.map((v, i) => v * eb[i]);
    const ctxS = ctx.slice(); ctxS[0] = -ctxS[0];
    const pair = (u, v, c) => dense(net.pair1[0], net.pair1[1], dense(net.pair0[0], net.pair0[1], [...u, ...v, ...prod, ...c], gelu))[0];
    let mu = pair(ea, eb, ctx) - pair(eb, ea, ctxS) + net.hca * ctx[0];
    for (let j = 0; j < xa.length; j++) mu += net.skip[j] * (xa[j] - xb[j]);
    const sym = [...ea.map((v, i) => v + eb[i]), ...prod, ...ctx.slice(1)];
    const s = dense(net.sig1[0], net.sig1[1], dense(net.sig0[0], net.sig0[1], sym, gelu))[0];
    return [mu, 6 + softplus(s + 4)];
  }
  function haversine(a, b) {
    if (!a || !b || a[0] == null || b[0] == null) return 800;
    const R = 6371, r = Math.PI / 180;
    const dl = (b[0] - a[0]) * r, dn = (b[1] - a[1]) * r;
    const h = Math.sin(dl / 2) ** 2 + Math.cos(a[0] * r) * Math.cos(b[0] * r) * Math.sin(dn / 2) ** 2;
    return 2 * R * Math.asin(Math.sqrt(h));
  }
  function predictMatchup(A, B, opts) {
    const F = MODEL.features, iRest = F.indexOf("rest"), iDist = F.indexOf("dist");
    const z = (i, v) => Math.max(-6, Math.min(6, (v - MODEL.norm.mean[i]) / MODEL.norm.std[i]));
    const build = (name, venue) => {
      const x = VEC.teams[name].slice();
      x[iRest] = z(iRest, 4);
      const d = opts.loc !== 0 ? (venue === name ? 0 : haversine(VEC.loc[opts.home], VEC.loc[name])) : opts.venue ? haversine([opts.venue.lat, opts.venue.lon], VEC.loc[name]) : 800;
      x[iDist] = z(iDist, Math.log1p(d));
      return x;
    };
    const xa = build(A, opts.home), xb = build(B, opts.home);
    const same = VEC.conf[A] && VEC.conf[A] === VEC.conf[B] ? 1 : 0;
    const ctx = [opts.loc, opts.loc === 0 ? 1 : 0, opts.type === "ncaa" ? 1 : 0, opts.type === "conf" ? 1 : 0, opts.type === "ncaa" ? 1 : 0, same, (opts.type === "ncaa" ? 138 : VEC.day) / 100];
    const mus = [], sg2 = [];
    for (const net of MODEL.nets) { const [m, s] = netForward(net, xa, xb, ctx); mus.push(m); sg2.push(s * s); }
    const mu = mus.reduce((a, b) => a + b, 0) / mus.length;
    const vb = mus.reduce((a, b) => a + (b - mu) ** 2, 0) / mus.length;
    const sigma = Math.sqrt(sg2.reduce((a, b) => a + b, 0) / sg2.length + vb);
    return { mu, sigma, p: Phi(mu / sigma) };
  }

  async function viewMatchup(_, params) {
    [MODEL, VEC] = await Promise.all([get("model.json"), get("vectors.json")]);
    const T = await teams();
    const names = Object.keys(VEC.teams).filter((n) => T[n]).sort();
    const top = Object.values(T).sort((a, b) => a.rank - b.rank);
    const A0 = params.get("a") && T[params.get("a")] ? params.get("a") : top[0].team;
    const B0 = params.get("b") && T[params.get("b")] ? params.get("b") : top[1].team;
    app.innerHTML = `
      <div class="page-head"><div><h1>Matchup Lab</h1><div class="sub">Any two teams, any site. The neural network runs in your browser. Travel distance to the venue, injuries and style interactions are included.</div></div></div>
      <div class="card pad">
        <datalist id="teamlist">${names.map((n) => `<option value="${esc(n)}">`).join("")}</datalist>
        <div class="mu-pickers">
          <div class="picker"><label>Team A</label><input type="text" id="ta" list="teamlist" value="${esc(A0)}"></div>
          <div class="vs">vs</div>
          <div class="picker"><label>Team B</label><input type="text" id="tb" list="teamlist" value="${esc(B0)}"></div>
        </div>
        <div class="controls" style="margin:14px 0 0">
          <select id="site"><option value="neutral">Neutral site (no travel info)</option><option value="a">Team A home</option><option value="b">Team B home</option>
            ${VEC.venues.map((v, i) => `<option value="v${i}">${esc(v.group)}: ${esc(v.city)}</option>`).join("")}</select>
          <select id="gtype"><option value="ncaa">NCAA tournament</option><option value="reg">Regular season</option><option value="conf">Conference tournament</option></select>
          <button class="icon-btn" id="swap" title="Swap teams" style="width:auto;padding:0 12px">⇄ swap</button>
        </div>
      </div>
      <div id="mres" style="margin-top:16px"></div>`;
    const go = () => {
      const A = $("#ta").value, B = $("#tb").value;
      if (!VEC.teams[A] || !VEC.teams[B] || A === B) { $("#mres").innerHTML = `<div class="card pad muted">Pick two different teams.</div>`; return; }
      const sv = $("#site").value, type = $("#gtype").value;
      const opts = { loc: 0, type, home: null, venue: null };
      if (sv === "a") { opts.loc = 1; opts.home = A; }
      else if (sv === "b") { opts.loc = -1; opts.home = B; }
      else if (sv.startsWith("v")) opts.venue = VEC.venues[+sv.slice(1)];
      history.replaceState(null, "", `#/matchup?a=${encodeURIComponent(A)}&b=${encodeURIComponent(B)}`);
      renderMatchup(A, B, opts, T);
    };
    ["ta", "tb"].forEach((id) => $("#" + id).addEventListener("change", go));
    $("#site").addEventListener("change", go); $("#gtype").addEventListener("change", go);
    $("#swap").addEventListener("click", () => { const a = $("#ta").value; $("#ta").value = $("#tb").value; $("#tb").value = a; go(); });
    go();
  }

  function renderMatchup(A, B, opts, T) {
    const r = predictMatchup(A, B, opts);
    const neutral = predictMatchup(A, B, { loc: 0, type: opts.type, venue: null });
    const neutralReg = predictMatchup(A, B, { loc: 0, type: "reg", venue: null }); // same context as Power
    const ta = T[A], tb = T[B];
    const lg = Object.values(T).reduce((s, t) => s + t.eff_o, 0) / Object.keys(T).length;
    const tempo = (ta.adjt + tb.adjt) / 2;
    const total = ((ta.eff_o + tb.eff_d - lg) + (tb.eff_o + ta.eff_d - lg)) * tempo / 100;
    const pa = total / 2 + r.mu / 2, pb = total / 2 - r.mu / 2;
    const edge = neutralReg.mu - (ta.power - tb.power);
    const rgb = (h) => h && /^[0-9a-f]{6}$/i.test(h) ? [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16)) : null;
    const close = (x, y) => !x || !y || Math.hypot(x[0] - y[0], x[1] - y[1], x[2] - y[2]) < 110;
    const ca = ta.color ? `#${ta.color}` : "#f97316";
    let cbHex = tb.color;
    if (close(rgb(ta.color), rgb(cbHex))) cbHex = !close(rgb(ta.color), rgb(tb.alt_color)) && tb.alt_color && tb.alt_color.toLowerCase() !== "ffffff" ? tb.alt_color : (close(rgb(ta.color), [249, 115, 22]) ? "6b7280" : "f97316");
    const cb = cbHex ? `#${cbHex}` : "#3b6fe0";
    const all = Object.values(T);
    const P = (k, v) => pctRank(all.map((t) => t[k]).filter((x) => x != null), v);
    // offense-vs-defense style duels (positive = advantage to the offense)
    const duels = [
      ["Shooting", "efg_o", 1, "efg_d", 1], ["Ball security", "tov_o", -1, "tov_d", -1], ["Off. rebounding", "orb_o", 1, "orb_d", 1],
      ["Getting to the line", "ftr_o", 1, "ftr_d", 1], ["3-pt shooting", "tp_o", 1, "tp_d", 1], ["Inside scoring", "twop_o", 1, "twop_d", 1],
    ];
    const duel = (off, def) => duels.map(([lab, ko, so, kd, sd]) => {
      const po = so > 0 ? P(ko, off[ko]) : 1 - P(ko, off[ko]);       // offense quality pct
      const pd = sd > 0 ? 1 - P(kd, def[kd]) : P(kd, def[kd]);       // defense quality pct
      return { lab, po, pd, adv: po - pd };
    }).sort((x, y) => Math.abs(y.adv) - Math.abs(x.adv)).slice(0, 4);
    const duelHtml = (off, def, offName, defName) => duel(off, def).map((d) => `
      <div class="row"><div><b>${esc(offName)}</b> ${esc(d.lab.toLowerCase())} <span class="faint">(${Math.round(d.po * 100)}th pct)</span></div>
      <div class="cmp-bar"><span style="width:${d.po * 50}%;background:${d.adv > 0 ? "var(--pos)" : "var(--faint)"}"></span><span style="width:${(1 - d.pd) * 50}%;background:transparent"></span></div>
      <div style="text-align:right"><b>${esc(defName)}</b> D <span class="faint">(${Math.round(d.pd * 100)}th pct)</span> ${d.adv > 0.25 ? `<span class="chip pos">edge O</span>` : d.adv < -0.25 ? `<span class="chip neg">edge D</span>` : ""}</div></div>`).join("");
    const inj = (t) => Math.abs(t.injury_pts) >= 0.1 ? `<span class="chip ${t.injury_pts < 0 ? "neg" : "pos"}">injuries ${sgn(t.injury_pts)} pts</span>` : `<span class="chip">full strength</span>`;
    $("#mres").innerHTML = `
      <div class="card pad">
        <div style="display:flex;justify-content:space-between;align-items:center;gap:10px">
          <a href="#/team/${slugify(A)}" class="teamcell">${logo(ta, "md")}<div><div class="teamname">${esc(A)}</div><div class="small muted">#${ta.rank} · ${esc(ta.conf)} · ${inj(ta)}</div></div></a>
          <a href="#/team/${slugify(B)}" class="teamcell" style="justify-content:flex-end;text-align:right"><div><div class="teamname">${esc(B)}</div><div class="small muted">#${tb.rank} · ${esc(tb.conf)} · ${inj(tb)}</div></div>${logo(tb, "md")}</a>
        </div>
        <div class="score"><div class="n">${Math.round(pa)}</div><div class="muted">projected</div><div class="n">${Math.round(pb)}</div></div>
        <div class="split" style="margin-top:10px"><div style="width:${(r.p * 100).toFixed(1)}%;background:${ca}">${pct(r.p, 1)}</div><div style="width:${(100 - r.p * 100).toFixed(1)}%;background:${cb}">${pct(1 - r.p, 1)}</div></div>
        <div class="stats" style="margin-top:16px">
          <div class="stat"><div class="k">Predicted margin</div><div class="v">${sgn(r.mu)}</div><div class="r">${esc(A)} perspective</div></div>
          <div class="stat"><div class="k">Uncertainty (σ)</div><div class="v">${fmt(r.sigma)}</div><div class="r">points</div></div>
          <div class="stat"><div class="k">Power gap</div><div class="v">${sgn(ta.power - tb.power)}</div><div class="r">neutral, from ratings</div></div>
          <div class="stat"><div class="k">Matchup edge</div><div class="v">${sgn(edge)}</div><div class="r">${Math.abs(edge) < 0.3 ? "styles neutral" : `favors ${esc(edge > 0 ? A : B)}`}</div></div>
          <div class="stat"><div class="k">Site effect</div><div class="v">${sgn(r.mu - neutral.mu)}</div><div class="r">home court / travel / context</div></div>
        </div>
      </div>
      <div class="grid cols-2" style="margin-top:16px">
        <div class="card pad edge-list"><h3>${esc(A)} offense vs ${esc(B)} defense</h3>${duelHtml(ta, tb, A, B)}</div>
        <div class="card pad edge-list"><h3>${esc(B)} offense vs ${esc(A)} defense</h3>${duelHtml(tb, ta, B, A)}</div>
      </div>
      <div class="small muted" style="margin-top:10px">Matchup edge = the network's neutral-site margin minus the gap in power ratings. It is the part of the forecast that comes from how these two teams' styles interact, beyond their overall strength.</div>`;
  }

  // ================================================================ GAMES
  async function viewGames() {
    const g = await get("games.json");
    const T = await teams();
    const byDate = {};
    g.upcoming.forEach((x) => (byDate[x.date] = byDate[x.date] || []).push(x));
    const recent = g.recent.slice(-150).reverse();
    const acc = g.season_accuracy;
    const gameRow = (x, done) => {
      const fav = x.p_a >= 0.5;
      const ta = T[x.team_a], tb = T[x.team_b];
      const mid = done
        ? `<div class="prob">${x.pts_a}-${x.pts_b}</div><div class="small ${((x.pts_a > x.pts_b) === fav) ? "win" : "loss"}">${((x.pts_a > x.pts_b) === fav) ? "✓ called" : "✗ upset"} · pre ${pct(Math.max(x.p_a, 1 - x.p_a))}</div>`
        : `<div class="prob">${pct(Math.max(x.p_a, 1 - x.p_a))}</div><div class="small muted">${esc(fav ? x.team_a : x.team_b)} ${sgn(Math.abs(x.margin))}</div>`;
      return `<div class="game">
        <div class="side">${logo(ta)}<a class="ellip ${fav ? "teamname" : ""}" href="#/team/${slugify(x.team_a)}">${esc(x.team_a)}</a><span class="faint small hide-sm">${ta ? "#" + ta.rank : ""}</span></div>
        <div class="mid">${mid}<div class="pbar" style="margin-top:6px"><span style="width:${(x.p_a * 100).toFixed(0)}%"></span></div></div>
        <div class="side r"><span class="faint small hide-sm">${tb ? "#" + tb.rank : ""}</span><a class="ellip ${!fav ? "teamname" : ""}" href="#/team/${slugify(x.team_b)}">${x.loc === -1 ? "@ " : x.loc === 0 ? "" : ""}${esc(x.team_b)}</a>${logo(tb)}</div></div>`;
    };
    app.innerHTML = `
      <div class="page-head"><div><h1>Games</h1><div class="sub">Win probabilities for the next 7 days, plus how the model's pre-game forecasts have done this season.</div></div></div>
      ${acc ? `<div class="stats" style="margin-bottom:18px">
        <div class="stat"><div class="k">Games this season</div><div class="v">${acc.n}</div></div>
        <div class="stat"><div class="k">Winners called</div><div class="v">${pct(acc.accuracy, 1)}</div></div>
        <div class="stat"><div class="k">Avg margin error</div><div class="v">${fmt(acc.mae)}</div><div class="r">points</div></div>
        <div class="stat"><div class="k">Log loss</div><div class="v">${fmt(acc.log_loss, 3)}</div></div></div>` : ""}
      <div class="grid cols-2">
        <div class="card"><div class="pad" style="padding:16px 20px 8px"><h3>Upcoming</h3></div>
          ${Object.keys(byDate).length ? Object.entries(byDate).map(([d, xs]) => `<div class="date-head">${new Date(d + "T12:00:00").toLocaleDateString(undefined, { weekday: "long", month: "short", day: "numeric" })}</div>${xs.sort((a, b) => Math.abs(a.p_a - .5) - Math.abs(b.p_a - .5)).map((x) => gameRow(x, false)).join("")}`).join("") : `<div class="pad muted">No D-I games scheduled in the next week.</div>`}
        </div>
        <div class="card"><div class="pad" style="padding:16px 20px 8px"><h3>Recent results</h3></div>${recent.map((x) => gameRow(x, true)).join("")}</div>
      </div>`;
  }

  // ================================================================ METHOD
  async function viewMethod() {
    let bt = null;
    try { bt = await get("backtest.json"); } catch (e) {}
    const order = ["MMAI net", "Efficiency only", "Elo", "Seed", "T-Rank (pre-game)", "MMAI net (T-Rank games)"];
    const ov = bt ? bt.overall : {};
    const best = Math.min(...order.filter((k) => ov[k]).map((k) => ov[k].log_loss));
    const btTable = bt ? `<div class="card table-wrap" style="margin:12px 0 4px"><table><thead><tr><th class="l">Model</th><th>Games</th><th>Log loss ↓</th><th>Brier ↓</th><th>Accuracy ↑</th></tr></thead><tbody>
      ${order.filter((k) => ov[k]).map((k) => `<tr><td class="l">${esc(k)}</td><td>${ov[k].n}</td><td class="${ov[k].log_loss === best ? "best" : ""}">${fmt(ov[k].log_loss, 4)}</td><td>${fmt(ov[k].brier, 4)}</td><td>${pct(ov[k].accuracy, 1)}</td></tr>`).join("")}
      </tbody></table></div>
      <div class="card table-wrap" style="margin-top:12px"><table><thead><tr><th class="l">Tournament</th>${["MMAI net", "Efficiency only", "Elo", "Seed", "T-Rank (pre-game)"].map((k) => `<th>${esc(k)}</th>`).join("")}</tr></thead><tbody>
      ${bt.per_season.map((s) => { const vals = ["MMAI net", "Efficiency only", "Elo", "Seed", "T-Rank (pre-game)"].map((k) => s[k] ? s[k].log_loss : null); const m = Math.min(...vals.filter((v) => v != null));
        return `<tr><td class="l">${s.season}</td>${vals.map((v) => `<td class="${v === m ? "best" : ""}">${v == null ? "—" : fmt(v, 3)}</td>`).join("")}</tr>`; }).join("")}
      </tbody></table></div><div class="small muted" style="margin-top:6px">Log loss per tournament (lower is better). T-Rank lines are only available from 2015.</div>` : `<p class="muted">Backtest not run yet.</p>`;
    app.innerHTML = `<div class="prose">
      <div class="page-head"><div><h1>How it works</h1><div class="sub">A refreshable, zero-cost pipeline: public data → point-in-time ratings → a matchup-aware neural network → bracket.</div></div></div>
      <h2>Backtest: NCAA tournaments 2012–2026</h2>
      <p>For each tournament, every model was fit <b>only on games played before that tournament started</b>. Games were then scored using availability that could be known before tip-off (recent missed games), not the actual lineups.</p>
      ${btTable}
      <h2>1 · Data (all public, no keys)</h2>
      <ul>
        <li><b>barttorvik.com</b>: every D-I game since 2008 with full box scores, plus ~110k player-game logs per season (minutes, usage, box-score plus/minus).</li>
        <li><b>ESPN</b> public scoreboard API: venues (city/state), neutral-site flags, NCAA seeds/regions/rounds, logos and colors.</li>
        <li><b>Covers.com / Rotowire</b> public injury pages, plus a hand-editable <code>overrides/injuries.yaml</code>.</li>
      </ul>
      <h2>2 · Adjusted efficiency (the KenPom-style layer)</h2>
      <p>For every team-game, points per 100 possessions are modeled as <code>league mean + Off[team] + Def[opponent] + home·loc</code>. The model is a recency-weighted ridge regression (45-day half-life). Each team is pulled toward a <b>preseason prior</b> learned from last season's rating and the player values of the current roster, so transfers count. The same solve gives opponent-adjusted style profiles: eFG%, turnover rate, rebounding, free-throw rate, 3-point rate and accuracy, 2P%, blocks, steals and assists, for offense and defense. Ratings are re-solved before every game day, so all historical features are leak-free.</p>
      <h2>3 · Players and injuries</h2>
      <p>Each player's value is a Bayesian blend: last season's box-score impact (plus class progression), recruiting rank for newcomers, then this season's minute-weighted game impact. Team <b>roster strength</b> = Σ minutes share × value. Ratings reflect the roster that actually played, so the injury adjustment is <code>Σ (availability now − historical participation) × minutes share × (value − replacement)</code>. A star who has played every game and is now out costs his team a lot. A player who has been out for weeks is already priced in.</p>
      <h2>4 · The matchup network</h2>
      <p>An encoder maps each team's ~45 features (efficiency, style O/D, roster, Elo, form, availability, rest, travel) to an embedding. A pair head scores <code>[A, B, A⊙B, context]</code>, so it can learn style interactions such as elite offensive rebounding against weak defensive rebounding. Margin = g(A,B) − g(B,A), which makes the model exactly antisymmetric. A second head predicts game-specific uncertainty σ, and P(win) = Φ(margin/σ). Context includes home/neutral/away, <b>each team's distance to the venue</b> (tournament proximity), NCAA/conference-tournament flags and same-conference games. Five seeds are averaged.</p>
      <h2>5 · Ratings from the model</h2>
      <p>The <b>Power</b> rating is each team's average predicted margin against every other D-I team on a neutral floor, computed by running the network on all ~66,000 pairs. <b>Matchup edge</b> is how far a specific forecast departs from the gap in power ratings.</p>
      <h2>6 · Bracket</h2>
      <p>2027 uses the new <b>76-team</b> format with a 12-game Opening Round (Dayton and Wichita). Until Selection Sunday, the field is projected: conference champions (or the best team in each conference) plus the best at-larges, seeded on an S-curve, with protected seeds placed at the nearest first-round site. After the bracket is announced it's read from ESPN automatically. Advancement odds are exact, from propagating each game's probability through the tree with venue-specific travel.</p>
      <h2>Refresh</h2>
      <p>A GitHub Action runs <code>python -m mmai daily</code> every morning and retrains weekly (<code>python -m mmai train</code>). The whole site is static JSON plus this page.</p>
    </div>`;
  }

  route();
})();
