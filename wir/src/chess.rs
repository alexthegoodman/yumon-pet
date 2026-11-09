//! WIR Chess: a written attention network that judges chess positions.
//!
//! The network reads the 64 squares (one token per square, `pos` = square, a1 = 0,
//! h1 = 7, a8 = 56) with `piece` in 0..12 (0 empty, 1-6 white P N B R Q K, 7-12 black)
//! and always judges from the side to move as white: the host mirrors the board
//! (square ^ 56, colours swapped) when black is to move. It computes:
//!
//! - attack maps: per square, the first piece in each of 8 directions (one graded head
//!   per direction: on the ray, occupied, closest wins) and whether it slides along that
//!   line, plus knight, pawn and king attackers (count heads over square-pair tables);
//!   attackers per side and each side's lowest attacker value;
//! - per-square terms: material + piece-square values (opening and endgame), mobility
//!   (attacked squares not holding an own piece), king zone attacks, hanging pieces
//!   (attacked and undefended, or attacked by something cheaper), passed pawns, rooks on
//!   open files;
//! - totals (one shared mean head over all squares) and the phase blend
//!   EG + phase (MG - EG) / 24, with phase x (MG - EG) summed by another head.
//!
//! `eval` = pst + mobility + king + hanging + passed + rooks with the default style
//! weights; the term totals are outputs too, so a host can recombine them (that is
//! editing the last layer). `reference` computes the same numbers in plain code.

use crate::seq::{self, SeqProgram, SeqRunner, SeqWeights};

pub const EMPTY: u8 = 0;
pub const PIECES: &str = ".PNBRQKpnbrqk";
/// Material, opening and endgame (centipawns), by piece type P N B R Q K.
pub const MAT_MG: [i64; 6] = [82, 337, 365, 477, 1025, 0];
pub const MAT_EG: [i64; 6] = [94, 281, 297, 512, 936, 0];
/// Values for hanging pieces (centipawns) and for lowest-attacker comparisons (pawns).
pub const HANG_VAL: [i64; 6] = [100, 320, 330, 500, 900, 0];
pub const UNIT_VAL: [i64; 6] = [1, 3, 3, 5, 9, 20];
pub const PHASE_W: [i64; 6] = [0, 1, 1, 2, 4, 0];
/// Passed pawn bonus by ranks advanced (0..7).
pub const PASSED: [i64; 8] = [0, 5, 10, 20, 35, 60, 100, 0];
/// Default style weights of the term totals (eval = pst + sum of weight x term).
pub const STYLE: [(&str, f64); 7] = [
    ("pst", 1.0),
    ("mobility", 4.0),
    ("king", 8.0),
    ("hang_us", -0.15),
    ("hang_them", 0.5),
    ("passed", 1.0),
    ("rooks", 20.0),
];
/// Direction steps (file, rank): N S E W NE NW SE SW. 0-3 orthogonal, 4-7 diagonal.
pub const DIRS: [(i64, i64); 8] = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)];

fn kind(p: u8) -> Option<usize> {
    (p != 0).then(|| (p as usize - 1) % 6)
}

fn white(p: u8) -> bool {
    (1..=6).contains(&p)
}

fn black(p: u8) -> bool {
    p >= 7
}

fn centre(s: usize) -> i64 {
    let (f, r) = ((s % 8) as i64, (s / 8) as i64);
    f.min(7 - f) + r.min(7 - r)
}

/// Piece-square value (material included) for a white piece of kind k on square s.
pub fn pst(k: usize, s: usize, endgame: bool) -> i64 {
    let (f, r) = ((s % 8) as i64, (s / 8) as i64);
    let c = centre(s);
    let mat = if endgame { MAT_EG[k] } else { MAT_MG[k] };
    let pos = match (k, endgame) {
        (0, false) => [0, 0, 5, 10, 20, 35, 55, 0][r as usize] + if (3..=4).contains(&f) && (2..=4).contains(&r) { 15 } else { 0 },
        (0, true) => [0, 0, 10, 20, 35, 60, 100, 0][r as usize],
        (1, false) => -40 + 10 * c,
        (1, true) => -30 + 8 * c,
        (2, _) => -10 + 4 * c,
        (3, false) => if r == 6 { 10 } else { 0 },
        (3, true) => 0,
        (4, false) => -5 + 2 * c,
        (4, true) => -10 + 5 * c,
        (5, false) => if r == 0 { [20, 30, 10, 0, 0, 10, 30, 20][f as usize] } else { -30 - 10 * r },
        (5, true) => -50 + 12 * c,
        _ => 0,
    };
    mat + pos
}

/// Signed piece-square value (white positive) of piece p on square s.
pub fn pst_signed(p: u8, s: usize, endgame: bool) -> i64 {
    match kind(p) {
        None => 0,
        Some(k) if white(p) => pst(k, s, endgame),
        Some(k) => -pst(k, s ^ 56, endgame),
    }
}

/// max(..) as a balanced tree (a left fold would add one layer per argument).
fn max_tree(xs: &[String]) -> String {
    match xs.len() {
        1 => xs[0].clone(),
        2 => format!("max({}, {})", xs[0], xs[1]),
        n => format!("max({}, {})", max_tree(&xs[..n / 2]), max_tree(&xs[n / 2..])),
    }
}

fn list(v: impl IntoIterator<Item = i64>) -> String {
    format!("[{}]", v.into_iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "))
}

fn piece_table(f: impl Fn(u8) -> i64) -> String {
    list((0..13u8).map(f))
}

/// The WIR 0.3 source of the chess network.
pub fn program() -> String {
    let mut s = String::from(
        "wir 0.3
# WIR Chess: judges a position from the side to move (always white; the host mirrors).
# Squares a1 = 0 .. h8 = 63; piece 0 empty, 1-6 white P N B R Q K, 7-12 black.
sequence len 64
input piece: token in [0, 12]

# --- geometry (compile-time tables over squares)
table steps(a, b) = max(abs(mod(b, 8) - mod(a, 8)), abs(div(b, 8) - div(a, 8)))
",
    );
    s += &format!("table ddf(d) = {}\n", list(DIRS.iter().map(|d| d.0)));
    s += &format!("table ddr(d) = {}\n", list(DIRS.iter().map(|d| d.1)));
    s += "table onray(d, q, k) = and(gt(steps(q, k), 0), eq(mod(k, 8) - mod(q, 8), ddf(d) * steps(q, k)), eq(div(k, 8) - div(q, 8), ddr(d) * steps(q, k)))
table near(d, q, k) = select(onray(d, q, k), 8 - steps(q, k), 0)
table knight(q, k) = or(and(eq(abs(mod(k, 8) - mod(q, 8)), 1), eq(abs(div(k, 8) - div(q, 8)), 2)), and(eq(abs(mod(k, 8) - mod(q, 8)), 2), eq(abs(div(k, 8) - div(q, 8)), 1)))
table kadj(q, k) = eq(steps(q, k), 1)
table wpawn_hits(q, k) = and(eq(div(q, 8) - div(k, 8), 1), eq(abs(mod(q, 8) - mod(k, 8)), 1))
table bpawn_hits(q, k) = and(eq(div(k, 8) - div(q, 8), 1), eq(abs(mod(q, 8) - mod(k, 8)), 1))
table wfront(q, k) = and(gt(div(k, 8), div(q, 8)), le(abs(mod(k, 8) - mod(q, 8)), 1))
table bfront(q, k) = and(lt(div(k, 8), div(q, 8)), le(abs(mod(k, 8) - mod(q, 8)), 1))
table samefile(q, k) = and(eq(mod(k, 8), mod(q, 8)), ne(k, q))

# --- piece tables
table occ(p) = ne(p, 0)
table is(p, c) = eq(p, c)
table pawn(p) = or(eq(p, 1), eq(p, 7))
";
    let wk = |p: u8| white(p) as i64;
    let bk = |p: u8| black(p) as i64;
    let slides = |p: u8, diag: bool| -> bool {
        matches!(kind(p), Some(4)) || matches!(kind(p), Some(3)) && !diag || matches!(kind(p), Some(2)) && diag
    };
    s += &format!("table white(p) = {}\n", piece_table(wk));
    s += &format!("table black(p) = {}\n", piece_table(bk));
    for (name, col, diag) in [("w_orth", true, false), ("w_diag", true, true), ("b_orth", false, false), ("b_diag", false, true)] {
        s += &format!("table {name}(p) = {}\n", piece_table(|p| (slides(p, diag) && white(p) == col && p != 0) as i64));
        // 30 - attacker value when it slides this way, else 0 (lowest attacker = 30 - max)
        s += &format!(
            "table {name}_u(p) = {}\n",
            piece_table(|p| if slides(p, diag) && white(p) == col && p != 0 { 30 - UNIT_VAL[kind(p).unwrap()] } else { 0 })
        );
    }
    s += &format!("table unit_w(p) = {}\n", piece_table(|p| if white(p) { UNIT_VAL[kind(p).unwrap()] } else { 0 }));
    s += &format!("table unit_b(p) = {}\n", piece_table(|p| if black(p) { UNIT_VAL[kind(p).unwrap()] } else { 0 }));
    s += &format!("table hang_w(p) = {}\n", piece_table(|p| if white(p) { HANG_VAL[kind(p).unwrap()] } else { 0 }));
    s += &format!("table hang_b(p) = {}\n", piece_table(|p| if black(p) { HANG_VAL[kind(p).unwrap()] } else { 0 }));
    s += &format!("table minor(p) = {}\n", piece_table(|p| matches!(kind(p), Some(1 | 2)) as i64));
    s += &format!("table rook(p) = {}\n", piece_table(|p| matches!(kind(p), Some(3)) as i64));
    s += &format!("table queen(p) = {}\n", piece_table(|p| matches!(kind(p), Some(4)) as i64));
    for (name, eg) in [("pst_mg", false), ("pst_eg", true)] {
        let rows: Vec<String> = (0..13u8).map(|p| list((0..64).map(|sq| pst_signed(p, sq, eg)))).collect();
        s += &format!("table {name}(p, s) = [{}]\n", rows.join(", "));
    }
    s += &format!("table passed_w(s) = {}\n", list((0..64).map(|sq| PASSED[sq / 8])));
    s += &format!("table passed_b(s) = {}\n", list((0..64).map(|sq| PASSED[7 - sq / 8])));
    s += "
# --- attack maps: the first piece in each direction, and who attacks from where
";
    let mut aw = vec![];
    let mut ab = vec![];
    let mut uw = vec![];
    let mut ub = vec![];
    for (d, _) in DIRS.iter().enumerate() {
        let kind = if d < 4 { "orth" } else { "diag" };
        let cond = format!("and(onray({d}, q(pos), k(pos)), occ(k(piece))), near({d}, q(pos), k(pos))");
        s += &format!("let rw{d} = best(w_{kind}(piece), {cond})\n");
        s += &format!("let rb{d} = best(b_{kind}(piece), {cond})\n");
        s += &format!("let uw{d} = best(w_{kind}_u(piece), {cond})\n");
        s += &format!("let ub{d} = best(b_{kind}_u(piece), {cond})\n");
        aw.push(format!("rw{d}"));
        ab.push(format!("rb{d}"));
        uw.push(format!("uw{d}"));
        ub.push(format!("ub{d}"));
    }
    s += "let wn = count(and(knight(q(pos), k(pos)), is(k(piece), 2)))
let bn = count(and(knight(q(pos), k(pos)), is(k(piece), 8)))
let wp = count(and(wpawn_hits(q(pos), k(pos)), is(k(piece), 1)))
let bp = count(and(bpawn_hits(q(pos), k(pos)), is(k(piece), 7)))
let wk = count(and(kadj(q(pos), k(pos)), is(k(piece), 6)))
let bk = count(and(kadj(q(pos), k(pos)), is(k(piece), 12)))
";
    s += &format!("output att_w = {} + wn + wp + wk\n", aw.join(" + "));
    s += &format!("output att_b = {} + bn + bp + bk\n", ab.join(" + "));
    for (side, mut u) in [("w", uw), ("b", ub)] {
        u.extend([format!("27 * clip({side}n, 0, 1)"), format!("29 * clip({side}p, 0, 1)"), format!("10 * clip({side}k, 0, 1)")]);
        s += &format!("output low_{side} = 30 - {}\n", max_tree(&u));
    }
    s += "
# --- per-square terms (white = side to move)
let hit_w = gt(att_w, 0)
let hit_b = gt(att_b, 0)
# hanging: attacked, and undefended or attacked by something cheaper
output hanging_w = and(hit_b, or(not(hit_w), lt(low_b, unit_w(piece))), gt(unit_w(piece), 0))
output hanging_b = and(hit_w, or(not(hit_b), lt(low_w, unit_b(piece))), gt(unit_b(piece), 0))
let hang_us_c = select(hanging_w, hang_w(piece), 0)
let hang_them_c = select(hanging_b, hang_b(piece), 0)
let mob_c = select(1 - white(piece), att_w, 0) - select(1 - black(piece), att_b, 0)
# king zone: squares next to a king, attacked by the other side
let king_c = select(clip(bk, 0, 1), att_w, 0) - select(clip(wk, 0, 1), att_b, 0)
let blockers_w = count(and(wfront(q(pos), k(pos)), is(k(piece), 7)))
let blockers_b = count(and(bfront(q(pos), k(pos)), is(k(piece), 1)))
let passed_c = select(and(is(piece, 1), le(blockers_w, 0)), passed_w(pos), 0) - select(and(is(piece, 7), le(blockers_b, 0)), passed_b(pos), 0)
let file_pawns = count(and(samefile(q(pos), k(pos)), pawn(k(piece))))
let rooks_c = and(is(piece, 4), le(file_pawns, 0)) - and(is(piece, 10), le(file_pawns, 0))

# --- totals over the board (one shared head) and the phase blend
let mg = 64 * mean(pst_mg(piece, pos), true)
let eg = 64 * mean(pst_eg(piece, pos), true)
output mobility = 64 * mean(mob_c, true)
output king = 64 * mean(king_c, true)
output hang_us = 64 * mean(hang_us_c, true)
output hang_them = 64 * mean(hang_them_c, true)
output passed = 64 * mean(passed_c, true)
output rooks = 64 * mean(rooks_c, true)
let dmg = mg - eg
let phase_x_d = select(minor(piece), dmg, 0) + 2 * select(rook(piece), dmg, 0) + 4 * select(queen(piece), dmg, 0)
output pst = eg + 64 * mean(phase_x_d, true) / 24
";
    let terms: Vec<String> = STYLE.iter().map(|(n, w)| format!("{w} * {n}")).collect();
    s += &format!("output eval = {}\n", terms.join(" + "));
    s
}

/// Output order of `program()`.
pub const OUTPUTS: [&str; 13] =
    ["att_w", "att_b", "low_w", "low_b", "hanging_w", "hanging_b", "mobility", "king", "hang_us", "hang_them", "passed", "rooks", "pst"];

/// Plain-code reference of everything the network computes. Per square: attackers,
/// lowest attacker (30 = none), hanging bits; totals; eval (output order of `program`).
#[derive(Debug, Clone, PartialEq)]
pub struct Reference {
    pub att_w: [i64; 64],
    pub att_b: [i64; 64],
    pub low_w: [i64; 64],
    pub low_b: [i64; 64],
    pub hanging_w: [i64; 64],
    pub hanging_b: [i64; 64],
    /// mobility, king, hang_us, hang_them, passed, rooks, pst
    pub totals: [f64; 7],
    pub eval: f64,
}

pub fn reference(b: &[u8; 64]) -> Reference {
    let (mut att_w, mut att_b) = ([0i64; 64], [0i64; 64]);
    let (mut low_w, mut low_b) = ([30i64; 64], [30i64; 64]);
    let at = |f: i64, r: i64| -> Option<usize> { ((0..8).contains(&f) && (0..8).contains(&r)).then(|| (f + 8 * r) as usize) };
    for q in 0..64 {
        let (f, r) = ((q % 8) as i64, (q / 8) as i64);
        let mut hit = |p: u8| {
            let v = UNIT_VAL[kind(p).unwrap()];
            if white(p) {
                att_w[q] += 1;
                low_w[q] = low_w[q].min(v);
            } else {
                att_b[q] += 1;
                low_b[q] = low_b[q].min(v);
            }
        };
        for (d, &(df, dr)) in DIRS.iter().enumerate() {
            let mut t = 1;
            while let Some(s) = at(f + df * t, r + dr * t) {
                if b[s] != 0 {
                    let k = kind(b[s]).unwrap();
                    if k == 4 || k == 3 && d < 4 || k == 2 && d >= 4 {
                        hit(b[s]);
                    }
                    break;
                }
                t += 1;
            }
        }
        for (df, dr) in [(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)] {
            if let Some(s) = at(f + df, r + dr)
                && kind(b[s]) == Some(1)
            {
                hit(b[s]);
            }
        }
        for df in [-1, 1] {
            if let Some(s) = at(f + df, r - 1)
                && b[s] == 1
            {
                hit(b[s]);
            }
            if let Some(s) = at(f + df, r + 1)
                && b[s] == 7
            {
                hit(b[s]);
            }
        }
        for df in -1..=1 {
            for dr in -1..=1 {
                if (df, dr) != (0, 0)
                    && let Some(s) = at(f + df, r + dr)
                    && kind(b[s]) == Some(5)
                {
                    hit(b[s]);
                }
            }
        }
    }
    let unit = |p: u8, w: bool| -> i64 { if p != 0 && white(p) == w { UNIT_VAL[kind(p).unwrap()] } else { 0 } };
    let mut hanging_w = [0i64; 64];
    let mut hanging_b = [0i64; 64];
    let (mut mob, mut king, mut hus, mut hthem, mut passed, mut rooks) = (0i64, 0i64, 0i64, 0i64, 0i64, 0i64);
    let (mut mg, mut eg, mut phase) = (0i64, 0i64, 0i64);
    let near_king = |q: usize, king: u8| -> bool {
        let (f, r) = ((q % 8) as i64, (q / 8) as i64);
        (0..64).any(|s| b[s] == king && s != q && ((s % 8) as i64 - f).abs().max(((s / 8) as i64 - r).abs()) == 1)
    };
    for q in 0..64 {
        let p = b[q];
        let (uw, ub) = (unit(p, true), unit(p, false));
        hanging_w[q] = (att_b[q] > 0 && (att_w[q] == 0 || low_b[q] < uw) && uw > 0) as i64;
        hanging_b[q] = (att_w[q] > 0 && (att_b[q] == 0 || low_w[q] < ub) && ub > 0) as i64;
        if hanging_w[q] == 1 {
            hus += HANG_VAL[kind(p).unwrap()];
        }
        if hanging_b[q] == 1 {
            hthem += HANG_VAL[kind(p).unwrap()];
        }
        if !white(p) {
            mob += att_w[q];
        }
        if !black(p) {
            mob -= att_b[q];
        }
        if near_king(q, 12) {
            king += att_w[q];
        }
        if near_king(q, 6) {
            king -= att_b[q];
        }
        let (f, r) = ((q % 8) as i64, (q / 8) as i64);
        let zone = |s: usize, ahead: bool| {
            let (sf, sr) = ((s % 8) as i64, (s / 8) as i64);
            (sf - f).abs() <= 1 && if ahead { sr > r } else { sr < r }
        };
        if p == 1 && !(0..64).any(|s| b[s] == 7 && zone(s, true)) {
            passed += PASSED[r as usize];
        }
        if p == 7 && !(0..64).any(|s| b[s] == 1 && zone(s, false)) {
            passed -= PASSED[7 - r as usize];
        }
        let open = !(0..64).any(|s| s != q && (s % 8) as i64 == f && (b[s] == 1 || b[s] == 7));
        if open && p == 4 {
            rooks += 1;
        }
        if open && p == 10 {
            rooks -= 1;
        }
        mg += pst_signed(p, q, false);
        eg += pst_signed(p, q, true);
        if let Some(k) = kind(p) {
            phase += PHASE_W[k];
        }
    }
    let pst = eg as f64 + phase as f64 * (mg - eg) as f64 / 24.0;
    let totals = [mob as f64, king as f64, hus as f64, hthem as f64, passed as f64, rooks as f64, pst];
    let eval = pst + STYLE[1].1 * totals[0] + STYLE[2].1 * totals[1] + STYLE[3].1 * totals[2] + STYLE[4].1 * totals[3]
        + STYLE[5].1 * totals[4] + STYLE[6].1 * totals[5];
    Reference { att_w, att_b, low_w, low_b, hanging_w, hanging_b, totals, eval }
}

/// FEN piece placement -> board (a1 = 0).
pub fn board_from_fen(fen: &str) -> Option<[u8; 64]> {
    let mut b = [0u8; 64];
    let place = fen.split_whitespace().next()?;
    let ranks: Vec<&str> = place.split('/').collect();
    if ranks.len() != 8 {
        return None;
    }
    for (i, row) in ranks.iter().enumerate() {
        let r = 7 - i;
        let mut f = 0;
        for ch in row.chars() {
            if let Some(n) = ch.to_digit(10) {
                f += n as usize;
            } else {
                let p = PIECES.find(ch)? as u8;
                if p == 0 || f > 7 {
                    return None;
                }
                b[f + 8 * r] = p;
                f += 1;
            }
        }
        if f != 8 {
            return None;
        }
    }
    Some(b)
}

/// Mirror for the other side to move: square ^ 56, colours swapped.
pub fn mirror(b: &[u8; 64]) -> [u8; 64] {
    let mut m = [0u8; 64];
    for s in 0..64 {
        let p = b[s ^ 56];
        m[s] = if p == 0 { 0 } else if p <= 6 { p + 6 } else { p - 6 };
    }
    m
}

/// A random board: each side gets one king and a random army (piece counts up to the
/// usual ones), pawns kept off the first and last rank. Not necessarily legal.
pub fn random_board(rng: &mut u64) -> [u8; 64] {
    let mut next = || {
        *rng ^= *rng << 13;
        *rng ^= *rng >> 7;
        *rng ^= *rng << 17;
        *rng
    };
    let mut b = [0u8; 64];
    let mut free: Vec<usize> = (0..64).collect();
    let place = |b: &mut [u8; 64], p: u8, free: &mut Vec<usize>, next: &mut dyn FnMut() -> u64| {
        let ok: Vec<usize> = free.iter().copied().filter(|&s| kind(p) != Some(0) || (8..56).contains(&s)).collect();
        if ok.is_empty() {
            return;
        }
        let s = ok[(next() % ok.len() as u64) as usize];
        b[s] = p;
        free.retain(|&x| x != s);
    };
    for side in [0u8, 6] {
        place(&mut b, 6 + side, &mut free, &mut next);
        for (k, max) in [(1u8, 8u64), (2, 2), (3, 2), (4, 2), (5, 1)] {
            let n = next() % (max + 1);
            for _ in 0..n {
                place(&mut b, k + side, &mut free, &mut next);
            }
        }
    }
    b
}

pub fn tokens(b: &[u8; 64]) -> Vec<Vec<i64>> {
    b.iter().map(|&p| vec![p as i64]).collect()
}

/// Compile the program; returns the program, weights and the proven output bounds.
pub fn build() -> crate::error::Result<(SeqProgram, SeqWeights, Vec<f64>)> {
    let (p, w, _) = seq::compile_str(&program())?;
    let bound = seq::error_bound(&p).outputs;
    Ok((p, w, bound))
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ChessCheck {
    pub boards: u64,
    /// Boards where a rounded per-square output differs from the reference.
    pub square_wrong: u64,
    /// Largest |network - reference| over the totals and eval.
    pub worst_total: f64,
    /// Largest |network - reference| over the per-square outputs (before rounding).
    pub worst_square: f64,
}

/// Network outputs (index by `OUTPUTS` then `eval`) against the reference on boards.
pub fn check(w: &SeqWeights, boards: impl Iterator<Item = [u8; 64]>) -> ChessCheck {
    let runner = SeqRunner::new(w);
    let mut c = ChessCheck::default();
    for b in boards {
        let net = runner.eval_multi(&tokens(&b));
        let r = reference(&b);
        c.boards += 1;
        let per = [&r.att_w, &r.att_b, &r.low_w, &r.low_b, &r.hanging_w, &r.hanging_b];
        let mut wrong = false;
        for (o, want) in per.iter().enumerate() {
            for s in 0..64 {
                let d = (net[s][o] - want[s] as f64).abs();
                c.worst_square = c.worst_square.max(d);
                wrong |= net[s][o].round() != want[s] as f64;
            }
        }
        c.square_wrong += wrong as u64;
        for (t, want) in r.totals.iter().enumerate() {
            c.worst_total = c.worst_total.max((net[0][6 + t] - want).abs());
        }
        c.worst_total = c.worst_total.max((net[0][13] - r.eval).abs());
    }
    c
}
