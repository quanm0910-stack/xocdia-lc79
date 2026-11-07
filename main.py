# ═══════════════════════════════════════════════════════════════
#   XOCDIA PREDICTOR - CHẴN/LẺ + VỊ
#   CRE @JOYBOY36VN
#   t.me/Joyboy36vn
# ═══════════════════════════════════════════════════════════════

import json, os, time, random
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, Query
import httpx

URL = "https://wcl.tele68.com/v1/chanlefull/lite-sessions"
DB = "/tmp/predictions.json"
WEIGHTS = {"pattern": 1.15, "stat": 1.00, "counter": 0.95, "trend": 0.88}
_cache = {"data": None, "ts": 0}

app = FastAPI(title="XocDia Predictor - CRE @JOYBOY36VN")

# ═══════════════════════════════════════════════════════════════
#   CHUẨN HOÁ KẾT QUẢ TỪ VỊ
# ═══════════════════════════════════════════════════════════════
def _ket_qua_chuan(s):
    """
    Tính chẵn/lẻ CHUẨN từ vị (số đồng đỏ).
    Số đỏ chẵn (0,2,4) -> CHẴN, số đỏ lẻ (1,3) -> LẺ
    Fallback resultTruyenThong nếu không có vi.
    """
    vi = s.get("vi") or ""
    up = str(vi).upper()
    do = up.count("DO") + up.count("ĐỎ") + up.count("RED")
    trang = up.count("TRANG") + up.count("TRẮNG") + up.count("WHITE")

    if do + trang == 4:
        return "chan" if do % 2 == 0 else "le"

    # Thử parse dices nếu resultVi không đủ
    co = s.get("co") or []
    if isinstance(co, list) and len(co) == 4:
        do2 = sum(1 for d in co if str(d).upper() in ("DO","ĐỎ","RED","1"))
        return "chan" if do2 % 2 == 0 else "le"

    return s["kq"]

# ───────────── FETCH ─────────────
async def fetch_sessions():
    if time.time() - _cache["ts"] < 3 and _cache["data"]:
        return _cache["data"]
    async with httpx.AsyncClient(timeout=5) as c:
        r = await c.get(URL, headers={"Cache-Control": "no-store"})
        r.raise_for_status()
        d = r.json()
    sessions = d.get("list") or d.get("data") if isinstance(d, dict) else d
    out = []
    for s in (sessions or []):
        item = {
            "id": s["id"],
            "kq": s["resultTruyenThong"],
            "vi": s.get("resultVi", ""),
            "co": s.get("dices", []),
        }
        item["kq_chuan"] = _ket_qua_chuan(item)
        out.append(item)
    _cache["data"], _cache["ts"] = out, time.time()
    return out

# ───────────── STORAGE ─────────────
def _load():
    if not os.path.exists(DB): return {}
    try:
        with open(DB, encoding="utf-8") as f: return json.load(f)
    except: return {}

def _save(d):
    with open(DB, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)

def _L(x): return "CHẴN" if x == "chan" else "LẺ"

# ───────────── 14 HEURISTIC ─────────────
def _wilson(p, n):
    if n == 0: return 0.5
    return max(0, p - 1.96 * (p * (1 - p) / n) ** 0.5)

def _streak(h):
    if not h: return 0
    n = 0
    for x in reversed(h):
        if x == h[-1]: n += 1
        else: break
    return n

def h_trend(h):
    p = h[-5:].count("chan") / 5
    if p >= 0.6: return ("chan", _wilson(p, 5), "trend")
    if p <= 0.4: return ("le", _wilson(1 - p, 5), "trend")

def h_revert(h):
    if len(h) < 8: return
    p = h[-8:].count("chan") / 8
    if p >= 0.75: return ("le", 0.65, "counter")
    if p <= 0.25: return ("chan", 0.65, "counter")

def h_momentum(h):
    if len(h) < 10: return
    d = h[-5:].count("chan") - h[-10:-5].count("chan")
    if d >= 2: return ("chan", 0.6, "trend")
    if d <= -2: return ("le", 0.6, "trend")

def h_markov(h):
    if len(h) < 3: return
    for o in (3, 2, 1):
        if len(h) <= o: continue
        st = tuple(h[-o:])
        nxt = [h[i + o] for i in range(len(h) - o) if tuple(h[i:i + o]) == st]
        if len(nxt) >= 2:
            c, l = nxt.count("chan"), nxt.count("le")
            if c > l: return ("chan", _wilson(c / len(nxt), len(nxt)), "stat")
            if l > c: return ("le", _wilson(l / len(nxt), len(nxt)), "stat")

def h_hotzone(h):
    if len(h) < 8: return
    p1, p0 = h[-4:].count("chan") / 4, h[-8:-4].count("chan") / 4
    if p1 - p0 > 0.25 and p1 >= 0.6: return ("chan", 0.65, "stat")
    if p1 - p0 < -0.25 and p1 <= 0.4: return ("le", 0.65, "stat")

def h_bayes(h):
    if len(h) < 5: return
    ws = ps = 0.0
    for i, x in enumerate(reversed(h)):
        w = 1.08 ** i; ws += w
        if x == "chan": ps += w
    p = ps / ws
    if p > 0.54: return ("chan", p, "stat")
    if p < 0.46: return ("le", 1 - p, "stat")

def h_majority(h):
    if len(h) < 7: return
    c = h[-7:].count("chan")
    if c >= 5: return ("chan", 0.58 + abs(c - 5) * 0.04, "trend")
    if c <= 2: return ("le", 0.58 + abs(c - 5) * 0.04, "trend")

def h_follow(h):
    s = _streak(h)
    if s in (2, 3): return (h[-1], 0.55, "pattern")

def h_break(h):
    s = _streak(h)
    if 4 <= s <= 6:
        return ("le" if h[-1] == "chan" else "chan", 0.6 + s * 0.02, "counter")

def h_dragon(h):
    if _streak(h) >= 7: return (h[-1], 0.75, "pattern")

def h_pingpong(h):
    if len(h) < 3: return
    a, b, c = h[-3], h[-2], h[-1]
    if a != b and b != c:
        return ("le" if h[-1] == "chan" else "chan", 0.65, "pattern")

def _match_pattern(h):
    if len(h) >= 6:
        x = h[-6:]
        if x[0] == x[1] == x[2] and x[3] == x[4] and x[5] != x[2] and x[5] != x[3]:
            return (x[5], 0.72)
        if x[0] == x[1] and x[2] == x[3] and x[4] == x[5] and x[0] != x[2] and x[2] != x[4]:
            return (x[4], 0.72)
        if x[0] == x[1] == x[2] and x[3] == x[4] == x[5] and x[0] != x[3]:
            return (x[3], 0.72)

    if len(h) >= 5:
        x = h[-5:]
        if x[0] == x[1] == x[2] and x[3] == x[4] and x[0] != x[3]:
            return (x[3], 0.68)
        if x[0] == x[4] and x[1] == x[2] == x[3] and x[0] != x[1]:
            return (x[0], 0.68)
        if x[0] == x[1] and x[2] == x[3] == x[4] and x[0] != x[2]:
            return (x[2], 0.65)
        if x[0] == x[1] == x[2] and x[3] == x[4] and x[0] != x[3]:
            return (x[3], 0.65)
        if x[0] != x[1] and x[1] == x[2] and x[3] == x[4] and x[1] != x[3]:
            return (x[3], 0.65)
        if x[0] == x[1] and x[2] != x[0] and x[3] == x[4]:
            return (x[3], 0.63)

    if len(h) >= 4:
        x = h[-4:]
        if x[0] == x[1] == x[2] and x[0] != x[3]:
            return (x[3], 0.65)
        if x[0] != x[1] and x[1] == x[2] == x[3]:
            return (x[1], 0.65)
        if x[0] == x[3] and x[1] == x[2] and x[0] != x[1]:
            return (x[0], 0.63)
        if x[0] == x[1] and x[2] == x[3] and x[0] != x[2]:
            return (x[2], 0.63)
        if x[0] != x[1] and x[1] == x[2] and x[2] != x[3]:
            return (x[3], 0.60)

    if len(h) >= 3:
        x = h[-3:]
        if x[0] == x[1] and x[1] != x[2]:
            return (x[2], 0.60)
        if x[0] != x[1] and x[1] == x[2]:
            return (x[1], 0.60)
        if x[0] == x[2] and x[0] != x[1]:
            return (x[0], 0.58)

    return None

def h_pattern_master(h):
    r = _match_pattern(h)
    if r is None: return
    side, conf = r
    return (side, conf, "pattern")

def h_runlen(h):
    if len(h) < 3: return
    runs, cur = [], 1
    for i in range(1, len(h)):
        if h[i] == h[i - 1]: cur += 1
        else: runs.append(cur); cur = 1
    runs.append(cur)
    if len(runs) < 2: return
    avg = sum(runs[:-1]) / (len(runs) - 1)
    if runs[-1] >= 2 * avg:
        return ("le" if h[-1] == "chan" else "chan", 0.65, "counter")

def h_cycle(h):
    if len(h) < 8: return
    for L in (1, 2, 3, 4):
        if h[-L:] == h[-2 * L:-L]:
            return ("le" if h[-1] == "chan" else "chan", 0.6, "pattern")

HEUR = [h_trend, h_revert, h_momentum, h_markov, h_hotzone, h_bayes,
        h_majority, h_follow, h_break, h_dragon, h_pingpong,
        h_pattern_master, h_runlen, h_cycle]

def predict(hist):
    if len(hist) < 5: return None
    sc = {"chan": 0.0, "le": 0.0}; vo = {"chan": 0, "le": 0}; det = []
    for fn in HEUR:
        r = fn(hist)
        if not r:
            det.append({"ten": fn.__name__, "skip": True}); continue
        side, conf, typ = r; w = WEIGHTS[typ]
        sc[side] += conf * w; vo[side] += 1
        det.append({"ten": fn.__name__, "ben": "CHẴN" if side == "chan" else "LẺ",
                    "conf": round(conf, 3), "type": typ})
    if sc["chan"] == sc["le"]: return None
    win = "chan" if sc["chan"] > sc["le"] else "le"
    return {"du_doan": "CHẴN" if win == "chan" else "LẺ",
            "do_tin_cay": round(sc[win] / (sc["chan"] + sc["le"]), 4),
            "tong_hop": {"score_chan": round(sc["chan"], 3), "score_le": round(sc["le"], 3),
                         "vote_chan": vo["chan"], "vote_le": vo["le"]},
            "chi_tiet_heuristic": det}

# ═══════════════════════════════════════════════════════════════
#   CỔNG PHỤ – DỰ ĐOÁN VỊ
# ═══════════════════════════════════════════════════════════════
def _parse_vi(s):
    """Chuyển 'vi' / 'co' thành {'do':int,'trang':int} tổng = 4."""
    vi = s.get("vi")
    if isinstance(vi, list):
        do = sum(1 for v in vi if str(v).upper() in ("DO", "ĐỎ", "RED"))
        trang = sum(1 for v in vi if str(v).upper() in ("TRANG", "TRẮNG", "WHITE"))
        if do + trang == 4:
            return {"do": do, "trang": trang}
    if isinstance(vi, str) and vi:
        up = vi.upper()
        if "-" in up:
            parts = up.replace(" ", "").split("-")
            try:
                a, b = int(parts[0]), int(parts[1])
                if a + b == 4:
                    return {"do": a, "trang": b}
            except: pass
        do = up.count("DO") + up.count("ĐỎ") + up.count("RED")
        trang = up.count("TRANG") + up.count("TRẮNG") + up.count("WHITE")
        if do + trang == 4:
            return {"do": do, "trang": trang}
    co = s.get("co")
    if isinstance(co, list) and len(co) == 4:
        do = sum(1 for d in co if str(d).upper() in ("DO","ĐỎ","RED","1"))
        trang = sum(1 for d in co if str(d).upper() in ("TRANG","TRẮNG","WHITE","0"))
        if do + trang == 4:
            return {"do": do, "trang": trang}
    return None

def _la_cua_chan(p):
    return (p["do"], p["trang"]) in [(2,2),(4,0),(0,4)]

def predict_vi(hist_vi, bet_threshold=3):
    """hist_vi: list {'do','trang'} cũ -> mới. Trả về dict hoặc None."""
    if not hist_vi:
        return None
    last = hist_vi[-1]
    last5 = hist_vi[-5:]

    # CỬA CHẴN – tứ tử (0-4 hoặc 4-0)
    if last in ({"do":4,"trang":0}, {"do":0,"trang":4}):
        return {"ten": "tứ tử đỏ và tứ tử trắng", "cua": ["4-0","0-4"], "loai": "CHẴN"}
    if len(hist_vi) >= 3 and all(_la_cua_chan(p) for p in hist_vi[-3:]):
        return {"ten": "tứ tử đỏ và tứ tử trắng", "cua": ["4-0","0-4"], "loai": "CHẴN"}
    if len(hist_vi) >= bet_threshold:
        recent = hist_vi[-bet_threshold:]
        if all(p == {"do":4,"trang":0} for p in recent) or \
           all(p == {"do":0,"trang":4} for p in recent):
            return {"ten": "tứ tử đỏ và tứ tử trắng", "cua": ["4-0","0-4"], "loai": "CHẴN"}

    # CỬA LẺ – đếm 1-3 / 3-1 trong 5 tay
    if len(last5) >= 5:
        cnt_A = sum(1 for p in last5 if p == {"do":1,"trang":3})
        cnt_B = sum(1 for p in last5 if p == {"do":3,"trang":1})
        if cnt_A > cnt_B:
            return {"ten": "1 đỏ 3 trắng", "cua": ["1-3"], "loai": "LẺ"}
        if cnt_B > cnt_A:
            return {"ten": "1 trắng 3 đỏ", "cua": ["3-1"], "loai": "LẺ"}
        return random.choice([
            {"ten": "1 đỏ 3 trắng", "cua": ["1-3"], "loai": "LẺ"},
            {"ten": "1 trắng 3 đỏ", "cua": ["3-1"], "loai": "LẺ"},
        ])
    return None

# ───────────── ROUTES ─────────────
@app.get("/dudoan")
async def dudoan():
    ss = sorted(await fetch_sessions(), key=lambda s: s["id"], reverse=True)
    if len(ss) < 5:
        raise HTTPException(400, {"cre": "CRE @JOYBOY36VN", "loi": "Cần >= 5 ván"})

    cur = ss[0]
    hist = list(reversed([s["kq_chuan"] for s in ss]))
    hist_vi = [v for v in (_parse_vi(s) for s in reversed(ss)) if v]

    r = predict(hist)
    vi = predict_vi(hist_vi) if hist_vi else None
    du_doan_vi = vi["ten"] if vi else None

    if not r:
        return {
            "cre": "CRE @JOYBOY36VN",
            "du_doan": None,
            "du_doan_vi": du_doan_vi,
            "ly_do": "Hoà – bỏ ván",
            "phien_hien_tai": cur["id"],
        }

    pid, t = cur["id"] + 1, datetime.now(timezone.utc).isoformat()
    db = _load()
    db[str(pid)] = {
        "du_doan": r["du_doan"],
        "du_doan_vi": du_doan_vi,
        "do_tin_cay": r["do_tin_cay"],
        "thoi_gian": t,
    }
    _save(db)

    return {
        "cre": "CRE @JOYBOY36VN",
        "phien_du_doan": pid,
        "du_doan": r["du_doan"],
        "du_doan_vi": du_doan_vi,
        "do_tin_cay": r["do_tin_cay"],
        "phien_hien_tai": cur["id"],
        "ket_qua": _L(cur["kq_chuan"]),
        "vi": cur["vi"],
        "lich_su_gan_nhat": [
            {"id": s["id"], "ket_qua": _L(s["kq_chuan"])}
            for s in ss[:5]
        ],
        "chi_tiet_heuristic": r["chi_tiet_heuristic"],
        "tong_hop": r["tong_hop"],
        "thoi_gian": t,
    }

@app.get("/lichsu")
async def lichsu(limit: int = Query(20, ge=1, le=100), page: int = Query(1, ge=1)):
    ss = sorted(await fetch_sessions(), key=lambda s: s["id"], reverse=True)
    db = _load()
    tong = len(ss)
    so_trang = max(1, (tong + limit - 1) // limit)
    items = ss[(page - 1) * limit: page * limit]

    out = []
    for s in items:
        p = db.get(str(s["id"]))
        kq = _L(s["kq_chuan"])
        if p:
            dung_icon = "✅" if p["du_doan"] == kq else "❌"
        else:
            dung_icon = None
        out.append({
            "id": s["id"],
            "ket_qua": kq,
            "vi": s["vi"],
            "du_doan": p["du_doan"] if p else None,
            "do_tin_cay": p["do_tin_cay"] if p else None,
            "du_doan_vi": p.get("du_doan_vi") if p else None,
            "dung": dung_icon,
        })

    return {
        "cre": "CRE @JOYBOY36VN",
        "tong_phien": tong,
        "trang": page,
        "so_trang": so_trang,
        "phien_moi_nhat": ss[0]["id"] if ss else None,
        "phien_cu_nhat": ss[-1]["id"] if ss else None,
        "lich_su": out,
    }

@app.get("/thongke")
async def thongke():
    ss = {s["id"]: s for s in await fetch_sessions()}
    db = _load()
    tong = dung = sai = 0
    conf = 0.0
    ls = []
    st_d = st_s = 0

    for pid_str in sorted(db, key=int):
        pid = int(pid_str)
        p = db[pid_str]
        if pid not in ss: continue

        kq = _L(ss[pid]["kq_chuan"])
        ok = p["du_doan"] == kq
        tong += 1
        conf += p.get("do_tin_cay", 0)

        if ok:
            dung += 1; st_d += 1; st_s = 0
        else:
            sai += 1; st_s += 1; st_d = 0

        ls.append({
            "phien": pid,
            "du_doan": p["du_doan"],
            "ket_qua": kq,
            "dung": "✅" if ok else "❌",
            "conf": p.get("do_tin_cay"),
            "du_doan_vi": p.get("du_doan_vi"),
        })

    return {
        "cre": "CRE @JOYBOY36VN",
        "tong_so_du_doan": tong,
        "so_dung": dung,
        "so_sai": sai,
        "ty_le_dung": round(dung / tong, 4) if tong else 0,
        "do_tin_cay_trung_binh": round(conf / tong, 4) if tong else 0,
        "chuoi_dung_hien_tai": st_d,
        "chuoi_sai_hien_tai": st_s,
        "lich_su_gan_nhat": ls[-20:][::-1],
    }

# ⬇️ Đoạn chạy cho Render
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
