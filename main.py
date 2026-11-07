# ═══════════════════════════════════════════════════════════════
#   XOCDIA PREDICTOR - CHẴN/LẺ + VỊ (VỊ KHỚP CỔNG CHÍNH)
#   CRE @JOYBOY36VN
#   t.me/Joyboy36vn
# ═══════════════════════════════════════════════════════════════

import json, os, time, random
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
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
#   CỔNG PHỤ – DỰ ĐOÁN VỊ (CHỈ KHI KHỚP CỔNG CHÍNH)
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

def predict_vi(hist_vi, cong_chinh=None, bet_threshold=3):
    """
    hist_vi: list {'do','trang'} cũ -> mới
    cong_chinh: "CHẴN" | "LẺ" | None
    Chỉ trả vị KHỚP với cổng chính:
      - CHẴN -> chỉ tứ tử (4-0 / 0-4)
      - LẺ   -> chỉ 1-3 / 3-1
      - None -> không trả vị
    """
    if not hist_vi or not cong_chinh:
        return None

    last = hist_vi[-1]
    last5 = hist_vi[-5:]

    # ═══════ CỬA CHẴN – CHỈ TỨ TỬ ═══════
    if cong_chinh == "CHẴN":
        if last in ({"do":4,"trang":0}, {"do":0,"trang":4}):
            return {"ten": "tứ tử đỏ và tứ tử trắng", "cua": ["4-0","0-4"], "loai": "CHẴN"}
        if len(hist_vi) >= 3 and all(_la_cua_chan(p) for p in hist_vi[-3:]):
            return {"ten": "tứ tử đỏ và tứ tử trắng", "cua": ["4-0","0-4"], "loai": "CHẴN"}
        if len(hist_vi) >= bet_threshold:
            recent = hist_vi[-bet_threshold:]
            if all(p == {"do":4,"trang":0} for p in recent) or \
               all(p == {"do":0,"trang":4} for p in recent):
                return {"ten": "tứ tử đỏ và tứ tử trắng", "cua": ["4-0","0-4"], "loai": "CHẴN"}
        return None

    # ═══════ CỬA LẺ – CHỈ 1-3 / 3-1 ═══════
    if cong_chinh == "LẺ":
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

    return None

def _vi_ten(vi_thuc):
    """Chuyển {'do','trang'} thành tên cửa vị."""
    if not vi_thuc:
        return None
    do, trang = vi_thuc["do"], vi_thuc["trang"]
    if (do, trang) in [(4,0),(0,4)]:
        return "tứ tử đỏ và tứ tử trắng"
    if (do, trang) == (1,3):
        return "1 đỏ 3 trắng"
    if (do, trang) == (3,1):
        return "1 trắng 3 đỏ"
    return "2-2"

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

    if not r:
        return {
            "cre": "CRE @JOYBOY36VN",
            "du_doan": None,
            "du_doan_vi": None,
            "ly_do": "Hoà – bỏ ván",
            "phien_hien_tai": cur["id"],
        }

    vi = predict_vi(hist_vi, cong_chinh=r["du_doan"]) if hist_vi else None
    du_doan_vi = vi["ten"] if vi else None

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

@app.get("/thongke")
async def thongke():
    ss = {s["id"]: s for s in await fetch_sessions()}
    db = _load()

    # ═══ CỔNG CHÍNH ═══
    tong = dung = sai = 0
    conf = 0.0
    st_d = st_s = 0
    # ═══ CỔNG PHỤ VỊ ═══
    vi_tong = vi_dung = vi_sai = 0
    vi_st_d = vi_st_s = 0
    # ═══ LỊCH SỬ ═══
    ls = []

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

        # Cổng phụ vị
        du_doan_vi = p.get("du_doan_vi")
        vi_thuc = _parse_vi(ss[pid])
        vi_thuc_ten = _vi_ten(vi_thuc) if vi_thuc else None
        vi_ok = None
        if du_doan_vi and vi_thuc_ten:
            vi_tong += 1
            vi_ok = (du_doan_vi == vi_thuc_ten)
            if vi_ok:
                vi_dung += 1; vi_st_d += 1; vi_st_s = 0
            else:
                vi_sai += 1; vi_st_s += 1; vi_st_d = 0

        ls.append({
            "phien": pid,
            "du_doan": p["du_doan"],
            "ket_qua": kq,
            "dung": "✅" if ok else "❌",
            "conf": p.get("do_tin_cay"),
            "du_doan_vi": du_doan_vi,
            "vi_thuc": vi_thuc_ten,
            "vi_dung": ("✅" if vi_ok else "❌") if vi_ok is not None else None,
        })

    return {
        "cre": "CRE @JOYBOY36VN",
        "cong_chinh": {
            "tong_so_du_doan": tong,
            "so_dung": dung,
            "so_sai": sai,
            "ty_le_dung": round(dung / tong, 4) if tong else 0,
            "do_tin_cay_trung_binh": round(conf / tong, 4) if tong else 0,
            "chuoi_dung_hien_tai": st_d,
            "chuoi_sai_hien_tai": st_s,
        },
        "cong_phu_vi": {
            "tong_so_du_doan": vi_tong,
            "so_dung": vi_dung,
            "so_sai": vi_sai,
            "ty_le_dung": round(vi_dung / vi_tong, 4) if vi_tong else 0,
            "chuoi_dung_hien_tai": vi_st_d,
            "chuoi_sai_hien_tai": vi_st_s,
        },
        "lich_su_gan_nhat": ls[-20:][::-1],
    }

# ⬇️ Đoạn chạy cho Render
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
