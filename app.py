import os
import math
import time
import threading
import requests
from io import BytesIO
from flask import Flask, request, jsonify, render_template, send_file, redirect, url_for, Response

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Shared state (single-user; no DB needed for MVP)
# ---------------------------------------------------------------------------
state = {
    "lat": None,
    "lon": None,
    "dest_lat": None,
    "dest_lon": None,
    "dest_label": "",
    "steps": [],
    "route_coords": [],   # [[lat,lon], ...] for map polyline
    "current_step": 0,
    "updated_at": 0,
    "osrm_error": None,
}
state_lock = threading.Lock()

OSRM_BASE      = "http://router.project-osrm.org/route/v1/driving"
NOMINATIM_BASE = "https://nominatim.openstreetmap.org/search"
NOMINATIM_UA   = "HondaGPSNav/1.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _osrm_route(lat1, lon1, lat2, lon2):
    url = f"{OSRM_BASE}/{lon1},{lat1};{lon2},{lat2}"
    params = {"overview": "full", "steps": "true", "geometries": "geojson"}
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=12)
            r.raise_for_status()
            data = r.json()
            if data.get("code") != "Ok":
                return None, [], f"OSRM: {data.get('code')}"
            route    = data["routes"][0]
            steps    = route["legs"][0]["steps"]
            # GeoJSON coords are [lon, lat] → flip to [lat, lon] for Leaflet
            raw_coords = route["geometry"]["coordinates"]
            coords = [[c[1], c[0]] for c in raw_coords]
            return steps, coords, None
        except requests.Timeout:
            if attempt == 2:
                return None, [], "OSRM timeout (3 intentos)"
        except Exception as e:
            return None, [], str(e)
    return None, [], "Error desconocido"


def _parse_step(step):
    maneuver = step.get("maneuver", {})
    mtype    = maneuver.get("type", "")
    mod      = maneuver.get("modifier", "")
    name     = step.get("name", "") or ""
    dist     = step.get("distance", 0)

    type_map = {
        "turn":           f"Gira {_mod_es(mod)}",
        "new name":       f"Continúa por",
        "depart":         "Sal hacia",
        "arrive":         "Llegaste",
        "merge":          f"Incorpora {_mod_es(mod)}",
        "on ramp":        f"Toma el ramal {_mod_es(mod)}",
        "off ramp":       f"Sal por el ramal {_mod_es(mod)}",
        "fork":           f"En el cruce, ve {_mod_es(mod)}",
        "end of road":    f"Al final, gira {_mod_es(mod)}",
        "roundabout":     "Entra a la rotonda",
        "exit roundabout":"Sal de la rotonda",
        "rotary":         "Entra a la rotonda",
        "exit rotary":    "Sal de la rotonda",
        "continue":       f"Continúa {_mod_es(mod)}",
    }
    action = type_map.get(mtype, mtype.capitalize())
    label  = f"{action} {name}".strip() if name else action

    loc = maneuver.get("location", [None, None])
    return {
        "instruction": label,
        "distance_m":  round(dist),
        "lat": loc[1],
        "lon": loc[0],
        "icon": _maneuver_icon(mtype, mod),
    }


def _maneuver_icon(mtype, mod):
    if mtype in ("arrive",):
        return "★"
    if mtype in ("depart",):
        return "↑"
    if mtype in ("roundabout", "rotary", "exit roundabout", "exit rotary"):
        return "↻"
    arrows = {
        "left":         "←",
        "right":        "→",
        "slight left":  "↖",
        "slight right": "↗",
        "sharp left":   "↙",
        "sharp right":  "↘",
        "straight":     "↑",
        "uturn":        "↩",
    }
    return arrows.get(mod, "↑")


def _mod_es(mod):
    return {
        "left":         "a la izquierda",
        "right":        "a la derecha",
        "slight left":  "ligeramente a la izquierda",
        "slight right": "ligeramente a la derecha",
        "sharp left":   "fuertemente a la izquierda",
        "sharp right":  "fuertemente a la derecha",
        "straight":     "recto",
        "uturn":        "en U",
    }.get(mod, mod)


def _fmt_dist(meters):
    if meters >= 1000:
        return f"{meters/1000:.1f} km"
    return f"{int(meters)} m"


def _advance_step():
    """Move current_step forward if close enough to next maneuver point."""
    s = state
    if not s["steps"] or s["lat"] is None:
        return
    steps = s["steps"]
    idx   = s["current_step"]
    if idx >= len(steps) - 1:
        return
    nxt = steps[idx]
    if nxt["lat"] is None:
        return
    d = _haversine(s["lat"], s["lon"], nxt["lat"], nxt["lon"])
    if d < 40:
        s["current_step"] = idx + 1


def _static_map_url(lat, lon, zoom=15, width=400, height=300, marker=True):
    """Return a tile-based static map image URL via geoapify (free tier)."""
    # Use Geoapify static maps API (free, no JS, returns PNG)
    key = os.environ.get("GEOAPIFY_KEY", "")
    if key:
        marker_param = f"&marker=lonlat:{lon},{lat};color:%23ff0000;size:medium" if marker else ""
        return (
            f"https://maps.geoapify.com/v1/staticmap"
            f"?style=osm-carto&width={width}&height={height}"
            f"&center=lonlat:{lon},{lat}&zoom={zoom}"
            f"{marker_param}&apiKey={key}"
        )
    # Fallback: openstreetmap.org bbox image (no API key needed, unreliable for hotlinking)
    # Use staticmap.net as a simple fallback
    delta = 0.005
    bbox = f"{lon-delta},{lat-delta},{lon+delta},{lat+delta}"
    return (
        f"https://staticmap.openstreetmap.de/staticmap.php"
        f"?center={lat},{lon}&zoom={zoom}&size={width}x{height}"
        + (f"&markers={lat},{lon},red-pushpin" if marker else "")
    )

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return redirect(url_for("mobile"))


@app.route("/mobile")
def mobile():
    """Page loaded on the Redmi — reads GPS and sets destination."""
    with state_lock:
        dest_label = state["dest_label"]
        dest_lat   = state["dest_lat"]
        dest_lon   = state["dest_lon"]
        has_pos    = state["lat"] is not None
    return render_template(
        "mobile.html",
        dest_label=dest_label,
        dest_lat=dest_lat,
        dest_lon=dest_lon,
        has_pos=has_pos,
    )


@app.route("/update_location", methods=["POST"])
def update_location():
    """Called by mobile JS every few seconds with current GPS coords."""
    data = request.get_json(force=True)
    lat  = float(data["lat"])
    lon  = float(data["lon"])
    with state_lock:
        first_fix = state["lat"] is None
        state["lat"] = lat
        state["lon"] = lon
        state["updated_at"] = time.time()
        route_ready = False
        # Auto-calculate route on first GPS fix if destination already set
        if first_fix and state["dest_lat"] is not None and not state["steps"]:
            raw, coords, err = _osrm_route(lat, lon, state["dest_lat"], state["dest_lon"])
            state["steps"]        = [_parse_step(s) for s in raw] if raw else []
            state["route_coords"] = coords
            state["osrm_error"]   = err
            state["current_step"] = 0
            route_ready = bool(state["steps"])
        else:
            _advance_step()
    return jsonify({"ok": True, "route_ready": route_ready})


@app.route("/set_destination", methods=["POST"])
def set_destination():
    """Set destination and (re)calculate route from current position."""
    data      = request.get_json(force=True)
    dest_lat  = float(data["lat"])
    dest_lon  = float(data["lon"])
    dest_label = data.get("label", f"{dest_lat:.5f},{dest_lon:.5f}")

    with state_lock:
        state["dest_lat"]   = dest_lat
        state["dest_lon"]   = dest_lon
        state["dest_label"] = dest_label
        state["current_step"] = 0

        if state["lat"] is not None:
            raw, coords, err = _osrm_route(state["lat"], state["lon"], dest_lat, dest_lon)
            state["steps"]        = [_parse_step(s) for s in raw] if raw else []
            state["route_coords"] = coords
            state["osrm_error"]   = err
        else:
            state["steps"]        = []
            state["route_coords"] = []
            state["osrm_error"]   = None

    return jsonify({"ok": True, "steps": len(state["steps"])})


@app.route("/cancel_route")
def cancel_route():
    with state_lock:
        state["dest_lat"]    = None
        state["dest_lon"]    = None
        state["dest_label"]  = ""
        state["steps"]       = []
        state["route_coords"] = []
        state["osrm_error"]  = None
        state["current_step"] = 0
    return jsonify({"ok": True})


@app.route("/recalculate")
def recalculate():
    """Force route recalculation (called when off-route)."""
    with state_lock:
        if state["lat"] is None or state["dest_lat"] is None:
            return jsonify({"ok": False, "reason": "no position or destination"})
        raw, coords, err = _osrm_route(
            state["lat"], state["lon"],
            state["dest_lat"], state["dest_lon"]
        )
        state["steps"]        = [_parse_step(s) for s in raw] if raw else []
        state["route_coords"] = coords
        state["osrm_error"]   = err
        state["current_step"] = 0
    return jsonify({"ok": True, "steps": len(state["steps"])})


@app.route("/status")
def status():
    """JSON status for mobile page polling."""
    with state_lock:
        idx   = state["current_step"]
        steps = state["steps"]
        step  = steps[idx] if steps and idx < len(steps) else None
        total_dist = sum(s["distance_m"] for s in steps[idx:]) if steps else 0
        return jsonify({
            "lat":         state["lat"],
            "lon":         state["lon"],
            "dest_lat":    state["dest_lat"],
            "dest_lon":    state["dest_lon"],
            "dest_label":  state["dest_label"],
            "step_index":  idx,
            "total_steps": len(steps),
            "instruction": step["instruction"] if step else "Sin ruta",
            "distance_m":  step["distance_m"] if step else 0,
            "remaining_m": total_dist,
            "has_gps":      state["lat"] is not None,
            "updated_at":   state["updated_at"],
            "osrm_error":   state["osrm_error"],
            "route_coords": state["route_coords"],
            "steps_list":   [
                {"i": s["instruction"], "d": s["distance_m"]}
                for s in steps
            ],
        })


@app.route("/search")
def search():
    """
    Proxy Nominatim search so the mobile page doesn't hit it directly
    (Nominatim requires a proper User-Agent and we control it server-side).
    """
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify([])
    try:
        r = requests.get(
            NOMINATIM_BASE,
            params={"q": q, "format": "json", "limit": 6, "addressdetails": 1},
            headers={"User-Agent": NOMINATIM_UA},
            timeout=8,
        )
        r.raise_for_status()
        results = [
            {
                "label": item.get("display_name", ""),
                "lat":   float(item["lat"]),
                "lon":   float(item["lon"]),
            }
            for item in r.json()
        ]
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.route("/display")
def display():
    """
    Simple HTML page for the Honda's Opera Mini / ICS browser.
    Big text, no JS, meta-refresh every 5 s.
    """
    with state_lock:
        idx       = state["current_step"]
        steps     = state["steps"]
        lat       = state["lat"]
        lon       = state["lon"]
        dest_lbl  = state["dest_label"] or "—"
        updated   = state["updated_at"]
        total     = len(steps)

    step      = steps[idx] if steps and idx < len(steps) else None
    next_step = steps[idx + 1] if steps and idx + 1 < len(steps) else None

    if step:
        instruction = step["instruction"]
        dist_str    = _fmt_dist(step["distance_m"])
        icon        = step.get("icon", "↑")
        progress    = f"Paso {idx+1} de {total}"
        remaining_m = sum(s["distance_m"] for s in steps[idx:])
        remaining   = _fmt_dist(remaining_m)
    else:
        instruction = "Sin ruta — configura destino en el móvil"
        dist_str    = ""
        icon        = ""
        progress    = ""
        remaining   = ""

    next_text = f"{next_step['icon']} {next_step['instruction']}" if next_step else ""

    age = int(time.time() - updated) if updated else None
    map_url = _static_map_url(lat, lon) if lat else None

    return render_template(
        "display.html",
        instruction=instruction,
        dist_str=dist_str,
        icon=icon,
        progress=progress,
        remaining=remaining,
        next_text=next_text,
        dest_label=dest_lbl,
        map_url=map_url,
        gps_age=age,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
