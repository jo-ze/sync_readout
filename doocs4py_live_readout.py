import time
import asyncio
from collections import deque

import numpy as np
import h5py
import matplotlib.pyplot as plt
import doocs4py
import pydoocs



# ---------------------------------------------------------------------------
# 1) Bunch-Pattern-Dekodierung
# ---------------------------------------------------------------------------

_DEST_LUT_FLASH = {0: np.nan, 1: "LAdJ (Laser adjust)", 2: 2,
                   3: "GnD (Gun dump/DUMP)", 4: 1, 8: 3, 12: "Tuning Mode"}
_DEST_LUT_XFEL = {0: np.nan, 1: "Laser stand-alone", 2: 1, 3: "I1St = I1 Stand-by",
                  4: 2, 5: "I1D = Injector dump", 6: 3, 7: "B2D = B2 dump", 8: 4}


def decode_dest(pattern, facility=''):
    """Dekodiert das Bunch-Pattern-Wort -> (Zeit [us], Destination) pro Bunch.

    Destination-Bits sitzen an Bit 18-21 des Pattern-Worts (LSB=Bit0).
    Numerisch identisch zur alten String/bin()-Variante, aber vektorisiert.
    """
    lut, t0 = (_DEST_LUT_XFEL, 800e-6) if facility == 'XFEL' else (_DEST_LUT_FLASH, 700e-6)

    lut_arr = np.full(16, np.nan, dtype=object)
    for k, v in lut.items():
        lut_arr[k] = v

    bits = (np.asarray(pattern, dtype=np.uint32) >> 18) & 0xF
    dest = lut_arr[bits]

    t = (t0 + np.arange(len(pattern)) / 9.02777784047068e6) * 1e6  # -> us
    return np.column_stack([t, dest])


# ---------------------------------------------------------------------------
# 2) Maskieren nach Destination
# ---------------------------------------------------------------------------

# def mask_data(data, condition, pattern_key='pattern', facility='XFEL', skip_keys=('setpoint',)):
#     """Filtert alle Kanaele in `data` auf Bunche mit gegebener `condition` (Destination).

#     Erwartet je Kanal {'t':..., 'y':...}; berechnet den Sample-Offset zum
#     Pattern ueber die Zeitachse und wendet die passende Teilmaske an.
#     """
#     pattern = decode_dest(data[pattern_key]['y'], facility=facility)
#     mask_full = pattern[:, 1] == condition
#     dt = float(pattern[1, 0]) - float(pattern[0, 0])

#     out = {pattern_key: pattern[mask_full]}
#     for name, entry in data.items():
#         if name == pattern_key:
#             continue
#         if name in skip_keys:
#             out[name] = entry
#             continue

#         t, y = np.asarray(entry['t']), np.asarray(entry['y'])
#         offset = int(round((t[0] - pattern[0, 0]) / dt))
#         start, end = max(offset, 0), min(offset + len(t), len(mask_full))
#         m = mask_full[start:end]
#         d0 = max(-offset, 0)

#         out[name] = {
#             't': t[d0:d0 + len(m)][m],
#             'y': y[d0:d0 + len(m)][m],
#             'bunch_idx': (start + np.arange(len(m)))[m],
#         }
#     return out
# def mask_data(data, condition, keys=None, pattern_key='pattern', facility='XFEL'):
#     """Maskiert Kanaele auf die Bunche mit `condition`.
#        keys=None -> alle Kanaele in `data` (altes Verhalten, z.B. fuer track_data).
#        keys=[...] -> nur diese (z.B. fuer to_long_df, spart Arbeit)."""
#     pattern = decode_dest(data[pattern_key]['y'], facility=facility)
#     mask_full = pattern[:, 1] == condition
#     dt = np.round(float(pattern[1, 0]) - float(pattern[0, 0]),4)

#     print(dt)
    
#     names = keys if keys is not None else [k for k in data if k != pattern_key]

#     print(pattern[:,0])
#     pattern[:,0]=[np.round(item ,4) for item in pattern[:,0]]
#     #print(pattern[:,0])
    
#     out = {pattern_key: pattern[mask_full]}
#     for name in names:
#         try:
#             t, y = np.asarray(data[name]['t']), np.asarray(data[name]['y'])
#             offset = int(round((t[0] - pattern[0, 0]) / dt))
#             start, end = max(offset, 0), min(offset + len(t), len(mask_full))
#             m = mask_full[start:end]
#             d0 = max(-offset, 0)
#             out[name] = {
#                 't': t[d0:d0+len(m)][m],
#                 'y': y[d0:d0+len(m)][m],
#                 'bunch_idx': (start + np.arange(len(m)))[m],
#             }
#         except:
#             pass
#     return out


def mask_data(data, condition, keys=None, pattern_key='pattern', facility='XFEL', tol=None):
    """Maskiert Kanaele auf die Bunche mit `condition`.
       keys=None -> alle Kanaele in `data`.
       tol -> maximale Zeitabweichung fuer 'nearest match' (Default: halber Pattern-Tick,
              erlaubt jeden Kanal zuzuordnen, egal ob 1:1 zur Pattern-Rate oder downgesampelt)."""
    pattern = decode_dest(data[pattern_key]['y'], facility=facility)
    t_pattern = pattern[:, 0].astype(float)
    dest = pattern[:, 1]
    mask_full = dest == condition

    dt = t_pattern[1] - t_pattern[0]
    if tol is None:
        tol = dt / 2

    names = keys if keys is not None else [k for k in data if k != pattern_key]
    out = {pattern_key: pattern[mask_full]}
    for name in names:
        try:
            t, y = np.asarray(data[name]['t']), np.asarray(data[name]['y'])
            t = t.astype(float)

            idx = np.searchsorted(t_pattern, t)
            idx = np.clip(idx, 1, len(t_pattern) - 1)
            left = idx - 1
            choose_left = np.abs(t - t_pattern[left]) <= np.abs(t - t_pattern[idx])
            nearest_idx = np.where(choose_left, left, idx)

            diff = np.abs(t - t_pattern[nearest_idx])
            valid = diff <= tol

            nearest_idx = nearest_idx[valid]
            t_valid = t[valid]
            y_valid = y[valid]

            keep = mask_full[nearest_idx]
            out[name] = {
                't': t_valid[keep],
                'y': y_valid[keep],
                'bunch_idx': nearest_idx[keep],
            }
        except Exception:
            pass
    return out

# ---------------------------------------------------------------------------
# 3) Synchrones Auslesen
# ---------------------------------------------------------------------------

def read_channels_sync(channels, pattern_key='pattern', ref_key=None, retries=10, retry_wait=0.05):
    """Liest alle Kanaele (ausser `pattern_key`) bunch-synchron ueber eine
    gemeinsame Macropulse-ID aus. `pattern` ist nicht synchronisierbar -> separater get().

    Bei gescheitertem Sync (uneinheitliche Macropulse-IDs, z.B. durch Jitter)
    wird bis zu `retries`-mal neu versucht, bevor ein RuntimeError geworfen wird.
    """
    sync = {k: v for k, v in channels.items() if k != pattern_key}
    ref_key = ref_key or list(sync)[-1]

    for attempt in range(retries):
        mp_ref = doocs4py.get(sync[ref_key]).macropulse
        results = doocs4py.get_multi_sync(list(sync.values()), macropulse=mp_ref)
        mps = {d.macropulse for d in results}
        if len(mps) == 1:
            break
        time.sleep(retry_wait)
    else:
        raise RuntimeError(f"Sync fehlgeschlagen nach {retries} Versuchen: {mps}")

    mp_id = mps.pop()

    data = {}
    for name, d in zip(sync, results):
        s = d.get_data()
        if hasattr(s, 'get_time_trace'):
            data[name] = {'t': s.get_time_trace(), 'y': s.get_data(), 'timestamp': s.time}
        else:
            data[name] = {'y': s, 'timestamp': d.timestamp.get_seconds_since_epoch()}

    if pattern_key in channels:
        d = doocs4py.get(channels[pattern_key])
        data[pattern_key] = {'y': d.get_data(), 'timestamp': time.time()}

    return {mp_id: data}


# ---------------------------------------------------------------------------
# 4) Scan-Ausfuehrung
# ---------------------------------------------------------------------------

def run_scan(setpoints, set_addr, channels, wait=0.5):
    """Faehrt `set_addr` nacheinander auf `setpoints` und liest je Setpoint
    synchron aus. Ergebnis: {macropulse_id: {kanal: {...}, 'setpoint': sp}}.
    """
    scan_data = {}
    for sp in setpoints:
        doocs4py.set(set_addr, sp)  # TODO pruefen: sollte das nicht doocs4py.set(...) sein?
        time.sleep(wait)
        d = read_channels_sync(channels, retries=50, retry_wait=0.01)
        for chans in d.values():
            chans['setpoint'] = sp
        scan_data.update(d)
    return scan_data


def track_data_simple(n_shots, channels, wait=0.1, readback_key='gun_phase'):
    """Passiver Scan: liest n_shots mal alle Kanaele synchron, ohne etwas zu setzen.
    (readback_key wird hier nicht verwendet - vermutlich totes Argument.)
    """
    scan_data = {}
    for _ in range(n_shots):
        scan_data.update(read_channels_sync(channels))
        time.sleep(wait)
    return scan_data


# ---------------------------------------------------------------------------
# 5) Live-Tracking mit Plot + vollstaendiger Datenhistorie
# ---------------------------------------------------------------------------
# lauffaehige VErsion: 
# async def track_data(channels, n_shots=None, window=50, wait=0.1,
#                       readback_key='gun_phase', signal_key='bam_181_b1', condition=4,
#                       live=True):
#     """Liest laufend synchron aus, maskiert nach `condition`, plottet die letzten
#     `window` Shots als Heatmap (Bunch-Index x Shot) und sammelt die volle
#     (unmaskierte) Historie in scan_data, inkl. 'readback'.
#     """
#     x_buf, bunch_buf, y_buf = deque(maxlen=window), deque(maxlen=window), deque(maxlen=window)
#     scan_data = {}

#     fig = ax = im = cbar = None
#     n_bunch = None
#     if live:
#         fig, ax = plt.subplots(figsize=(9, 5))
#         im = ax.pcolormesh(np.arange(2), np.arange(window), np.full((window, 2), np.nan),
#                             shading='nearest', cmap='RdBu_r')
#         cbar = fig.colorbar(im, ax=ax, label='arrival time (fs)')
#         ax.set_xlabel('Bunch index')
#         ax.set_ylabel('shot in window (newest -> top)')
#         n_bunch = 2
#         plt.show()

#     i = 0
#     try:
#         while n_shots is None or i < n_shots:
#             d = read_channels_sync(channels)
#             mp_id, entry = next(iter(d.items()))

#             sp = np.mean(entry[readback_key]['y'])
#             dm = mask_data(entry, condition=condition)  # Achtung: default facility hier 'XFEL'

#             entry['readback'] = sp
#             scan_data[mp_id] = entry

#             x_buf.append(sp)
#             bunch_buf.append(dm[signal_key]['bunch_idx'])
#             y_buf.append(dm[signal_key]['y'])

#             if live:
#                 all_bunches = np.unique(np.concatenate(bunch_buf))
#                 bunch_to_col = {b: c for c, b in enumerate(all_bunches)}
#                 n = len(x_buf)

#                 M = np.full((window, len(all_bunches)), np.nan)
#                 for row, (bidx, y) in enumerate(zip(bunch_buf, y_buf)):
#                     M[window - n + row, [bunch_to_col[b] for b in bidx]] = y

#                 if len(all_bunches) != n_bunch:
#                     ax.clear()
#                     im = ax.pcolormesh(all_bunches, np.arange(window), M,
#                                         shading='nearest', cmap='RdBu_r')
#                     ax.set_xlabel('Bunch index')
#                     ax.set_ylabel('shot in window (newest -> top)')
#                     n_bunch = len(all_bunches)
#                 else:
#                     im.set_array(M.ravel())
#                     valid = ~np.isnan(M)
#                     if valid.any():
#                         im.set_clim(np.nanmin(M), np.nanmax(M))

#                 cbar.update_normal(im)
#                 ax.set_title(f'shot {i}')
#                 fig.canvas.draw_idle()

#             await asyncio.sleep(wait)
#             i += 1
#     except (KeyboardInterrupt, asyncio.CancelledError):
#         print(f'Gestoppt bei shot {i}')

#     return scan_data, x_buf, bunch_buf, y_buf


def _match_by_bunch(dm, keys):
    """Matched Werte fuer `keys` ueber gemeinsame bunch_idx (Mengenschnitt).
       Generalisiert die alte intersect1d-Logik auf beliebig viele Kanaele."""
    common = None
    for k in keys:
        s = set(dm[k]['bunch_idx'].tolist())
        common = s if common is None else common & s
    common = np.array(sorted(common))
    out = []
    for k in keys:
        lut = dict(zip(dm[k]['bunch_idx'].tolist(), dm[k]['y']))
        out.append(np.array([lut[b] for b in common]))
    return common, out


async def track_data(channels, n_shots=None, window=50, wait=0.1,
                      readback_key='gun_phase', signal_key='bam_181_b1', condition=4,
                      live=True, plot_mode='heatmap', x_key=None, y_key=None, c_key=None,
                      facility='XFEL'):
    """plot_mode:
       'heatmap'   - bisheriges Verhalten
       'scatter'   - x_key vs y_key, Farbe=bunch_idx, bunch-gematcht
       'scatter3d' - x_key vs y_key vs bunch_idx (Z), Farbe=c_key, bunch-gematcht
       Bei 'scatter'/'scatter3d' haelt x_buf/y_buf/c_buf die gematchten Arrays pro Shot,
       nicht mehr den skalaren readback."""
    if plot_mode == 'scatter' and (x_key is None or y_key is None):
        raise ValueError("plot_mode='scatter' braucht x_key und y_key")
    if plot_mode == 'scatter3d' and (x_key is None or y_key is None or c_key is None):
        raise ValueError("plot_mode='scatter3d' braucht x_key, y_key und c_key")

    x_buf, bunch_buf, y_buf = deque(maxlen=window), deque(maxlen=window), deque(maxlen=window)
    c_buf = deque(maxlen=window)
    scan_data = {}

    fig = ax = im = cbar = sc = None
    n_bunch = None
    if live:
        if plot_mode == 'scatter3d':
            fig = plt.figure(figsize=(9, 7))
            ax = fig.add_subplot(111, projection='3d')
            sc = ax.scatter([], [], [], c=[], cmap='viridis', s=10)
            cbar = fig.colorbar(sc, ax=ax, label=c_key)
            ax.set_xlabel(x_key)
            ax.set_ylabel(y_key)
            ax.set_zlabel('bunch_idx')
        elif plot_mode == 'scatter':
            fig, ax = plt.subplots(figsize=(9, 5))
            sc = ax.scatter([], [], c=[], cmap='viridis', s=10)
            cbar = fig.colorbar(sc, ax=ax, label='bunch_idx')
            ax.set_xlabel(x_key)
            ax.set_ylabel(y_key)
        else:
            fig, ax = plt.subplots(figsize=(9, 5))
            im = ax.pcolormesh(np.arange(2), np.arange(window), np.full((window, 2), np.nan),
                                shading='nearest', cmap='RdBu_r')
            cbar = fig.colorbar(im, ax=ax, label='arrival time (fs)')
            ax.set_xlabel('Bunch index')
            ax.set_ylabel('shot in window (newest -> top)')
            n_bunch = 2
        plt.show()
        # Kaltstart-Fix: Comm-Kanal des Widget-Backends etablieren, bevor die
        # Schleife mit draw_idle() loslegt. Der zusaetzliche await gibt dem
        # Jupyter-Kernel einen echten Yield-Punkt, um den ipympl-Handshake
        # abzuschliessen (ein synchrones draw() allein reichte offenbar nicht).
        fig.canvas.draw()
        fig.canvas.flush_events()
        await asyncio.sleep(0.2)

    i = 0
    try:
        while n_shots is None or i < n_shots:
            d = read_channels_sync(channels)
            mp_id, entry = next(iter(d.items()))

            sp = np.mean(entry[readback_key]['y'])

            if plot_mode == 'scatter3d':
                dm = mask_data(entry, condition=condition, facility=facility,
                                keys=[x_key, y_key, c_key])
                common, (xv, yv, cv) = _match_by_bunch(dm, [x_key, y_key, c_key])
                x_buf.append(xv)
                y_buf.append(yv)
                c_buf.append(cv)
                bunch_buf.append(common)
            elif plot_mode == 'scatter':
                dm = mask_data(entry, condition=condition, facility=facility, keys=[x_key, y_key])
                common, (xv, yv) = _match_by_bunch(dm, [x_key, y_key])
                x_buf.append(xv)
                y_buf.append(yv)
                bunch_buf.append(common)
            else:
                dm = mask_data(entry, condition=condition, facility=facility)
                x_buf.append(sp)
                bunch_buf.append(dm[signal_key]['bunch_idx'])
                y_buf.append(dm[signal_key]['y'])

            entry['readback'] = sp
            scan_data[mp_id] = entry

            if live:
                if plot_mode == 'scatter3d':
                    xs = np.concatenate(x_buf)
                    ys = np.concatenate(y_buf)
                    zs = np.concatenate(bunch_buf).astype(float)
                    cs = np.concatenate(c_buf)

                    sc._offsets3d = (xs, ys, zs)
                    sc.set_array(cs)
                    if len(cs):
                        sc.set_clim(cs.min(), cs.max())
                    if len(xs):
                        ax.set_xlim(xs.min(), xs.max())
                        ax.set_ylim(ys.min(), ys.max())
                        ax.set_zlim(zs.min(), zs.max())
                    ax.set_title(f'shot {i}')
                elif plot_mode == 'scatter':
                    xs = np.concatenate(x_buf)
                    ys = np.concatenate(y_buf)
                    cs = np.concatenate(bunch_buf).astype(float)

                    sc.set_offsets(np.column_stack([xs, ys]))
                    sc.set_array(cs)
                    if len(cs):
                        sc.set_clim(cs.min(), cs.max())
                    ax.update_datalim(np.column_stack([xs, ys]))
                    ax.autoscale_view()
                    ax.set_title(f'shot {i}')
                else:
                    all_bunches = np.unique(np.concatenate(bunch_buf))
                    bunch_to_col = {b: c for c, b in enumerate(all_bunches)}
                    n = len(x_buf)

                    M = np.full((window, len(all_bunches)), np.nan)
                    for row, (bidx, y) in enumerate(zip(bunch_buf, y_buf)):
                        M[window - n + row, [bunch_to_col[b] for b in bidx]] = y

                    if len(all_bunches) != n_bunch:
                        ax.clear()
                        im = ax.pcolormesh(all_bunches, np.arange(window), M,
                                            shading='nearest', cmap='RdBu_r')
                        ax.set_xlabel('Bunch index')
                        ax.set_ylabel('shot in window (newest -> top)')
                        n_bunch = len(all_bunches)
                    else:
                        im.set_array(M.ravel())
                        valid = ~np.isnan(M)
                        if valid.any():
                            im.set_clim(np.nanmin(M), np.nanmax(M))

                    cbar.update_normal(im)
                    ax.set_title(f'shot {i}')

                fig.canvas.draw_idle()
                fig.canvas.flush_events()

            await asyncio.sleep(wait)
            i += 1
    except (KeyboardInterrupt, asyncio.CancelledError):
        print(f'Gestoppt bei shot {i}')

    return scan_data, x_buf, bunch_buf, y_buf, c_buf

# ---------------------------------------------------------------------------
# 6) Speichern der Scan-Daten (HDF5)
# ---------------------------------------------------------------------------

def save_scan_data(fname, scan_data):
    """Speichert scan_data {mp_id: {kanal: {'t','y',...} | skalar}} als HDF5.

    Struktur: /<mp_id>/<kanal>/{t,y,timestamp,bunch_idx,...} als Datasets;
    skalare Eintraege (z.B. 'setpoint', 'readback') werden als Attribute
    der jeweiligen mp_id-Gruppe abgelegt.
    """
    with h5py.File(fname, 'w') as f:
        for mp_id, entry in scan_data.items():
            grp = f.create_group(str(mp_id))
            for name, val in entry.items():
                if isinstance(val, dict):
                    sub = grp.create_group(name)
                    for k, v in val.items():
                        sub.create_dataset(k, data=np.asarray(v))
                else:
                    grp.attrs[name] = val