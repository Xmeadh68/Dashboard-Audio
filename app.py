"""Dashboard Speech Emotion Recognition - IndoWaveSentiment. Jalankan: streamlit run app.py"""
import json, sys, tempfile, time, hashlib
from pathlib import Path
import numpy as np, pandas as pd, streamlit as st
import plotly.express as px, plotly.graph_objects as go
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import app_utils as U

HERE = Path(__file__).parent; A = HERE / "assets"
st.set_page_config(page_title="Suara & Emosi", page_icon="🎙️", layout="wide")
st.markdown("""<style>
.card{border-radius:18px;padding:18px 22px;color:#fff;text-align:center}
.card .e{font-size:68px;line-height:1.1}.card .n{font-size:30px;font-weight:700}.card .c{opacity:.9}
.pill{display:inline-block;padding:4px 12px;border-radius:999px;color:#fff;font-weight:600;margin:2px}
.hint{color:#6b7280;font-size:0.9rem}
</style>""", unsafe_allow_html=True)

@st.cache_data
def assets():
    d = {n: (pd.read_csv(A / f"{n}.csv") if (A / f"{n}.csv").exists() else None)
         for n in ("results_all", "ablation_all", "leaderboard", "dataset_meta", "oof_all")}
    d["sel"] = json.load(open(A / "selection.json")) if (A / "selection.json").exists() else None
    return d

@st.cache_resource
def model():
    try:
        sys.path.insert(0, str(HERE / "src")); import infer
        return infer.load(str(HERE / "model"), str(HERE / "src")), infer, None
    except Exception as e:
        return None, None, repr(e)

def predict(f):
    pre, infer, err = model()
    if pre is None: st.error(f"Model belum termuat: {err}"); st.stop()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as t: t.write(f.getvalue())
    labels, p = infer.predict(pre, t.name); return t.name, labels, p

def audio_in(key):
    c1, c2 = st.columns(2); up = c1.file_uploader("Unggah file .wav", type=["wav"], key=f"u_{key}")
    rec = c2.audio_input("…atau rekam suaramu", key=f"r_{key}") if hasattr(st, "audio_input") else None
    return up or rec

def prob_chart(labels, p):
    d = pd.DataFrame({"emosi": labels, "p": p}); d["nama"] = d.emosi.map(lambda e: U.EMO[e][0] + " " + U.EMO[e][1])
    d = d.set_index("emosi").loc[U.ORDER].reset_index()
    fig = go.Figure(go.Bar(x=d.p, y=d.nama, orientation="h", marker_color=[U.EMO[e][2] for e in d.emosi],
                           text=[f"{v:.0%}" for v in d.p], textposition="outside"))
    fig.update_layout(height=300, xaxis=dict(range=[0, 1.1], tickformat=".0%"), yaxis=dict(autorange="reversed"), margin=dict(l=0, r=10, t=10, b=0))
    return fig

def sent_bar(labels, p):
    s = U.sentiment_probs(labels, p)
    fig = go.Figure([go.Bar(x=[v], y=["Sentimen"], orientation="h", name=k, marker_color=U.SENT_COLOR[k], text=f"{k} {v:.0%}", textposition="inside") for k, v in s.items()])
    fig.update_layout(barmode="stack", height=110, showlegend=False, xaxis=dict(visible=False, range=[0, 1]), margin=dict(l=0, r=0, t=0, b=0)); return fig

def signal_tab(path):
    import librosa
    v = U.signal_views(librosa.load(path, sr=U.SR, mono=True)[0])
    c = st.columns(4); c[0].metric("Durasi", f"{v['dur']:.1f} dtk"); c[1].metric("Pitch rata-rata", "–" if np.isnan(v["f0_mean"]) else f"{v['f0_mean']:.0f} Hz")
    c[2].metric("Rentang pitch", "–" if np.isnan(v["f0_range"]) else f"{v['f0_range']:.0f} Hz"); c[3].metric("Bagian bersuara", f"{v['voiced']:.0%}")
    st.caption("Model hanya mendengar **intonasi** (dataset memakai satu kalimat yang sama), jadi kontur pitch dan energi di bawah adalah petunjuk utamanya.")
    f1 = go.Figure(go.Scatter(x=v["t_wave"][::20], y=v["y"][::20], line=dict(width=1))); f1.update_layout(height=170, title="Gelombang suara", margin=dict(t=30, b=0)); st.plotly_chart(f1, width="stretch")
    f2 = px.imshow(v["mel"], x=v["t_mel"], aspect="auto", origin="lower", color_continuous_scale="magma", title="Log-mel spectrogram"); f2.update_layout(height=260, margin=dict(t=30, b=0)); st.plotly_chart(f2, width="stretch")
    f3 = go.Figure(); f3.add_scatter(x=v["t"], y=v["f0"], name="Pitch F0 (Hz)", mode="lines+markers"); f3.add_scatter(x=v["t"], y=v["rms"], name="Energi (RMS)", yaxis="y2")
    f3.update_layout(height=260, title="Kontur pitch & energi", yaxis=dict(title="Hz"), yaxis2=dict(overlaying="y", side="right", title="RMS"), margin=dict(t=30, b=0)); st.plotly_chart(f3, width="stretch")

D = assets(); sel = D["sel"]
st.title("🎙️ Suara & Emosi")
st.caption("Klasifikasi emosi dari intonasi suara bahasa Indonesia · dataset IndoWaveSentiment · model dipilih dari perbandingan 3 metode fitur × 3 lapis model")
if sel is None: st.warning("Aset hasil belum ada. Jalankan notebook *Evaluasi Gabungan* untuk membuat `assets/` dan `model/`.")

T = st.tabs(["🎤 Prediksi", "🏆 Perbandingan model", "🔍 Analisis kesalahan", "🎯 Tantangan emosi", "📚 Tentang dataset"])

# ---------------- Prediksi ----------------
with T[0]:
    if sel: st.info(f"Model aktif: **{sel['model']}** pada fitur **{U.METHOD_NAMES[sel['metode']]}** · macro-F1 validasi silang {sel['f1']:.2f} ± {sel['f1_std']:.2f}")
    f = audio_in("pred")
    if not f: st.markdown("<p class='hint'>Ucapkan kalimat apa saja selama 2–4 detik dengan emosi tertentu. Model dilatih pada satu kalimat dari 10 aktor profesional, jadi hasil pada kalimat lain bisa kurang akurat.</p>", unsafe_allow_html=True)
    else:
        st.audio(f); path, labels, p = predict(f); top = labels[int(np.argmax(p))]; e, nm, col = U.EMO[top]
        c1, c2 = st.columns([1, 2])
        c1.markdown(f"<div class='card' style='background:{col}'><div class='e'>{e}</div><div class='n'>{nm}</div><div class='c'>keyakinan {p.max():.0%}</div></div>", unsafe_allow_html=True)
        c1.markdown(f"<span class='pill' style='background:{U.SENT_COLOR[U.SENT[top]]}'>sentimen {U.SENT[top]}</span>", unsafe_allow_html=True)
        c2.plotly_chart(prob_chart(labels, p), width="stretch"); c2.plotly_chart(sent_bar(labels, p), width="stretch")
        order = np.argsort(p)[::-1]
        if p[order[0]] < 0.5: st.warning("Model kurang yakin. Coba ucapkan dengan intonasi lebih tegas, atau rekam di tempat yang lebih tenang.")
        if {labels[order[0]], labels[order[1]]} == {"happy", "surprise"}: st.info("Senang dan terkejut memang sering tertukar, bahkan pada penilaian manusia di survei validasi dataset.")
        with st.expander("🔬 Lihat di balik suaramu (pitch, energi, spektrogram)", expanded=True): signal_tab(path)

# ---------------- Perbandingan ----------------
with T[1]:
    R, S = D["results_all"], sel
    if R is None: st.info("Belum ada hasil.")
    else:
        base = R[R.varian == "base"]
        st.success(f"🏆 Model terpilih: **{S['model']}** · fitur **{U.METHOD_NAMES[S['metode']]}** · lapis **{S['lapis']}** · macro-F1 **{S['f1']:.3f} ± {S['f1_std']:.3f}**")
        g = base[base.lapis.isin(["ML", "MLP"])].groupby(["model", "metode"]).macro_f1.mean().unstack().reindex([m for m in ["SVM", "RF", "XGBoost", "MLP"] if m in set(base.model)])
        st.subheader("Fitur mana yang paling informatif? (Lapis 1 & 2: model sama, fitur beda)")
        fig = px.imshow(g.rename(columns=U.METHOD_NAMES), text_auto=".2f", color_continuous_scale="YlGn", zmin=0, zmax=1, aspect="auto"); st.plotly_chart(fig, width="stretch")
        st.caption("Rata-rata per metode: " + " · ".join(f"{U.METHOD_NAMES[k]} {v:.3f}" for k, v in g.mean().items()))
        a = base.groupby(["metode", "lapis", "model"]).agg(f1=("macro_f1", "mean"), sd=("macro_f1", "std"), acc=("accuracy", "mean"), ms=("infer_time_ms", "mean")).reset_index()
        a["metode"] = a.metode.map(U.METHOD_NAMES); a["nama"] = a.metode + " · " + a.model
        lp = st.multiselect("Tampilkan lapis", sorted(a.lapis.unique()), default=sorted(a.lapis.unique()))
        b = a[a.lapis.isin(lp)].sort_values("f1", ascending=False)
        st.plotly_chart(px.bar(b, x="nama", y="f1", error_y="sd", color="metode", labels={"f1": "macro-F1", "nama": ""}, height=380), width="stretch")
        st.dataframe(b[["metode", "lapis", "model", "f1", "sd", "acc", "ms"]].round(3), width="stretch", hide_index=True,
                     column_config={"f1": st.column_config.ProgressColumn("macro-F1", min_value=0, max_value=1), "sd": "std antar fold", "acc": "akurasi", "ms": "ms/prediksi"})
        st.subheader("Apakah denoising/augmentasi membantu? (ablasi)")
        AB = D["ablation_all"]
        if AB is not None and (AB.varian != "base").any():
            rows = []
            for (m, mod), gg in AB.groupby(["metode", "model"]):
                bs = gg[gg.varian == "base"].set_index("fold").macro_f1
                for v in [x for x in gg.varian.unique() if x != "base"]:
                    d_ = gg[gg.varian == v].set_index("fold").macro_f1 - bs
                    rows.append(dict(perlakuan=f"{U.METHOD_NAMES[m]} · {mod} · {v}", selisih=d_.mean(), fold_membaik=f"{int((d_ > 0).sum())}/{len(d_)} fold"))
            ab = pd.DataFrame(rows); fig = px.bar(ab, x="selisih", y="perlakuan", orientation="h", text="fold_membaik", color=ab.selisih > 0, color_discrete_map={True: "#2E9E6B", False: "#D2514F"})
            fig.update_layout(showlegend=False, height=60 + 50 * len(ab), xaxis_title="Δ macro-F1 dibanding langkah dasar"); st.plotly_chart(fig, width="stretch")
            st.caption("Dengan 5 fold, selisih kecil (kurang dari 1 std) belum tentu bermakna.")
        else: st.caption("Belum ada hasil ablasi.")

# ---------------- Analisis kesalahan ----------------
with T[2]:
    O, M = D["oof_all"], D["dataset_meta"]
    if O is None: st.info("Belum ada prediksi out-of-fold.")
    else:
        cols = [c for c in O.columns if c.count("|") >= 3 and c.endswith("|base")]
        c1, c2, c3 = st.columns([3, 1.3, 1.3])
        col = c1.selectbox("Model", cols, index=cols.index(sel["key"]) if sel and sel["key"] in cols else 0, format_func=U.nice_key)
        mode = c2.radio("Tampilan", ["5 emosi", "3 sentimen"], horizontal=True); norm = c3.checkbox("Normalisasi baris", True)
        yt, yp = O.y_true.values, O[col].values
        if mode == "3 sentimen": yt, yp = U.to_sent(yt), U.to_sent(yp); labs = ["positif", "netral", "negatif"]
        else: labs = U.ORDER
        cm = pd.crosstab(pd.Categorical(yt, labs), pd.Categorical(yp, labs), dropna=False).values
        z = cm / cm.sum(1, keepdims=True) if norm else cm
        fig = px.imshow(z, x=labs, y=labs, text_auto=".0%" if norm else True, color_continuous_scale="Blues", aspect="auto", labels=dict(x="Prediksi", y="Sebenarnya"))
        l, r = st.columns(2); l.subheader("Confusion matrix"); l.plotly_chart(fig, width="stretch")
        pr, rc, f1, _ = precision_recall_fscore_support(yt, yp, labels=labs, zero_division=0)
        d = pd.DataFrame({"kelas": labs, "presisi": pr, "recall": rc, "F1": f1}).melt("kelas")
        r.subheader("Presisi, recall, F1 per kelas"); r.plotly_chart(px.bar(d, x="kelas", y="value", color="variable", barmode="group", height=380), width="stretch")
        st.metric("Akurasi gabungan", f"{accuracy_score(yt, yp):.1%}")
        tc = U.top_confusions(yt, yp, labs); 
        if tc: st.markdown("**Kesalahan terbanyak:** " + " · ".join(f"{a_} → {b_} ({n}×)" for a_, b_, n in tc))
        if M is not None:
            dd = M.merge(O[["filename", "y_true", col]], on="filename"); dd["benar"] = dd.y_true == dd[col]
            x1, x2, x3 = st.columns(3)
            for w, (k, ttl) in zip((x1, x2, x3), (("gender", "Gender (L/P)"), ("intensity", "Intensitas"), ("actor", "Aktor"))):
                s = dd.groupby(k).benar.mean().reset_index(); s[k] = s[k].astype(str)
                w.markdown(f"**Akurasi per {ttl}**"); w.plotly_chart(px.bar(s, x=k, y="benar", range_y=[0, 1], height=260), width="stretch")

# ---------------- Tantangan ----------------
with T[3]:
    st.subheader("🎯 Tantangan: bisakah kamu mengelabui AI?")
    st.write("Pilih emosi target, ucapkan kalimat dengan emosi itu, dan lihat seberapa yakin model. Skor = peluang emosi target menurut model.")
    tgt = st.radio("Emosi target", U.ORDER, format_func=lambda e: f"{U.EMO[e][0]} {U.EMO[e][1]}", horizontal=True)
    f = audio_in("chal"); st.session_state.setdefault("hist", []); st.session_state.setdefault("seen", set())
    if f:
        uid = (hashlib.md5(f.getvalue()).hexdigest(), tgt); _, labels, p = predict(f); sc = float(p[labels.index(tgt)]) * 100; top = labels[int(np.argmax(p))]
        if uid not in st.session_state["seen"]:
            st.session_state["seen"].add(uid); st.session_state["hist"].append(dict(waktu=time.strftime("%H:%M:%S"), target=U.EMO[tgt][1], terdeteksi=U.EMO[top][1], skor=round(sc)))
        st.metric("Skor", f"{sc:.0f} / 100", help="peluang emosi target menurut model")
        st.progress(min(sc / 100, 1.0)); st.plotly_chart(prob_chart(labels, p), width="stretch")
        st.success("🔥 Ekspresimu meyakinkan!" if sc >= 70 else "👍 Lumayan, tapi model masih ragu." if sc >= 40 else f"🤔 Model mendengarnya sebagai {U.EMO[top][1].lower()}. Coba ubah nada, tempo, atau kerasnya suara.")
    H = pd.DataFrame(st.session_state["hist"])
    if len(H):
        c1, c2 = st.columns([2, 1]); c1.dataframe(H.iloc[::-1], width="stretch", hide_index=True)
        c2.metric("Skor terbaik", int(H.skor.max())); c2.metric("Rata-rata", f"{H.skor.mean():.0f}")
        st.plotly_chart(px.line(H, y="skor", markers=True, range_y=[0, 100], height=240), width="stretch")

# ---------------- Dataset ----------------
with T[4]:
    M = D["dataset_meta"]; st.subheader("IndoWaveSentiment")
    c = st.columns(4); c[0].metric("Rekaman", "300"); c[1].metric("Aktor", "10 (5 L, 5 P)"); c[2].metric("Emosi", "5"); c[3].metric("Intensitas", "normal & strong")
    st.write("Setiap aktor mengucapkan kalimat yang sama, *“Kualitas HP ini cukup bagus”*, dengan 5 emosi × 2 intensitas × 3 pengulangan di studio rekaman. Karena kalimatnya sama, model hanya belajar dari **intonasi**.")
    if M is not None:
        d = M.groupby(["emotion", "intensity"]).size().reset_index(name="n"); st.plotly_chart(px.bar(d, x="emotion", y="n", color="intensity", barmode="group", height=280), width="stretch")
    sm = sorted((A / "samples").glob("*.wav")) if (A / "samples").exists() else []
    if sm:
        st.markdown("**Contoh audio (aktor 01, intensitas normal)**"); cc = st.columns(len(sm))
        for w, s in zip(cc, sm): w.markdown(f"{U.EMO.get(s.stem, ('', s.stem))[0]} {s.stem}"); w.audio(str(s))
    with st.expander("Tiga metode fitur yang dibandingkan"):
        st.markdown("- **MFCC** (40 koefisien + delta + delta-delta)  \n- **Log-Mel Spectrogram** (128 band)  \n- **Prosodik** (ZCR, RMS, spectral centroid/rolloff/contrast, chroma, F0)  \nMasing-masing diuji dengan ML klasik, MLP identik, dan model DL khusus, dengan validasi silang per aktor (5 fold).")
    st.warning("Keterbatasan: hanya 10 aktor profesional, satu kalimat, emosi diperankan, dan label *happy* sering tertukar dengan *surprise*. Jangan dipakai untuk keputusan nyata tentang seseorang.")
    st.caption("Bustamin, A., Rizky, A.M., Warni, E., Areni, I.S., Indrabayu (2024). IndoWaveSentiment: Indonesian audio dataset for emotion classification. Data in Brief 57, 111138. DOI 10.1016/j.dib.2024.111138 · Lisensi CC BY-NC 4.0 (non-komersial).")
