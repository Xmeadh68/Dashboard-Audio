"""Fungsi bantu dashboard (dipisah dari app.py agar bisa diuji)."""
import numpy as np, pandas as pd, librosa

SR = 16000
ORDER = ["neutral", "happy", "surprise", "disgust", "disappointed"]
EMO = {"neutral": ("😐", "Netral", "#8E9AAF"), "happy": ("😄", "Senang", "#F4B942"), "surprise": ("😲", "Terkejut", "#F07C3A"),
       "disgust": ("🤢", "Jijik", "#5FA777"), "disappointed": ("😞", "Kecewa", "#4F86E8")}
SENT = {"neutral": "netral", "happy": "positif", "surprise": "positif", "disgust": "negatif", "disappointed": "negatif"}
SENT_COLOR = {"positif": "#2E9E6B", "netral": "#8E9AAF", "negatif": "#D2514F"}
METHOD_NAMES = {"mfcc": "MFCC", "mel": "Log-Mel", "prosodic": "Prosodik"}

def signal_views(y):
    """Waveform, log-mel, kontur pitch (F0) & energi dari rekaman pengguna."""
    hop = 512
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=SR, n_fft=2048, hop_length=hop, n_mels=96, fmax=8000), ref=np.max)
    f0, vf, _ = librosa.pyin(y, fmin=65, fmax=500, sr=SR, frame_length=2048, hop_length=hop)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    n = min(len(f0), len(rms)); t = librosa.times_like(rms, sr=SR, hop_length=hop)[:n]
    fv = f0[:n][vf[:n].astype(bool)]
    return dict(t_wave=np.arange(len(y)) / SR, y=y, mel=mel, t_mel=librosa.times_like(mel[0], sr=SR, hop_length=hop),
                t=t, f0=f0[:n], rms=rms[:n], dur=len(y) / SR,
                f0_mean=float(np.mean(fv)) if len(fv) else float("nan"), f0_range=float(np.ptp(fv)) if len(fv) else float("nan"),
                voiced=float(np.mean(vf[:n])) if n else 0.0, rms_mean=float(rms.mean()))

def sentiment_probs(labels, p):
    out = {"positif": 0.0, "netral": 0.0, "negatif": 0.0}
    for l, v in zip(labels, p): out[SENT[l]] += float(v)
    return out

def to_sent(s): return pd.Series(s).map(SENT).values

def top_confusions(yt, yp, labels, k=3):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(yt, yp, labels=labels)
    pairs = [(labels[i], labels[j], int(cm[i, j])) for i in range(len(labels)) for j in range(len(labels)) if i != j and cm[i, j] > 0]
    return sorted(pairs, key=lambda t: -t[2])[:k]

def nice_key(k):
    parts = k.split("|")
    return f"{METHOD_NAMES.get(parts[0], parts[0])} · {parts[2]}" + (f" ({parts[3]})" if len(parts) > 3 and parts[3] != "base" else "")
