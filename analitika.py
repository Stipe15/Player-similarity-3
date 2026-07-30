"""Analitički sloj: percentili, 2D ugradnje, jedinstvenost i Altair grafovi.

Modul namjerno NE uvozi streamlit — sve su funkcije čiste, pa ih je lako
testirati i cachirati izvana. Podijeljen je na dva dijela:

    1. RAČUN      — percentili, ugradnje, matrica sličnosti, jedinstvenost
    2. GRAFOVI    — graditelji Altair grafova (svaki vraća alt.Chart)

Grafovi koriste Altair jer dolazi zajedno sa Streamlitom, pa aplikacija za
vizualizacije ne dobiva nijednu novu ovisnost pri deployu.
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import prijevodi as p
import similarity as s

# ---------------------------------------------------------------------------
# Paleta (usklađena s tamnom temom aplikacije)
# ---------------------------------------------------------------------------
TAMNA = "#0E1114"
LINIJA = "#262C34"
TEKST = "#9AA2AC"
TEKST_JAK = "#E9EBEE"
AKCENT = "#D9637A"
PRIGUSENO = "#4A525C"

# Obitelji statistika izvedene iz prefiksa stupaca u similarity.ZNACAJKE.
KATEGORIJE = {
    "ATT_": "Napad",
    "CAR_": "Vođenje lopte",
    "DEF_": "Obrana",
    "PAS_": "Dodavanje",
}

BOJE_KATEGORIJA = {
    "Napad": "#D9637A",
    "Vođenje lopte": "#C9A227",
    "Obrana": "#5B8FB9",
    "Dodavanje": "#6FBF9A",
}


def kategorija(stupac: str) -> str:
    """Obitelj statistike ('Napad', 'Obrana', …) iz prefiksa naziva stupca."""
    for prefiks, naziv in KATEGORIJE.items():
        if stupac.startswith(prefiks):
            return naziv
    return "Ostalo"


# ---------------------------------------------------------------------------
# 1. RAČUN
# ---------------------------------------------------------------------------
def percentili(prostor: s.ProstorIgraca) -> np.ndarray:
    """Percentilni rang svakog igrača po svakoj značajci, u odnosu na SVE igrače.

    Vraća matricu (n × d) s vrijednostima u [0, 100].
    """
    n = len(prostor.X)
    return rankdata(prostor.X, axis=0) / n * 100.0


def percentili_po_ulozi(prostor: s.ProstorIgraca) -> np.ndarray:
    """Isto, ali rang se računa UNUTAR pozicijske uloge svakog igrača.

    Postotak dodavanja stopera znači nešto posve drugo naspram napadača nego
    naspram drugih stopera, pa aplikacija nudi obje osnovice.
    """
    izlaz = np.zeros_like(prostor.X, dtype=float)
    uloge = prostor.uloge.to_numpy()
    for oznaka in np.unique(uloge):
        maska = uloge == oznaka
        skupina = prostor.X[maska]
        izlaz[maska] = rankdata(skupina, axis=0) / len(skupina) * 100.0
    return izlaz


def ugradnja_2d(prostor: s.ProstorIgraca, metoda: str = "pca") -> np.ndarray:
    """2D projekcija svih igrača za mapu sličnosti.

    Ugradnja je GLOBALNA — ista za svakog igrača, mijenja se samo isticanje —
    pa se izvana cachira i računa jednom za cijelu aplikaciju.
    """
    if metoda == "tsne":
        model = TSNE(n_components=2, random_state=42, perplexity=30, init="pca")
        return model.fit_transform(prostor.X)
    return PCA(n_components=2, random_state=42).fit_transform(prostor.X)


def matrica_slicnosti(
    prostor: s.ProstorIgraca,
    alpha: float = 1.0,
    koristi_M: bool = True,
    udaljenost: bool = False,
) -> np.ndarray:
    """Puna n×n matrica sličnosti (VEĆE = sličnije).

    Vektorizirana inačica `similarity.rezultati_slicnosti` — ista formula, ali
    odjednom za sve parove. Pozivanje po retku bilo bi preskupo jer se član
    aᵀMa računa einsumom, pa bi se posao ponavljao 1828 puta.
    """
    X = prostor.X

    if udaljenost:
        kvadrati = (X**2).sum(axis=1)
        d2 = kvadrati[:, None] + kvadrati[None, :] - 2.0 * (X @ X.T)
        return -np.sqrt(np.clip(d2, 0.0, None))

    G = X @ (prostor.M @ X.T) if koristi_M else X @ X.T
    if alpha == 0.0:
        return G

    norme = np.sqrt(np.clip(np.diag(G), 1e-12, None))
    return G / np.power(np.outer(norme, norme), alpha)


def jedinstvenost(
    S: np.ndarray, idx: int, k: int = 10
) -> tuple[np.ndarray, float, float]:
    """Koliko je igrač `idx` neponovljiv u odnosu na ostatak lige.

    Ideja: igrač čijih je 10 najsličnijih i dalje daleko od njega je rijedak
    profil; igrač okružen gomilom bliskih dvojnika je lako zamjenjiv. Zato
    uspoređujemo njegov prosjek top-k sličnosti s istom veličinom za SVE igrače.

    Vraća (rezultati_prema_ostalima, prosjek_top_k, jedinstvenost_0_100),
    gdje je 100 = najrjeđi profil u skupu.
    """
    S = S.copy()
    np.fill_diagonal(S, -np.inf)

    # Prosjek k najvećih sličnosti u svakom retku (bez samog igrača).
    najblizi = np.partition(S, -k, axis=1)[:, -k:]
    prosjeci = najblizi.mean(axis=1)

    rang = rankdata(prosjeci)[idx] / len(prosjeci) * 100.0
    rezultati = S[idx]
    return rezultati, float(prosjeci[idx]), float(100.0 - rang)


# ---------------------------------------------------------------------------
# 2. GRAFOVI
# ---------------------------------------------------------------------------
def primijeni_temu(graf: alt.Chart) -> alt.Chart:
    """Tamna tema usklađena s ostatkom aplikacije.

    Konfiguriramo graf izravno (umjesto globalne Altair teme) jer se API za
    registraciju teme mijenjao između Altair verzija.
    """
    return (
        graf.configure(background=TAMNA)
        .configure_axis(
            labelColor=TEKST,
            titleColor=TEKST,
            gridColor=LINIJA,
            domainColor=LINIJA,
            tickColor=LINIJA,
            labelFontSize=11,
            titleFontSize=11,
        )
        .configure_legend(labelColor=TEKST, titleColor=TEKST, labelFontSize=11)
        .configure_view(stroke=None)
    )


def graf_percentila(
    prostor: s.ProstorIgraca, idx: int, pct: np.ndarray, osnovica: str, jezik: str = "hr"
) -> alt.Chart:
    """Vodoravne trake: percentil igrača po svakoj značajci, po obiteljima.

    Radar-graf namjerno NE koristimo — iskrivljuje površinu i otežava čitanje
    pojedinačne vrijednosti. Trake se čitaju izravno i sortirane su po obitelji.
    """
    redci = []
    for j, stupac in enumerate(prostor.znacajke):
        redci.append(
            {
                "znacajka": p.znacajka(stupac, jezik),
                "percentil": float(pct[idx, j]),
                "kategorija": p.kategorija_naziv(kategorija(stupac), jezik),
                "vrijednost": float(prostor.df.iloc[idx][stupac]),
            }
        )
    df = pd.DataFrame(redci)

    # Unutar obitelji sortiramo silazno po percentilu; obitelji idu redom.
    domena_kat = [p.kategorija_naziv(k, jezik) for k in BOJE_KATEGORIJA]
    redoslijed_kat = [k for k in domena_kat if k in set(df["kategorija"])]
    df["_k"] = df["kategorija"].map({k: i for i, k in enumerate(redoslijed_kat)})
    df = df.sort_values(["_k", "percentil"], ascending=[True, False])
    poredak = df["znacajka"].tolist()

    trake = (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=2, height=13)
        .encode(
            x=alt.X(
                "percentil:Q",
                scale=alt.Scale(domain=[0, 100]),
                title=p.t("graf_pct_x_title", jezik, osnovica=osnovica),
            ),
            y=alt.Y("znacajka:N", sort=poredak, title=None),
            color=alt.Color(
                "kategorija:N",
                scale=alt.Scale(domain=domena_kat, range=list(BOJE_KATEGORIJA.values())),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip("znacajka:N", title=p.t("graf_pct_tt_statistika", jezik)),
                alt.Tooltip("vrijednost:Q", title=p.t("graf_pct_tt_per90", jezik), format=".2f"),
                alt.Tooltip("percentil:Q", title=p.t("graf_pct_tt_percentil", jezik), format=".0f"),
                alt.Tooltip("kategorija:N", title=p.t("graf_pct_tt_obitelj", jezik)),
            ],
        )
    )

    medijan = (
        alt.Chart(pd.DataFrame({"x": [50]}))
        .mark_rule(color=PRIGUSENO, strokeDash=[4, 4])
        .encode(x="x:Q")
    )

    return primijeni_temu((trake + medijan).properties(height=max(320, 19 * len(df))))


def graf_mape(
    prostor: s.ProstorIgraca,
    idx: int,
    koordinate: np.ndarray,
    slicni: list[str],
    metoda: str,
    jezik: str = "hr",
) -> alt.Chart:
    """Mapa sličnosti: svi igrači u 2D, s istaknutim ciljem i njegovim susjedima.

    Ovo je jedini graf koji pokazuje sam PROSTOR u kojem model traži sličnost —
    pa se vidi zašto su ciljevi susjedi to što jesu.

    Interni nazivi stupaca ("player"/"role"/"league") ostaju fiksni engleski
    Vega-referenci bez obzira na jezik prikaza; prevedeni naslovi idu preko
    `alt.Tooltip(title=...)`, tako da promjena jezika ne dira kodiranja polja.
    """
    uloge_prevedene = [p.prevedi_ulogu(u, jezik) for u in prostor.df["ULOGA"]]
    df = pd.DataFrame(
        {
            "x": koordinate[:, 0],
            "y": koordinate[:, 1],
            "player": prostor.df["NAME"].to_numpy(),
            "role": uloge_prevedene,
            "league": prostor.df["LEAGUE"].to_numpy()
            if prostor.ima_lige
            else "—",
        }
    )

    cilj_ime = prostor.df.iloc[idx]["NAME"]
    df_ostali = df.drop(index=idx)
    df_susjedi = df[df["player"].isin(slicni)]
    df_cilj = df.iloc[[idx]]

    opisi = [
        alt.Tooltip("player:N", title=p.t("word_igrac", jezik)),
        alt.Tooltip("role:N", title=p.t("word_uloga", jezik)),
        alt.Tooltip("league:N", title=p.t("word_liga", jezik)),
    ]

    oblak = (
        alt.Chart(df_ostali)
        .mark_circle(size=22, opacity=0.30)
        .encode(
            x=alt.X("x:Q", title=None, axis=alt.Axis(labels=False, ticks=False)),
            y=alt.Y("y:Q", title=None, axis=alt.Axis(labels=False, ticks=False)),
            color=alt.Color("role:N", legend=alt.Legend(title=None, orient="top", columns=2)),
            tooltip=opisi,
        )
    )

    susjedi = (
        alt.Chart(df_susjedi)
        .mark_point(size=110, filled=True, opacity=0.95, stroke=TEKST_JAK, strokeWidth=1)
        .encode(x="x:Q", y="y:Q", color=alt.value("#C9A227"), tooltip=opisi)
    )

    tocka_cilja = (
        alt.Chart(df_cilj)
        .mark_point(size=300, filled=True, stroke=TEKST_JAK, strokeWidth=2)
        .encode(x="x:Q", y="y:Q", color=alt.value(AKCENT), tooltip=opisi)
    )

    natpis = (
        alt.Chart(df_cilj)
        .mark_text(dy=-20, color=TEKST_JAK, fontSize=13, fontWeight="bold")
        .encode(x="x:Q", y="y:Q", text=alt.value(cilj_ime))
    )

    graf = (
        (oblak + susjedi + tocka_cilja + natpis)
        .properties(height=520, title=p.t("graf_mapa_naslov", jezik, metoda=metoda.upper()))
        .interactive()
        .resolve_scale(color="independent")
    )
    return primijeni_temu(graf).configure_title(color=TEKST, fontSize=12, anchor="start")


def graf_jedinstvenosti(
    rezultati: np.ndarray, idx: int, prag_top: float, mjera: str, jezik: str = "hr"
) -> alt.Chart:
    """Histogram sličnosti svih ostalih igrača prema cilju.

    Uska raspodjela stisnuta uz desni rub znači gomilu bliskih dvojnika; dugačak
    rep prema lijevo znači rijedak profil.
    """
    vrijednosti = np.delete(rezultati, idx)
    vrijednosti = vrijednosti[np.isfinite(vrijednosti)]
    df = pd.DataFrame({"slicnost": vrijednosti})

    histogram = (
        alt.Chart(df)
        .mark_bar(color=AKCENT, opacity=0.85)
        .encode(
            x=alt.X("slicnost:Q", bin=alt.Bin(maxbins=45), title=p.t("graf_jed_x_title", jezik, mjera=mjera)),
            y=alt.Y("count():Q", title=p.t("graf_jed_y_title", jezik)),
            tooltip=[alt.Tooltip("count():Q", title=p.t("graf_jed_tt_igraca", jezik))],
        )
    )

    granica = (
        alt.Chart(pd.DataFrame({"x": [prag_top]}))
        .mark_rule(color=TEKST_JAK, strokeDash=[5, 4], size=1.5)
        .encode(x="x:Q", tooltip=alt.value(p.t("graf_jed_tt_prag", jezik)))
    )

    return primijeni_temu((histogram + granica).properties(height=300))
