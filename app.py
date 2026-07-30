"""Streamlit aplikacija — tražilica sličnih nogometnih igrača.

Sučelje je tanki sloj oko `similarity.py`: sva matematika (standardizacija,
uloge, objedinjeni model sličnosti) živi ondje i dijeli je i notebook i ova
aplikacija, pa njihovi rezultati ne mogu razići se.

Pokretanje lokalno:
    streamlit run app.py
"""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import analitika as a
import similarity as s

# "?app=1" razlikuje početnu (marketinšku) stranicu od stvarnog alata, tako da
# oba CTA gumba na landing.html mogu voditi ovamo običnim <a href="?app=1">.
JE_POCETNA = st.query_params.get("app") != "1"

st.set_page_config(
    page_title="Podudarnost — tražilica sličnih igrača",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed" if JE_POCETNA else "expanded",
)

if JE_POCETNA:
    st.markdown(
        """
        <style>
        .block-container{padding:0 !important;max-width:100% !important;}
        header[data-testid="stHeader"]{background:transparent;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    landing_html = (Path(__file__).parent / "landing.html").read_text(encoding="utf-8")
    # st.markdown() koristi Python-Markdown ispod haube: prazan redak usred bloka
    # sirovog HTML-a prekida "raw HTML" način rada, pa sve nakon njega završi kao
    # escapean tekst umjesto pravih oznaka. Uklanjamo prazne retke da to spriječimo.
    landing_html = "\n".join(line for line in landing_html.splitlines() if line.strip())
    st.markdown(landing_html, unsafe_allow_html=True)
    st.stop()

# ---------------------------------------------------------------------------
# Izgled — usklađeno s odobrenom maketom (claret akcent, serif za istaknuta
# imena, tabelarni brojevi). Streamlitova vlastita tema (.streamlit/config.toml)
# postavlja iste boje za ugrađene widgete; ovaj CSS pokriva samo dijelove koje
# iscrtavamo ručno (profil, redci rezultata, panel usporedbe).
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
:root{
  --ink:#E9EBEE; --ink-2:#9AA2AC; --ink-3:#6E7681;
  --line:#262C34; --surface:#161A1F; --surface-2:#1B2027;
  --accent:#D9637A; --accent-wash:#2A171C; --track:#232931;
  --pos:#D9637A; --neg:#7E8792;
  --font-display:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
  --font-mono:ui-monospace,"Cascadia Mono","Segoe UI Mono",Consolas,monospace;
}
.target-name{font-family:var(--font-display);font-size:32px;font-weight:600;
  letter-spacing:-0.01em;color:var(--ink);}
.chip{display:inline-block;font-size:12px;padding:3px 10px;border-radius:100px;
  background:var(--accent-wash);color:var(--accent);
  border:1px solid rgba(217,99,122,.30);white-space:nowrap;}
.target-meta{font-size:13px;color:var(--ink-3);font-variant-numeric:tabular-nums;}
.section-label{font-size:11px;letter-spacing:.1em;text-transform:uppercase;
  color:var(--ink-3);font-weight:600;margin:2px 0 6px;}
.prow{display:grid;grid-template-columns:180px 1fr 52px;align-items:center;
  gap:12px;font-size:12.5px;padding:2px 0;}
.prow .pname{color:var(--ink-2);text-align:right;}
.prow .pval{font-family:var(--font-mono);font-size:11.5px;color:var(--ink-3);
  font-variant-numeric:tabular-nums;}
.axis{position:relative;height:13px;background:var(--track);border-radius:2px;}
.axis::before{content:"";position:absolute;left:50%;top:-2px;bottom:-2px;
  width:1px;background:var(--line);}
.axis span{position:absolute;top:2px;bottom:2px;border-radius:1px;}
.row{display:grid;
  grid-template-columns:26px minmax(130px,1.1fr) minmax(0,1.1fr) 92px 56px 100px;
  align-items:center;gap:12px;padding:8px 4px;border-bottom:1px solid var(--line);
  font-size:13.5px;}
.row:hover{background:var(--surface-2);}
.rank{font-family:var(--font-mono);font-size:11.5px;color:var(--ink-3);
  font-variant-numeric:tabular-nums;}
.rname{font-weight:500;color:var(--ink);}
.rrole,.rliga{font-size:11.5px;color:var(--ink-3);overflow:hidden;
  text-overflow:ellipsis;white-space:nowrap;}
.rmin{font-family:var(--font-mono);font-size:11.5px;color:var(--ink-3);
  font-variant-numeric:tabular-nums;text-align:right;}
.meter{height:6px;background:var(--track);border-radius:3px;overflow:hidden;}
.meter i{display:block;height:100%;background:var(--accent);border-radius:3px;}
.rscore{font-family:var(--font-mono);font-size:12.5px;
  font-variant-numeric:tabular-nums;text-align:right;color:var(--ink);}
.row-head{display:grid;
  grid-template-columns:26px minmax(130px,1.1fr) minmax(0,1.1fr) 92px 56px 100px;
  gap:12px;padding:0 4px 6px;font-size:11px;letter-spacing:.08em;
  text-transform:uppercase;color:var(--ink-3);font-weight:600;
  border-bottom:1px solid var(--line);}
.flag{border-left:3px solid var(--accent);background:var(--accent-wash);
  padding:10px 14px;font-size:13.5px;color:var(--ink-2);border-radius:0 4px 4px 0;
  margin-bottom:6px;}
.flag b{color:var(--ink);}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Podaci (teško za izračunati, pa se cachira kao jedan dijeljeni resurs)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Učitavanje igrača i računanje uloga…")
def _ucitaj() -> s.ProstorIgraca:
    return s.ucitaj_prostor()


prostor = _ucitaj()


# ---------------------------------------------------------------------------
# Analitika (stranica igrača) — cachirano jer je 2D ugradnja i matrica
# sličnosti isti izračun za SVAKOG igrača; mijenja se samo tko je istaknut.
# Vodeća podvlaka u imenu argumenta govori Streamlitu da preskoči hashiranje
# `prostor` objekta (sadrži numpy nizove, skupo za hashiranje svaki rerun).
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Računam percentile…")
def _cache_percentili(_prostor: s.ProstorIgraca) -> np.ndarray:
    return a.percentili(_prostor)


@st.cache_data(show_spinner="Računam percentile unutar uloga…")
def _cache_percentili_uloga(_prostor: s.ProstorIgraca) -> np.ndarray:
    return a.percentili_po_ulozi(_prostor)


@st.cache_data(show_spinner="Računam 2D projekciju (t-SNE traje malo dulje)…")
def _cache_ugradnja(_prostor: s.ProstorIgraca, metoda: str) -> np.ndarray:
    return a.ugradnja_2d(_prostor, metoda)


@st.cache_data(show_spinner="Računam matricu sličnosti…")
def _cache_matrica(_prostor: s.ProstorIgraca, alpha: float, koristi_M: bool) -> np.ndarray:
    return a.matrica_slicnosti(_prostor, alpha=alpha, koristi_M=koristi_M)


def _pronadi_tocno(prostor: s.ProstorIgraca, ime: str) -> int | None:
    """Točno (ne fuzzy) podudaranje imena — sigurno za razrješavanje URL-a.

    `similarity.pronadi_igraca` namjerno dopušta djelomično podudaranje radi
    pretrage; ovdje to NIJE poželjno jer bi URL mogao razriješiti pogrešnog
    igrača (npr. ?igrac=Nico bi trebao dati grešku, ne prvog 'Nico *').
    """
    poklapanja = prostor.df.index[prostor.df["NAME"] == ime]
    return int(poklapanja[0]) if len(poklapanja) else None

METODA_OBJASNJENJA = {
    "soft_cosine": (
        "Zadano. Traži igrače sličnog STILA igre, a pritom uvažava da su "
        "neke statistike prirodno povezane (npr. tko puno dodaje, obično "
        "puno dodaje i u završnu trećinu) pa ih ne broji dvaput. "
        "Najuravnoteženiji izbor za većinu upita."
    ),
    "cosine": (
        "Traži igrače sličnog STILA igre — gleda samo 'oblik' profila "
        "(u čemu je igrač relativno jak ili slab), a ne koliko je toga "
        "ukupno odradio. Dobro za usporedbu igrača s različitom minutažom."
    ),
    "euclidean": (
        "Traži igrače koji su najbliži cilju u svemu odjednom — i po "
        "stilu i po razini doprinosa. Stroža mjera: kažnjava svaku "
        "razliku, veliku ili malu."
    ),
    "dot_product": (
        "Traži igrače koji rade ISTE stvari u SLIČNOJ KOLIČINI (npr. "
        "slično puno golova i dodavanja). Favorizira igrače s visokom "
        "minutažom — dobro za 'jednako produktivnu' zamjenu, manje za "
        "sličan stil."
    ),
}

ALPHA_OPCIJE = [0.0, 0.25, 0.5, 0.75, 1.0]


# ---------------------------------------------------------------------------
# Stanje sučelja
# ---------------------------------------------------------------------------
def _init_stanje() -> None:
    st.session_state.setdefault("metoda", "soft_cosine")
    st.session_state.setdefault("alpha", s.PRESETI["soft_cosine"]["alpha"])
    st.session_state.setdefault("koristi_M", s.PRESETI["soft_cosine"]["koristi_M"])
    st.session_state.setdefault("prilagodjeno", False)
    st.session_state.setdefault("usporedba_s", None)


def _odaberi_metodu() -> None:
    k = st.session_state["metoda_radio"]
    st.session_state["metoda"] = k
    st.session_state["alpha"] = s.PRESETI[k]["alpha"]
    st.session_state["koristi_M"] = s.PRESETI[k]["koristi_M"]
    st.session_state["prilagodjeno"] = False


def _dodirnuo_napredno() -> None:
    st.session_state["alpha"] = st.session_state["alpha_slider"]
    st.session_state["koristi_M"] = st.session_state["matM_checkbox"]
    st.session_state["prilagodjeno"] = True


_init_stanje()


# ---------------------------------------------------------------------------
# Stranica pojedinog igrača — puna statistika + analitički grafovi na zahtjev
# ---------------------------------------------------------------------------
def _prikazi_zaglavlje_igraca(cilj: pd.Series, prostor: s.ProstorIgraca) -> None:
    top1, top2, top3 = st.columns([3, 1, 2])
    with top1:
        st.markdown(f'<div class="target-name">{html.escape(cilj["NAME"])}</div>', unsafe_allow_html=True)
    with top2:
        st.markdown(
            f'<div style="padding-top:10px"><span class="chip">{html.escape(cilj["ULOGA"])}</span></div>',
            unsafe_allow_html=True,
        )
    with top3:
        liga = f'{cilj["LEAGUE"]} · ' if prostor.ima_lige else ""
        st.markdown(
            f'<div style="padding-top:14px" class="target-meta">'
            f'{liga}{int(cilj["APPS"])} nastupa · {int(cilj["MINS"]):,} minuta</div>'.replace(",", "."),
            unsafe_allow_html=True,
        )


def _stranica_igraca(prostor: s.ProstorIgraca, idx: int) -> None:
    cilj = prostor.df.iloc[idx]
    ime = cilj["NAME"]

    if st.button("← Natrag na pretragu"):
        del st.query_params["igrac"]
        st.rerun()

    _prikazi_zaglavlje_igraca(cilj, prostor)
    st.divider()

    # --- puna tablica statistika -------------------------------------------------
    st.markdown('<div class="section-label">Sve statistike</div>', unsafe_allow_html=True)
    naziv_uloge = cilj["ULOGA"]
    osnovica = st.radio(
        "Percentil u odnosu na:",
        ["Sve igrače", f"Istu ulogu ({naziv_uloge})"],
        horizontal=True,
        key="percentil_osnovica",
        help=(
            "Postotak dodavanja stopera znači nešto drugo naspram napadača nego "
            "naspram drugih stopera — obje osnovice su korisne."
        ),
    )
    prema_ulozi = osnovica.startswith("Istu")
    pct = _cache_percentili_uloga(prostor) if prema_ulozi else _cache_percentili(prostor)
    osnovica_txt = f"uloga: {naziv_uloge}" if prema_ulozi else "svi igrači"

    tablica = pd.DataFrame(
        {
            "Statistika": [s.NAZIVI.get(c, c) for c in prostor.znacajke],
            "Obitelj": [a.kategorija(c) for c in prostor.znacajke],
            "Per 90": [float(cilj[c]) for c in prostor.znacajke],
            "Z-vrijednost": prostor.X[idx].round(2),
            "Percentil": pct[idx].round(1),
        }
    ).sort_values("Percentil", ascending=False, ignore_index=True)

    st.dataframe(
        tablica,
        hide_index=True,
        width="stretch",
        column_config={
            "Percentil": st.column_config.ProgressColumn(
                "Percentil", min_value=0, max_value=100, format="%.0f"
            ),
            "Per 90": st.column_config.NumberColumn("Per 90", format="%.2f"),
            "Z-vrijednost": st.column_config.NumberColumn("Z-vrijednost", format="%+.2f"),
        },
    )

    st.divider()

    # --- grafovi na zahtjev -------------------------------------------------------
    naziv_metode = s.PRESETI[st.session_state["metoda"]]["naziv"] if not st.session_state["prilagodjeno"] else f"prilagođeno (α={st.session_state['alpha']:.2f})"
    st.markdown('<div class="section-label">Analitički grafovi</div>', unsafe_allow_html=True)
    st.caption(
        f"Grafovi sličnosti ispod koriste trenutno odabranu metodu na stranici za "
        f"pretragu: **{naziv_metode}**."
    )

    # 1) Profil percentila
    if st.button("📊 Generiraj profil percentila", key="btn_pct"):
        st.session_state["prikazi_graf_percentila"] = True
    if st.session_state.get("prikazi_graf_percentila"):
        st.altair_chart(
            a.graf_percentila(prostor, idx, pct, osnovica_txt), use_container_width=True, theme=None
        )
        st.caption(
            "Percentil po svakoj od 29 statistika, grupirano po obitelji. "
            "Isprekidana crta je medijan (50. percentil)."
        )

    # zajednički izračun za preostala dva grafa (sličnost prema svim igračima)
    S = _cache_matrica(prostor, st.session_state["alpha"], st.session_state["koristi_M"])
    top10 = s.slicni_igraci(
        prostor, idx, preset=None, alpha=st.session_state["alpha"],
        koristi_M=st.session_state["koristi_M"], n=10,
    )

    # 2) Mapa sličnosti
    if st.button("🗺️ Generiraj mapu sličnosti", key="btn_mapa"):
        st.session_state["prikazi_graf_mape"] = True
    if st.session_state.get("prikazi_graf_mape"):
        metoda_2d = st.radio(
            "2D metoda",
            ["pca", "tsne"],
            format_func=lambda m: "PCA (brzo)" if m == "pca" else "t-SNE (sporije, jasniji klasteri)",
            horizontal=True,
            key="mapa_metoda",
        )
        koordinate = _cache_ugradnja(prostor, metoda_2d)
        st.altair_chart(
            a.graf_mape(prostor, idx, koordinate, top10["NAME"].tolist(), metoda_2d),
            use_container_width=True,
            theme=None,
        )
        st.caption(
            "Svih 1828 igrača u 2D prostoru u kojem model traži sličnost. "
            "Veća claret točka je cilj, žute točke su njegovih 10 najsličnijih."
        )

    # 3) Jedinstvenost
    if st.button("🎯 Generiraj graf jedinstvenosti", key="btn_jed"):
        st.session_state["prikazi_graf_jedinstvenosti"] = True
    if st.session_state.get("prikazi_graf_jedinstvenosti"):
        rezultati_svih, prosjek_top10, jedinstvenost = a.jedinstvenost(S, idx)
        st.metric("Jedinstvenost", f"{jedinstvenost:.0f} / 100")
        st.altair_chart(
            a.graf_jedinstvenosti(rezultati_svih, idx, prosjek_top10, naziv_metode.lower()),
            use_container_width=True,
            theme=None,
        )
        st.caption(
            "Raspodjela sličnosti svih ostalih igrača prema cilju. Što je crta (prag top-10) "
            "dalje od gomile, to je cilj rjeđi profil — 100 = najrjeđi u skupu, 0 = najzamjenjiviji."
        )

    st.divider()
    st.caption(
        "Podaci: https://theanalyst.com/, sezona 2025/26. Uloge su izvedene "
        "K-Means klasteriranjem statistika."
    )


_ime_iz_urla = st.query_params.get("igrac")
if _ime_iz_urla:
    _idx_urla = _pronadi_tocno(prostor, _ime_iz_urla)
    if _idx_urla is None:
        st.error(f"Igrač „{_ime_iz_urla}” nije pronađen u skupu podataka.")
        if st.button("← Natrag na pretragu"):
            del st.query_params["igrac"]
            st.rerun()
    else:
        _stranica_igraca(prostor, _idx_urla)
    st.stop()

# ---------------------------------------------------------------------------
# Bočna traka — pretraga, metoda, napredne postavke, filtri
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚽ Podudarnost")
    st.caption(f"{len(prostor.df)} igrača · top-5 europskih liga")

    imena = sorted(prostor.imena)
    zadano = "Bruno Fernandes" if "Bruno Fernandes" in imena else imena[0]
    igrac = st.selectbox(
        "Igrač",
        options=imena,
        index=imena.index(zadano),
        help="Upiši dio imena za pretragu — lista se filtrira dok tipkaš.",
    )

    st.markdown("**Metoda**")
    metoda = st.radio(
        "Metoda",
        options=list(s.PRESETI.keys()),
        format_func=lambda k: s.PRESETI[k]["naziv"],
        key="metoda_radio",
        index=list(s.PRESETI.keys()).index(st.session_state["metoda"]),
        on_change=_odaberi_metodu,
        label_visibility="collapsed",
    )
    st.caption(s.PRESETI[st.session_state["metoda"]]["opis"])

    with st.expander("Što znače metode?"):
        for k, spec in s.PRESETI.items():
            st.markdown(f"**{spec['naziv']}**")
            st.caption(METODA_OBJASNJENJA[k])

    with st.expander("Napredno", expanded=st.session_state["prilagodjeno"]):
        st.select_slider(
            "Stil ↔ količina",
            options=ALPHA_OPCIJE,
            value=st.session_state["alpha"],
            format_func=lambda v: f"{v:.2f}",
            key="alpha_slider",
            on_change=_dodirnuo_napredno,
            help=(
                "Klizač bira što 'sličnost' znači. Skroz desno ('stil'): "
                "traži igrače koji igraju na isti način, bez obzira igraju "
                "li malo ili puno. Skroz lijevo ('količina'): traži igrače "
                "koji ostvaruju sličan BROJ istih akcija — golova, "
                "dodavanja — čak i ako im je stil drugačiji. Sredina je "
                "mješavina to dvoje."
            ),
        )
        st.checkbox(
            "Uvaži povezanost statistika",
            value=st.session_state["koristi_M"],
            key="matM_checkbox",
            on_change=_dodirnuo_napredno,
            help=(
                "Kad je uključeno, model zna da su neke statistike "
                "prirodno povezane (npr. dodavanja i dodavanja u završnu "
                "trećinu) pa ih ne tretira kao potpuno nezavisne. Obično "
                "daje profinjenije rezultate."
            ),
        )
        if st.session_state["prilagodjeno"]:
            st.caption("Prilagođeno — više ne prati odabranu metodu iznad.")

    st.markdown("**Filtri**")
    ista_uloga = st.checkbox("Samo ista uloga", value=False, key="ista_uloga_checkbox")
    izbaci_ligu = st.checkbox("Izbaci istu ligu", value=False, key="izbaci_ligu_checkbox")
    min_minuta = st.slider(
        "Najmanje minuta", min_value=450, max_value=3420, value=450, step=90,
        key="min_minuta_slider",
    )
    n_rezultata = st.slider(
        "Broj rezultata", min_value=5, max_value=25, value=10, key="n_rezultata_slider"
    )

# ---------------------------------------------------------------------------
# Izračun
# ---------------------------------------------------------------------------
idx = s.pronadi_igraca(prostor, igrac)[0]
cilj = prostor.df.iloc[idx]

rezultati = s.slicni_igraci(
    prostor,
    idx,
    preset=None,
    alpha=st.session_state["alpha"],
    koristi_M=st.session_state["koristi_M"],
    n=n_rezultata,
    ista_uloga=ista_uloga,
    izbaci_istu_ligu=izbaci_ligu,
    min_minuta=min_minuta,
)

if st.session_state["usporedba_s"] not in set(rezultati["NAME"]):
    st.session_state["usporedba_s"] = (
        rezultati.iloc[0]["NAME"] if len(rezultati) else None
    )

# ---------------------------------------------------------------------------
# Glavni prostor — cilj i njegov profil
# ---------------------------------------------------------------------------
top1, top2, top3, top4 = st.columns([3, 1, 1.6, 1])
with top1:
    st.markdown(f'<div class="target-name">{html.escape(cilj["NAME"])}</div>', unsafe_allow_html=True)
with top2:
    st.markdown(
        f'<div style="padding-top:10px"><span class="chip">{html.escape(cilj["ULOGA"])}</span></div>',
        unsafe_allow_html=True,
    )
with top3:
    liga = f'{cilj["LEAGUE"]} · ' if prostor.ima_lige else ""
    st.markdown(
        f'<div style="padding-top:14px" class="target-meta">'
        f'{liga}{int(cilj["APPS"])} nastupa · {int(cilj["MINS"]):,} minuta</div>'
        .replace(",", "."),
        unsafe_allow_html=True,
    )
with top4:
    if st.button("Profil i grafovi →", key="prof_cilj"):
        st.query_params["igrac"] = cilj["NAME"]
        st.rerun()

st.markdown('<div class="section-label">Profil — najizraženije značajke</div>', unsafe_allow_html=True)

z_ciljni = prostor.X[idx]
najistaknutije = np.argsort(-np.abs(z_ciljni))[:6]
redci = []
for j in najistaknutije:
    z = float(z_ciljni[j])
    naziv = s.NAZIVI.get(prostor.znacajke[j], prostor.znacajke[j])
    w = min(abs(z) / 6, 1) * 50
    stil = f"left:50%;width:{w}%;background:var(--pos);" if z >= 0 else f"right:50%;width:{w}%;background:var(--neg);"
    predznak = "+" if z >= 0 else ""
    redci.append(
        f'<div class="prow"><span class="pname">{html.escape(naziv)}</span>'
        f'<span class="axis"><span style="{stil}"></span></span>'
        f'<span class="pval">{predznak}{z:.2f}</span></div>'
    )
st.markdown("".join(redci), unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------------------
# Rezultati
# ---------------------------------------------------------------------------
naziv_nacina = (
    f"prilagođeno · α = {st.session_state['alpha']:.2f}"
    if st.session_state["prilagodjeno"]
    else s.PRESETI[st.session_state["metoda"]]["naziv"].lower()
)
st.markdown(f"**{len(rezultati)} najsličnijih igrača** &nbsp;·&nbsp; {naziv_nacina}")

if len(rezultati) == 0:
    st.info("Nijedan igrač ne zadovoljava odabrane filtre.")
else:
    maks = float(rezultati["SLICNOST"].max()) or 1.0
    st.markdown(
        '<div class="row-head"><span></span><span>Igrač</span><span>Uloga</span>'
        '<span>Liga</span><span>Min.</span><span>Sličnost</span></div>',
        unsafe_allow_html=True,
    )
    for i, r in rezultati.iterrows():
        sirina = max(4, (r["SLICNOST"] / maks) * 100)
        liga_txt = html.escape(r["LEAGUE"]) if prostor.ima_lige else ""
        cols = st.columns([0.04, 0.30, 1, 0.13, 0.13], gap="small")
        with cols[0]:
            st.markdown(
                f'<div class="row" style="grid-template-columns:26px 1fr;border-bottom:none;padding:8px 0">'
                f'<span class="rank">{i + 1}</span><span></span></div>',
                unsafe_allow_html=True,
            )
        with cols[1]:
            st.markdown(
                f'<div style="padding:8px 0"><span class="rname">{html.escape(r["NAME"])}</span><br>'
                f'<span class="rrole">{html.escape(r["ULOGA"])}</span></div>',
                unsafe_allow_html=True,
            )
        with cols[2]:
            st.markdown(
                f'<div style="padding:8px 0;display:grid;grid-template-columns:1fr 70px 60px 90px;'
                f'gap:12px;align-items:center">'
                f'<span class="rliga">{liga_txt}</span>'
                f'<span class="rmin">{int(r["MINS"]):,}′</span>'.replace(",", ".") +
                f'<span class="meter"><i style="width:{sirina}%"></i></span>'
                f'<span class="rscore">{r["SLICNOST"]:.1f} %</span></div>',
                unsafe_allow_html=True,
            )
        with cols[3]:
            if st.button("Usporedi ↓", key=f"cmp_{i}_{r['NAME']}"):
                st.session_state["usporedba_s"] = r["NAME"]
        with cols[4]:
            if st.button("Profil →", key=f"prof_{i}_{r['NAME']}"):
                st.query_params["igrac"] = r["NAME"]
                st.rerun()

# ---------------------------------------------------------------------------
# Panel usporedbe — zašto su dva igrača slična/različita
# ---------------------------------------------------------------------------
if st.session_state["usporedba_s"] and len(rezultati):
    st.divider()
    drugi_idx = s.pronadi_igraca(prostor, st.session_state["usporedba_s"])[0]
    razlika = s.usporedi_profile(prostor, idx, drugi_idx)

    st.markdown(
        f"**{html.escape(cilj['NAME'])}** &nbsp;↔&nbsp; "
        f"**{html.escape(st.session_state['usporedba_s'])}**"
    )
    st.caption("Standardizirane (z) vrijednosti — 0 je prosjek lige na toj statistici.")

    lijevo, desno = st.columns(2)

    def _crtaj_razlaganje(naslov: str, podskup) -> str:
        redci = [f'<div class="section-label">{naslov}</div>']
        for _, r in podskup.iterrows():
            za, zb = float(r["z_a"]), float(r["z_b"])
            wa = min(abs(za) / 6, 1) * 50
            wb = min(abs(zb) / 6, 1) * 50
            stil_a = f"left:50%;width:{wa}%;background:var(--pos);" if za >= 0 else f"right:50%;width:{wa}%;background:var(--pos);"
            stil_b = f"left:50%;width:{wb}%;background:var(--neg);" if zb >= 0 else f"right:50%;width:{wb}%;background:var(--neg);"
            redci.append(
                f'<div class="prow"><span class="pname">{html.escape(r["znacajka"])}</span>'
                f'<span class="axis"><span style="{stil_a}"></span></span>'
                f'<span class="pval">{za:+.2f}</span></div>'
                f'<div class="prow"><span class="pname"></span>'
                f'<span class="axis"><span style="{stil_b}"></span></span>'
                f'<span class="pval">{zb:+.2f}</span></div>'
            )
        return "".join(redci)

    with lijevo:
        st.markdown(_crtaj_razlaganje("Najveće razlike", razlika.head(4)), unsafe_allow_html=True)
    with desno:
        st.markdown(_crtaj_razlaganje("Najbliže podudaranje", razlika.tail(4)), unsafe_allow_html=True)

st.divider()
st.caption(
    "Podaci: https://theanalyst.com/, sezona 2025/26 (podaci/top5_stats_combined.csv). "
    "Uloge su izvedene K-Means klasteriranjem statistika — skup podataka nema stupac s "
    "pozicijom, pa 'uloga' opisuje stil igre, ne službenu poziciju."
)
