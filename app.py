"""Streamlit aplikacija — tražilica sličnih nogometnih igrača.

Sučelje je tanki sloj oko `similarity.py`: sva matematika (standardizacija,
uloge, objedinjeni model sličnosti) živi ondje i dijeli je i notebook i ova
aplikacija, pa njihovi rezultati ne mogu razići se. Prikaz je dvojezičan
(hr/en) preko `prijevodi.py`; `similarity.py` ostaje isključivo hrvatski jer
ga dijeli i notebook.

Pokretanje lokalno:
    streamlit run app.py
"""

from __future__ import annotations

import html
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import streamlit as st

import analitika as a
import prijevodi as p
import similarity as s

# "?app=1" razlikuje početnu (marketinšku) stranicu od stvarnog alata, tako da
# oba CTA gumba na landing.html mogu voditi ovamo običnim <a href="?app=1">.
JE_POCETNA = st.query_params.get("app") != "1"

JEZIK = st.query_params.get("lang", p.ZADANI_JEZIK)
if JEZIK not in p.JEZICI:
    JEZIK = p.ZADANI_JEZIK


def _url_s_parametrima(**zamjene: str) -> str:
    """URL s trenutnim query parametrima, uz navedene zamjene/dodatke.

    Koristi se za sve navigacijske poveznice (jezik, natrag, profil) tako da
    se pritiskom na njih NE izgube ostali parametri (npr. ?app=1&igrac=...).
    """
    trenutni = dict(st.query_params)
    trenutni.update(zamjene)
    trenutni = {k: v for k, v in trenutni.items() if v not in (None, "")}
    return "?" + urlencode(trenutni)


def _jezik_toggle_html() -> str:
    aktivan_hr = "active" if JEZIK == "hr" else ""
    aktivan_en = "active" if JEZIK == "en" else ""
    return (
        '<div class="lang-toggle">'
        f'<a href="{html.escape(_url_s_parametrima(lang="hr"))}" class="lang-link {aktivan_hr}">HR</a>'
        '<span class="lang-sep">·</span>'
        f'<a href="{html.escape(_url_s_parametrima(lang="en"))}" class="lang-link {aktivan_en}">EN</a>'
        "</div>"
    )


def _render_landing(jezik: str) -> str:
    """Učitaj landing.html i popuni {{PLACEHOLDER}} tokene prijevodima."""
    tekst = (Path(__file__).parent / "landing.html").read_text(encoding="utf-8")

    zamjene = {
        "LANG_TOGGLE": _jezik_toggle_html(),
        "URL_APP": html.escape(_url_s_parametrima(app="1")),
        "LANDING_EYEBROW": p.t("landing_eyebrow", jezik),
        "LANDING_H1": p.t("landing_h1", jezik),
        "LANDING_LEAD": p.t("landing_lead", jezik),
        "LANDING_CTA_PRIMARY": p.t("landing_cta_primary", jezik),
        "LANDING_CTA_SECONDARY": p.t("landing_cta_secondary", jezik),
        "LANDING_EXAMPLE_EYEBROW": p.t("landing_example_eyebrow", jezik),
        "LANDING_EXAMPLE_H2": p.t("landing_example_h2", jezik),
        "LANDING_EXAMPLE_LEAD": p.t("landing_example_lead", jezik),
        "LABEL_PROFIL_ZNACAJKE": p.t("label_profil_znacajke", jezik),
        "LANDING_ROLE_PLAYMAKER": p.t("landing_role_playmaker", jezik),
        "LANDING_ROLE_ORGANIZATOR": p.t("landing_role_organizator", jezik),
        "LANDING_ROLE_BOX2BOX": p.t("landing_role_box2box", jezik),
        "LANDING_EXAMPLE_META": p.t("landing_example_meta", jezik),
        "ZNACAJKA_PAS_TOTAL_FINAL_THIRD": p.znacajka("PAS_TOTAL (FINAL THIRD PASSES)", jezik),
        "ZNACAJKA_CAR_ENDED_CHANCE": p.znacajka("CAR_ENDED WITH CHANCE", jezik),
        "ZNACAJKA_PAS_TOTAL_OPEN_PLAY": p.znacajka("PAS_TOTAL (OPEN PLAY PASSES)", jezik),
        "ZNACAJKA_DEF_TACKLES": p.znacajka("DEF_TACKLES", jezik),
        "ZNACAJKA_DEF_TOTAL_AERIAL": p.znacajka("DEF_TOTAL (AERIAL DUELS)", jezik),
        "LANDING_SIMILAR_HEADER": p.t("landing_similar_header", jezik),
        "LANDING_SIMILAR_METHOD": p.t("landing_similar_method", jezik),
        "WORD_IGRAC": p.t("word_igrac", jezik),
        "WORD_ULOGA": p.t("word_uloga", jezik),
        "WORD_LIGA": p.t("word_liga", jezik),
        "WORD_MIN": p.t("word_min", jezik),
        "WORD_SLICNOST": p.t("word_slicnost", jezik),
        "LANDING_ILLUSTRATIVE": p.t("landing_illustrative", jezik),
        "LANDING_FAQ_EYEBROW": p.t("landing_faq_eyebrow", jezik),
        "LANDING_FAQ_H2": p.t("landing_faq_h2", jezik),
        "LANDING_FAQ_Q1": p.t("landing_faq_q1", jezik),
        "LANDING_FAQ_A1": p.t("landing_faq_a1", jezik),
        "LANDING_FAQ_Q2": p.t("landing_faq_q2", jezik),
        "LANDING_FAQ_A2": p.t("landing_faq_a2", jezik),
        "LANDING_FAQ_Q3": p.t("landing_faq_q3", jezik),
        "LANDING_FAQ_A3": p.t("landing_faq_a3", jezik),
        "LANDING_FAQ_Q4": p.t("landing_faq_q4", jezik),
        "LANDING_FAQ_A4": p.t("landing_faq_a4", jezik),
        "LANDING_FAQ_Q5": p.t("landing_faq_q5", jezik),
        "LANDING_FAQ_A5": p.t("landing_faq_a5", jezik),
        "LANDING_CTA2_TITLE": p.t("landing_cta2_title", jezik),
        "LANDING_CTA2_LEAD": p.t("landing_cta2_lead", jezik),
        "LANDING_FOOTER": p.t("landing_footer", jezik),
    }
    for kljuc, vrijednost in zamjene.items():
        tekst = tekst.replace("{{" + kljuc + "}}", vrijednost)

    # st.markdown() koristi Python-Markdown ispod haube: prazan redak usred bloka
    # sirovog HTML-a prekida "raw HTML" način rada, pa sve nakon njega završi kao
    # escapean tekst umjesto pravih oznaka. Uklanjamo prazne retke da to spriječimo.
    return "\n".join(line for line in tekst.splitlines() if line.strip())


st.set_page_config(
    page_title=p.t("app_page_title", JEZIK),
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
    st.markdown(_render_landing(JEZIK), unsafe_allow_html=True)
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
.row-head-label{font-size:11px;letter-spacing:.08em;text-transform:uppercase;
  color:var(--ink-3);font-weight:600;padding-bottom:6px;
  border-bottom:1px solid var(--line);display:block;}
.flag{border-left:3px solid var(--accent);background:var(--accent-wash);
  padding:10px 14px;font-size:13.5px;color:var(--ink-2);border-radius:0 4px 4px 0;
  margin-bottom:6px;}
.flag b{color:var(--ink);}
.lang-toggle{display:flex;gap:6px;align-items:center;font-size:12px;
  font-weight:600;letter-spacing:.04em;}
.lang-link{color:var(--ink-3);text-decoration:none;padding:2px 4px;}
.lang-link:hover{color:var(--ink);}
.lang-link.active{color:var(--accent);}
.lang-sep{color:var(--ink-3);}
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


def _naziv_metode(jezik: str) -> str:
    """Ime trenutno odabrane metode za prikaz (prati preset ili prilagođeno)."""
    if st.session_state["prilagodjeno"]:
        return p.t("prilagodjeno_alpha", jezik, alpha=st.session_state["alpha"])
    return p.t(f"metoda_naziv_{st.session_state['metoda']}", jezik)


# ---------------------------------------------------------------------------
# Stranica pojedinog igrača — puna statistika + analitički grafovi na zahtjev
# ---------------------------------------------------------------------------
def _prikazi_zaglavlje_igraca(cilj: pd.Series, prostor: s.ProstorIgraca, jezik: str) -> None:
    top1, top2, top3 = st.columns([3, 1, 2])
    with top1:
        st.markdown(f'<div class="target-name">{html.escape(cilj["NAME"])}</div>', unsafe_allow_html=True)
    with top2:
        uloga_txt = html.escape(p.prevedi_ulogu(cilj["ULOGA"], jezik))
        st.markdown(
            f'<div style="padding-top:10px"><span class="chip">{uloga_txt}</span></div>',
            unsafe_allow_html=True,
        )
    with top3:
        liga = f'{cilj["LEAGUE"]} · ' if prostor.ima_lige else ""
        meta = p.t(
            "meta_nastupa_minuta", jezik,
            apps=int(cilj["APPS"]), mins=p.formatiraj_broj(int(cilj["MINS"]), jezik),
        )
        st.markdown(
            f'<div style="padding-top:14px" class="target-meta">{liga}{meta}</div>',
            unsafe_allow_html=True,
        )


def _stranica_igraca(prostor: s.ProstorIgraca, idx: int) -> None:
    cilj = prostor.df.iloc[idx]
    ime = cilj["NAME"]

    gore1, gore2 = st.columns([1, 20])
    with gore1:
        if st.button(p.t("btn_natrag", JEZIK)):
            del st.query_params["igrac"]
            st.rerun()
    with gore2:
        st.markdown(
            f'<div style="text-align:right">{_jezik_toggle_html()}</div>', unsafe_allow_html=True
        )

    _prikazi_zaglavlje_igraca(cilj, prostor, JEZIK)
    st.divider()

    # --- puna tablica statistika -------------------------------------------------
    st.markdown(f'<div class="section-label">{p.t("label_sve_statistike", JEZIK)}</div>', unsafe_allow_html=True)
    naziv_uloge = p.prevedi_ulogu(cilj["ULOGA"], JEZIK)
    osnovica = st.radio(
        p.t("label_percentil_odnos", JEZIK),
        [p.t("opcija_sve_igrace", JEZIK), p.t("opcija_istu_ulogu", JEZIK, uloga=naziv_uloge)],
        horizontal=True,
        key="percentil_osnovica",
        help=p.t("help_percentil_osnovica", JEZIK),
    )
    prema_ulozi = osnovica != p.t("opcija_sve_igrace", JEZIK)
    pct = _cache_percentili_uloga(prostor) if prema_ulozi else _cache_percentili(prostor)
    osnovica_txt = (
        p.t("osnovica_uloga", JEZIK, uloga=naziv_uloge) if prema_ulozi else p.t("osnovica_svi_igraci", JEZIK)
    )

    tablica = pd.DataFrame(
        {
            p.t("col_statistika", JEZIK): [p.znacajka(c, JEZIK) for c in prostor.znacajke],
            p.t("col_obitelj", JEZIK): [p.kategorija_naziv(a.kategorija(c), JEZIK) for c in prostor.znacajke],
            p.t("col_per90", JEZIK): [float(cilj[c]) for c in prostor.znacajke],
            p.t("col_zvrijednost", JEZIK): prostor.X[idx].round(2),
            p.t("col_percentil", JEZIK): pct[idx].round(1),
        }
    ).sort_values(p.t("col_percentil", JEZIK), ascending=False, ignore_index=True)

    st.dataframe(
        tablica,
        hide_index=True,
        width="stretch",
        column_config={
            p.t("col_percentil", JEZIK): st.column_config.ProgressColumn(
                p.t("col_percentil", JEZIK), min_value=0, max_value=100, format="%.0f"
            ),
            p.t("col_per90", JEZIK): st.column_config.NumberColumn(p.t("col_per90", JEZIK), format="%.2f"),
            p.t("col_zvrijednost", JEZIK): st.column_config.NumberColumn(
                p.t("col_zvrijednost", JEZIK), format="%+.2f"
            ),
        },
    )

    st.divider()

    # --- grafovi na zahtjev -------------------------------------------------------
    naziv_metode = _naziv_metode(JEZIK)
    st.markdown(f'<div class="section-label">{p.t("label_analiticki_grafovi", JEZIK)}</div>', unsafe_allow_html=True)
    st.caption(p.t("caption_grafovi_metoda", JEZIK, metoda=naziv_metode))

    # 1) Profil percentila
    if st.button(p.t("btn_generiraj_pct", JEZIK), key="btn_pct"):
        st.session_state["prikazi_graf_percentila"] = True
    if st.session_state.get("prikazi_graf_percentila"):
        st.altair_chart(
            a.graf_percentila(prostor, idx, pct, osnovica_txt, JEZIK), use_container_width=True, theme=None
        )
        st.caption(p.t("caption_graf_pct", JEZIK))

    # zajednički izračun za preostala dva grafa (sličnost prema svim igračima)
    S = _cache_matrica(prostor, st.session_state["alpha"], st.session_state["koristi_M"])
    top10 = s.slicni_igraci(
        prostor, idx, preset=None, alpha=st.session_state["alpha"],
        koristi_M=st.session_state["koristi_M"], n=10,
    )

    # 2) Mapa sličnosti
    if st.button(p.t("btn_generiraj_mapa", JEZIK), key="btn_mapa"):
        st.session_state["prikazi_graf_mape"] = True
    if st.session_state.get("prikazi_graf_mape"):
        metoda_2d = st.radio(
            p.t("label_2d_metoda", JEZIK),
            ["pca", "tsne"],
            format_func=lambda m: p.t("opcija_pca", JEZIK) if m == "pca" else p.t("opcija_tsne", JEZIK),
            horizontal=True,
            key="mapa_metoda",
        )
        koordinate = _cache_ugradnja(prostor, metoda_2d)
        st.altair_chart(
            a.graf_mape(prostor, idx, koordinate, top10["NAME"].tolist(), metoda_2d, JEZIK),
            use_container_width=True,
            theme=None,
        )
        st.caption(p.t("caption_graf_mapa", JEZIK, n=len(prostor.df)))

    # 3) Jedinstvenost
    if st.button(p.t("btn_generiraj_jed", JEZIK), key="btn_jed"):
        st.session_state["prikazi_graf_jedinstvenosti"] = True
    if st.session_state.get("prikazi_graf_jedinstvenosti"):
        rezultati_svih, prosjek_top10, jedinstvenost = a.jedinstvenost(S, idx)
        st.metric(p.t("metric_jedinstvenost", JEZIK), f"{jedinstvenost:.0f} / 100")
        st.altair_chart(
            a.graf_jedinstvenosti(rezultati_svih, idx, prosjek_top10, naziv_metode.lower(), JEZIK),
            use_container_width=True,
            theme=None,
        )
        st.caption(p.t("caption_graf_jed", JEZIK))

    st.divider()
    st.caption(p.t("footer_igrac_stranica", JEZIK))


_ime_iz_urla = st.query_params.get("igrac")
if _ime_iz_urla:
    _idx_urla = _pronadi_tocno(prostor, _ime_iz_urla)
    if _idx_urla is None:
        st.error(p.t("greska_igrac_nije_pronadjen", JEZIK, ime=_ime_iz_urla))
        if st.button(p.t("btn_natrag", JEZIK)):
            del st.query_params["igrac"]
            st.rerun()
    else:
        _stranica_igraca(prostor, _idx_urla)
    st.stop()

# ---------------------------------------------------------------------------
# Bočna traka — pretraga, metoda, napredne postavke, filtri
# ---------------------------------------------------------------------------
with st.sidebar:
    zag1, zag2 = st.columns([2, 1])
    with zag1:
        st.markdown(f"### {p.t('app_brand', JEZIK)}")
    with zag2:
        st.markdown(f'<div style="padding-top:10px">{_jezik_toggle_html()}</div>', unsafe_allow_html=True)
    st.caption(p.t("app_sidebar_subtitle", JEZIK, n=len(prostor.df)))

    imena = sorted(prostor.imena)
    zadano = "Bruno Fernandes" if "Bruno Fernandes" in imena else imena[0]
    igrac = st.selectbox(
        p.t("label_igrac", JEZIK),
        options=imena,
        index=imena.index(zadano),
        help=p.t("help_igrac", JEZIK),
    )

    st.markdown(f"**{p.t('label_metoda', JEZIK)}**")
    metoda = st.radio(
        p.t("label_metoda", JEZIK),
        options=list(s.PRESETI.keys()),
        format_func=lambda k: p.t(f"metoda_naziv_{k}", JEZIK),
        key="metoda_radio",
        index=list(s.PRESETI.keys()).index(st.session_state["metoda"]),
        on_change=_odaberi_metodu,
        label_visibility="collapsed",
    )
    st.caption(p.t(f"metoda_opis_{st.session_state['metoda']}", JEZIK))

    with st.expander(p.t("label_sto_znace_metode", JEZIK)):
        for k in s.PRESETI:
            st.markdown(f"**{p.t(f'metoda_naziv_{k}', JEZIK)}**")
            st.caption(p.t(f"metoda_objasnjenje_{k}", JEZIK))

    with st.expander(p.t("label_napredno", JEZIK), expanded=st.session_state["prilagodjeno"]):
        st.select_slider(
            p.t("label_stil_kolicina", JEZIK),
            options=ALPHA_OPCIJE,
            value=st.session_state["alpha"],
            format_func=lambda v: f"{v:.2f}",
            key="alpha_slider",
            on_change=_dodirnuo_napredno,
            help=p.t("help_stil_kolicina", JEZIK),
        )
        st.checkbox(
            p.t("label_uvazi_povezanost", JEZIK),
            value=st.session_state["koristi_M"],
            key="matM_checkbox",
            on_change=_dodirnuo_napredno,
            help=p.t("help_uvazi_povezanost", JEZIK),
        )
        if st.session_state["prilagodjeno"]:
            st.caption(p.t("caption_prilagodjeno", JEZIK))

    st.markdown(f"**{p.t('label_filtri', JEZIK)}**")
    ista_uloga = st.checkbox(p.t("label_ista_uloga", JEZIK), value=False, key="ista_uloga_checkbox")
    izbaci_ligu = st.checkbox(p.t("label_izbaci_ligu", JEZIK), value=False, key="izbaci_ligu_checkbox")
    min_minuta = st.slider(
        p.t("label_min_minuta", JEZIK), min_value=450, max_value=3420, value=450, step=90,
        key="min_minuta_slider",
    )
    n_rezultata = st.slider(
        p.t("label_broj_rezultata", JEZIK), min_value=5, max_value=25, value=10, key="n_rezultata_slider"
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
        f'<div style="padding-top:10px"><span class="chip">{html.escape(p.prevedi_ulogu(cilj["ULOGA"], JEZIK))}</span></div>',
        unsafe_allow_html=True,
    )
with top3:
    liga = f'{cilj["LEAGUE"]} · ' if prostor.ima_lige else ""
    meta = p.t(
        "meta_nastupa_minuta", JEZIK,
        apps=int(cilj["APPS"]), mins=p.formatiraj_broj(int(cilj["MINS"]), JEZIK),
    )
    st.markdown(
        f'<div style="padding-top:14px" class="target-meta">{liga}{meta}</div>',
        unsafe_allow_html=True,
    )
with top4:
    if st.button(p.t("btn_profil_i_grafovi", JEZIK), key="prof_cilj"):
        st.query_params["igrac"] = cilj["NAME"]
        st.rerun()

st.markdown(f'<div class="section-label">{p.t("label_profil_znacajke", JEZIK)}</div>', unsafe_allow_html=True)

z_ciljni = prostor.X[idx]
najistaknutije = np.argsort(-np.abs(z_ciljni))[:6]
redci = []
for j in najistaknutije:
    z = float(z_ciljni[j])
    naziv = p.znacajka(prostor.znacajke[j], JEZIK)
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
st.markdown(f"**{p.t('najslicnijih_igraca', JEZIK, n=len(rezultati))}** &nbsp;·&nbsp; {_naziv_metode(JEZIK).lower()}")

if len(rezultati) == 0:
    st.info(p.t("nijedan_igrac_filtri", JEZIK))
else:
    maks = float(rezultati["SLICNOST"].max()) or 1.0

    # Zaglavlje mora dijeliti IDENTIČNU strukturu stupaca kao retci ispod (isti
    # st.columns tjedine, ista unutarnja mreža unutar cols[2]) — ranije je
    # zaglavlje bilo zaseban, ručno namješten CSS grid koji se s pravim
    # retcima poklapao tek slučajno, pa se raspao čim je dodan 5. stupac.
    head_cols = st.columns([0.04, 0.30, 1, 0.13, 0.13], gap="small")
    with head_cols[0]:
        st.markdown('<div class="row-head-label">&nbsp;</div>', unsafe_allow_html=True)
    with head_cols[1]:
        st.markdown(f'<div class="row-head-label">{p.t("col_igrac_uloga", JEZIK)}</div>', unsafe_allow_html=True)
    with head_cols[2]:
        st.markdown(
            '<div style="display:grid;grid-template-columns:1fr 70px 60px 90px;gap:12px">'
            f'<span class="row-head-label">{p.t("word_liga", JEZIK)}</span>'
            f'<span class="row-head-label" style="text-align:right">{p.t("word_min", JEZIK)}</span>'
            '<span class="row-head-label"></span>'
            f'<span class="row-head-label" style="text-align:right">{p.t("word_slicnost", JEZIK)}</span></div>',
            unsafe_allow_html=True,
        )
    with head_cols[3]:
        st.markdown('<div class="row-head-label">&nbsp;</div>', unsafe_allow_html=True)
    with head_cols[4]:
        st.markdown('<div class="row-head-label">&nbsp;</div>', unsafe_allow_html=True)

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
            uloga_r = html.escape(p.prevedi_ulogu(r["ULOGA"], JEZIK))
            st.markdown(
                f'<div style="padding:8px 0"><span class="rname">{html.escape(r["NAME"])}</span><br>'
                f'<span class="rrole">{uloga_r}</span></div>',
                unsafe_allow_html=True,
            )
        with cols[2]:
            min_txt = p.formatiraj_broj(int(r["MINS"]), JEZIK)
            st.markdown(
                f'<div style="padding:8px 0;display:grid;grid-template-columns:1fr 70px 60px 90px;'
                f'gap:12px;align-items:center">'
                f'<span class="rliga">{liga_txt}</span>'
                f'<span class="rmin">{min_txt}′</span>'
                f'<span class="meter"><i style="width:{sirina}%"></i></span>'
                f'<span class="rscore">{r["SLICNOST"]:.1f} %</span></div>',
                unsafe_allow_html=True,
            )
        with cols[3]:
            if st.button(p.t("btn_usporedi", JEZIK), key=f"cmp_{i}_{r['NAME']}"):
                st.session_state["usporedba_s"] = r["NAME"]
        with cols[4]:
            if st.button(p.t("btn_profil", JEZIK), key=f"prof_{i}_{r['NAME']}"):
                st.query_params["igrac"] = r["NAME"]
                st.rerun()

# ---------------------------------------------------------------------------
# Panel usporedbe — zašto su dva igrača slična/različita
# ---------------------------------------------------------------------------
if st.session_state["usporedba_s"] and len(rezultati):
    st.divider()
    drugi_idx = s.pronadi_igraca(prostor, st.session_state["usporedba_s"])[0]
    razlika = s.usporedi_profile(prostor, idx, drugi_idx)
    razlika["znacajka"] = razlika["znacajka"].map(
        lambda naziv_hr: p.t(
            f"znacajka_{s.NAZIVI_OBRNUTO.get(naziv_hr, '')}", JEZIK
        ) if f"znacajka_{s.NAZIVI_OBRNUTO.get(naziv_hr, '')}" in p.T else naziv_hr
    )

    st.markdown(
        f"**{html.escape(cilj['NAME'])}** &nbsp;↔&nbsp; "
        f"**{html.escape(st.session_state['usporedba_s'])}**"
    )
    st.caption(p.t("caption_standardizirane", JEZIK))

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
        st.markdown(_crtaj_razlaganje(p.t("najvece_razlike", JEZIK), razlika.head(4)), unsafe_allow_html=True)
    with desno:
        st.markdown(_crtaj_razlaganje(p.t("najblize_podudaranje", JEZIK), razlika.tail(4)), unsafe_allow_html=True)

st.divider()
st.caption(p.t("footer_glavna", JEZIK))
