"""Model sličnosti nogometnih igrača na podacima top-5 europskih liga.

Ovaj modul sadrži SAMO model sličnosti — bez usporedne analize metrika koja je
služila radu. Koriste ga i notebook i Streamlit aplikacija, pa logika postoji
na jednom mjestu.

Objedinjeni model
-----------------
Četiri izvorne metrike nisu četiri neovisna modela, nego dva "gumba" na istoj
formuli:

    sim(a, b) = (aᵀ M b) / (aᵀ M a)^(α/2) · (bᵀ M b)^(α/2)

    M = I,    α = 1  ->  kosinusna sličnost
    M = korr, α = 1  ->  soft-kosinusna sličnost
    M = I,    α = 0  ->  skalarni produkt

Euklidska udaljenost nije poseban slučaj te formule, ali na standardiziranim
podacima nakon L2-normalizacije vrijedi d² = 2 − 2·cos, pa daje isti poredak kao
kosinus — dakle mjeri istu os, samo kao udaljenost umjesto kao kut.

Time se dva parametra prevode u dva razumljiva pitanja:
    α  — "sličan STIL" (α=1) naspram "slična KOLIČINA doprinosa" (α=0),
    M  — smiju li povezane statistike (npr. dodavanja i dodavanja u završnu
         trećinu) doprinositi zajedno umjesto kao neovisne osi.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ZADANI_CSV = Path(__file__).parent / "podaci" / "top5_stats_combined.csv"

# ---------------------------------------------------------------------------
# Prostor značajki
# ---------------------------------------------------------------------------
# Sve su vrijednosti već per-90, dakle mjere STIL igre, a ne odigranu minutažu.
#
# Izbačene su izvedene značajke koje su točna funkcija drugih zadržanih — inače
# bi isti aspekt igre ušao u račun dvaput i dobio dvostruku težinu. Pravilo:
# od trojke {volumen, uspješni, postotak} zadržavamo volumen i postotak.
IZBACENE = {
    "ATT_GOALS VS XG": "= GOALS − XG (linearna kombinacija)",
    "ATT_CONV %": "= GOALS / SHOTS (šum finiširanja; XG PER SHOT već nosi kvalitetu šuta)",
    "CAR_DISTANCE (M) (ALL CARRIES)": "= TOTAL × AVG",
    "CAR_DISTANCE (M) (PROGRESSIVE)": "= TOTAL × AVG",
    "DEF_WON (GROUND DUELS)": "= TOTAL × %",
    "DEF_WON (AERIAL DUELS)": "= TOTAL × %",
    "PAS_SUCCESSFUL (OPEN PLAY PASSES)": "= TOTAL × %",
    "PAS_SUCCESSFUL (FINAL THIRD PASSES)": "= TOTAL × %",
    "PAS_SUCCESSFUL (CROSSES)": "= TOTAL × %",
}

ZNACAJKE = [
    # napad
    "ATT_GOALS",
    "ATT_XG",
    "ATT_SHOTS",
    "ATT_SOT",
    "ATT_XG PER SHOT",
    # vođenje lopte
    "CAR_TOTAL (ALL CARRIES)",
    "CAR_AVG (M) (ALL CARRIES)",
    "CAR_TOTAL (PROGRESSIVE)",
    "CAR_AVG (M) (PROGRESSIVE)",
    "CAR_ENDED WITH SHOT",
    "CAR_ENDED WITH GOAL",
    "CAR_ENDED WITH CHANCE",
    "CAR_ENDED WITH ASSIST",
    # obrana
    "DEF_TACKLES",
    "DEF_INTS",
    "DEF_POS WON",
    "DEF_BLOCKS",
    "DEF_CLEARANCES",
    "DEF_TOTAL (GROUND DUELS)",
    "DEF_% (GROUND DUELS)",
    "DEF_TOTAL (AERIAL DUELS)",
    "DEF_% (AERIAL DUELS)",
    # dodavanje
    "PAS_TOTAL (OPEN PLAY PASSES)",
    "PAS_% (OPEN PLAY PASSES)",
    "PAS_TOTAL (FINAL THIRD PASSES)",
    "PAS_% (FINAL THIRD PASSES)",
    "PAS_TOTAL (CROSSES)",
    "PAS_% (CROSSES)",
    "PAS_THROUGH BALLS",
]

# Čitljivi nazivi za prikaz u sučelju (ulaze i u automatska imena uloga).
NAZIVI = {
    "ATT_GOALS": "golovi",
    "ATT_XG": "xG",
    "ATT_SHOTS": "udarci",
    "ATT_SOT": "udarci u okvir",
    "ATT_XG PER SHOT": "xG po udarcu",
    "CAR_TOTAL (ALL CARRIES)": "vođenja lopte",
    "CAR_AVG (M) (ALL CARRIES)": "prosj. duljina vođenja",
    "CAR_TOTAL (PROGRESSIVE)": "progresivna vođenja",
    "CAR_AVG (M) (PROGRESSIVE)": "prosj. duljina progresije",
    "CAR_ENDED WITH SHOT": "vođenja do udarca",
    "CAR_ENDED WITH GOAL": "vođenja do gola",
    "CAR_ENDED WITH CHANCE": "vođenja do prilike",
    "CAR_ENDED WITH ASSIST": "vođenja do asistencije",
    "DEF_TACKLES": "oduzimanja",
    "DEF_INTS": "presijecanja",
    "DEF_POS WON": "osvojene lopte",
    "DEF_BLOCKS": "blokovi",
    "DEF_CLEARANCES": "izbijanja",
    "DEF_TOTAL (GROUND DUELS)": "duel na tlu",
    "DEF_% (GROUND DUELS)": "% duela na tlu",
    "DEF_TOTAL (AERIAL DUELS)": "zračni dueli",
    "DEF_% (AERIAL DUELS)": "% zračnih duela",
    "PAS_TOTAL (OPEN PLAY PASSES)": "dodavanja",
    "PAS_% (OPEN PLAY PASSES)": "% dodavanja",
    "PAS_TOTAL (FINAL THIRD PASSES)": "dodavanja u završnu trećinu",
    "PAS_% (FINAL THIRD PASSES)": "% u završnu trećinu",
    "PAS_TOTAL (CROSSES)": "centaršutevi",
    "PAS_% (CROSSES)": "% centaršuteva",
    "PAS_THROUGH BALLS": "ubačaji iza obrane",
}

# Presetovi: (α, koristi_M, je_udaljenost) — vidi dokumentaciju modula.
PRESETI = {
    "soft_cosine": {
        "naziv": "Soft-kosinusna",
        "alpha": 1.0,
        "koristi_M": True,
        "udaljenost": False,
        "opis": "Kao kosinusna, ali uvažava da su statistike međusobno povezane.",
    },
    "cosine": {
        "naziv": "Kosinusna",
        "alpha": 1.0,
        "koristi_M": False,
        "udaljenost": False,
        "opis": "Mjeri smjer profila — sličan stil bez obzira na razinu doprinosa.",
    },
    "euclidean": {
        "naziv": "Euklidska",
        "alpha": 1.0,
        "koristi_M": False,
        "udaljenost": True,
        "opis": "Udaljenost u prostoru značajki — kažnjava svaku razliku, i u stilu i u razini.",
    },
    "dot_product": {
        "naziv": "Skalarni produkt",
        "alpha": 0.0,
        "koristi_M": False,
        "udaljenost": False,
        "opis": "Nagrađuje iste akcije u sličnoj KOLIČINI — favorizira volumne igrače.",
    },
}


# ---------------------------------------------------------------------------
# Učitavanje i prostor igrača
# ---------------------------------------------------------------------------
@dataclass
class ProstorIgraca:
    """Sve što je potrebno za računanje sličnosti, izračunato jednom."""

    df: pd.DataFrame  # metapodaci + sirove per-90 vrijednosti
    X: np.ndarray  # standardizirane značajke (n × d)
    znacajke: list[str]
    M: np.ndarray  # korelacijska matrica značajki (za soft-kosinus)
    uloge: pd.Series = field(default=None)  # oznaka uloge po igraču
    nazivi_uloga: dict[int, str] = field(default_factory=dict)

    @property
    def imena(self) -> list[str]:
        return self.df["NAME"].tolist()

    @property
    def ima_lige(self) -> bool:
        return "LEAGUE" in self.df.columns


def _u_broj(s: pd.Series) -> pd.Series:
    """Podnosi i '83.6%' i 83.6 — ovisno o tome je li CSV prošao build_dataset.py."""
    if s.dtype.kind in "if":
        return s
    return pd.to_numeric(
        s.astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False),
        errors="coerce",
    )


def ucitaj_prostor(
    csv_path: str | Path = ZADANI_CSV,
    min_minuta: int = 450,
    n_uloga: int | None = None,
) -> ProstorIgraca:
    """Učitaj CSV i pripremi standardizirani prostor značajki.

    Parametri
    ---------
    csv_path : put do objedinjenog CSV-a (izlaz iz build_dataset.py).
    min_minuta : donji prag minutaže; per-90 stope su na malom uzorku nestabilne.
    n_uloga : broj klastera uloga; None -> automatski izbor po silhouette skoru.
    """
    df = pd.read_csv(csv_path)
    df["NAME"] = df["NAME"].astype(str).str.strip()

    nedostaju = [c for c in ZNACAJKE if c not in df.columns]
    if nedostaju:
        raise ValueError(f"CSV-u nedostaju očekivani stupci: {nedostaju}")

    for c in ZNACAJKE + ["MINS", "APPS"]:
        df[c] = _u_broj(df[c])

    df = df[df["MINS"] >= min_minuta].reset_index(drop=True)
    df[ZNACAJKE] = df[ZNACAJKE].fillna(0.0)

    # Standardizacija je nužna prije bilo kakvog računanja sličnosti: značajke su
    # u posve različitim mjerama (postoci 0–100, dodavanja ~40, golovi ~0.1).
    X = StandardScaler().fit_transform(df[ZNACAJKE].to_numpy(dtype=float))

    M = np.nan_to_num(np.corrcoef(X, rowvar=False), nan=0.0)

    prostor = ProstorIgraca(df=df, X=X, znacajke=list(ZNACAJKE), M=M)
    prostor.uloge, prostor.nazivi_uloga = izvedi_uloge(prostor, n_uloga)
    prostor.df["ULOGA"] = prostor.uloge.map(prostor.nazivi_uloga)
    return prostor


# ---------------------------------------------------------------------------
# Uloge (zamjena za pozicije — skup podataka nema stupac s pozicijom)
# ---------------------------------------------------------------------------
def izvedi_uloge(
    prostor: ProstorIgraca, n_uloga: int | None = None
) -> tuple[pd.Series, dict[int, str]]:
    """Podijeli igrače u uloge K-Means klasteriranjem standardiziranih značajki.

    Skup podataka nema pozicije, pa ulogu izvodimo IZ SAME IGRE. Kako značajke
    ravnomjerno pokrivaju napad, vođenje, obranu i dodavanje, klasteri ispadnu
    prepoznatljivo pozicijski (stoperi, bekovi, veznjaci, krilni, napadači).

    Ako n_uloga nije zadan, bira se broj klastera s najboljim silhouette skorom.
    """
    X = prostor.X

    if n_uloga is None:
        najbolji, najbolji_skor = 6, -1.0
        for k in range(4, 11):
            oznake = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
            skor = silhouette_score(X, oznake)
            if skor > najbolji_skor:
                najbolji, najbolji_skor = k, skor
        n_uloga = najbolji

    oznake = KMeans(n_clusters=n_uloga, random_state=42, n_init=10).fit_predict(X)

    # Ime uloge izvodimo iz onoga što klaster IMA, a ne što mu nedostaje:
    # naziv tipa "↓udarci · ↓udarci u okvir" istinit je, ali kao opis uloge
    # beskoristan. Zato uzimamo dvije najizraženije POZITIVNE značajke, a na
    # negativne se vraćamo samo ako klaster ni u čemu ne odskače naviše.
    nazivi: dict[int, str] = {}
    for k in range(n_uloga):
        centar = X[oznake == k].mean(axis=0)
        pozitivne = [j for j in np.argsort(-centar) if centar[j] > 0.25][:2]
        if pozitivne:
            dijelovi = [NAZIVI.get(prostor.znacajke[j], prostor.znacajke[j]) for j in pozitivne]
            nazivi[k] = " · ".join(dijelovi)
        else:
            j = int(np.argmin(centar))
            naziv = NAZIVI.get(prostor.znacajke[j], prostor.znacajke[j])
            nazivi[k] = f"nizak volumen ({naziv})"

    return pd.Series(oznake, name="uloga"), nazivi


# ---------------------------------------------------------------------------
# Jezgra: objedinjena formula sličnosti
# ---------------------------------------------------------------------------
def rezultati_slicnosti(
    X: np.ndarray,
    idx: int,
    M: np.ndarray | None = None,
    alpha: float = 1.0,
    udaljenost: bool = False,
) -> np.ndarray:
    """Sličnost igrača `idx` prema svim ostalim retcima matrice X.

    Uvijek vraća vrijednosti kod kojih je VEĆE = SLIČNIJE (euklidska se vraća
    kao negativna udaljenost), pa pozivatelj uvijek sortira silazno.

    alpha : 1.0 -> pune normalizacije (stil), 0.0 -> bez normalizacije (količina).
            Međuvrijednosti glatko interpoliraju između to dvoje.
    M     : korelacijska matrica značajki za soft-kosinus; None -> jedinična.
    """
    cilj = X[idx]

    if udaljenost:
        return -np.sqrt(np.sum((X - cilj) ** 2, axis=1))

    if M is None:
        Mb = cilj
        aMa = np.sum(X**2, axis=1)
    else:
        Mb = M @ cilj
        # aᵀ M a za svaki redak a, bez građenja n×n matrice
        aMa = np.einsum("ij,jk,ik->i", X, M, X)

    brojnik = X @ Mb
    bMb = float(cilj @ Mb)

    if alpha == 0.0:
        return brojnik

    nazivnik = np.power(np.clip(aMa, 1e-12, None), alpha / 2) * np.power(
        max(bMb, 1e-12), alpha / 2
    )
    return brojnik / nazivnik


# ---------------------------------------------------------------------------
# Pretraga igrača
# ---------------------------------------------------------------------------
def pronadi_igraca(prostor: ProstorIgraca, upit: str) -> list[int]:
    """Vrati indekse igrača čije ime odgovara upitu (točno pa djelomično)."""
    imena = prostor.df["NAME"]
    upit = upit.strip()
    if not upit:
        return []

    tocno = imena.str.lower() == upit.lower()
    if tocno.any():
        return imena.index[tocno].tolist()

    djelomicno = imena.str.contains(upit, case=False, na=False, regex=False)
    return imena.index[djelomicno].tolist()


def slicni_igraci(
    prostor: ProstorIgraca,
    igrac: str | int,
    preset: str | None = "soft_cosine",
    alpha: float | None = None,
    koristi_M: bool | None = None,
    n: int = 10,
    ista_uloga: bool = False,
    izbaci_istu_ligu: bool = False,
    min_minuta: int = 0,
) -> pd.DataFrame:
    """Vrati n najsličnijih igrača zadanome.

    Zadaje se ILI `preset` (jedna od četiri poznate metrike) ILI izravno
    `alpha`/`koristi_M` za objedinjeni model. Izričito zadani alpha/koristi_M
    imaju prednost pred presetom.
    """
    if isinstance(igrac, str):
        pogodci = pronadi_igraca(prostor, igrac)
        if not pogodci:
            raise KeyError(f"Igrač '{igrac}' nije pronađen.")
        idx = pogodci[0]
    else:
        idx = int(igrac)

    spec = PRESETI.get(preset, PRESETI["soft_cosine"]) if preset else PRESETI["soft_cosine"]
    a = spec["alpha"] if alpha is None else float(alpha)
    m = spec["koristi_M"] if koristi_M is None else bool(koristi_M)
    je_udaljenost = spec["udaljenost"] if preset else False

    rezultat = rezultati_slicnosti(
        prostor.X, idx, M=prostor.M if m else None, alpha=a, udaljenost=je_udaljenost
    )

    out = prostor.df.copy()
    out["_rezultat"] = rezultat
    cilj = prostor.df.iloc[idx]

    out = out.drop(index=idx)
    if ista_uloga and "ULOGA" in out.columns:
        out = out[out["ULOGA"] == cilj["ULOGA"]]
    if izbaci_istu_ligu and prostor.ima_lige:
        # Igraci koji su zimi promijenili ligu imaju vise liga u stupcu
        # ("La Liga / Premier League"), pa obicna usporedba nizova ne bi
        # prepoznala preklapanje. Usporedujemo skupove liga.
        ciljne_lige = {dio.strip() for dio in str(cilj["LEAGUE"]).split("/")}
        out = out[
            ~out["LEAGUE"].apply(
                lambda v: bool({dio.strip() for dio in str(v).split("/")} & ciljne_lige)
            )
        ]
    if min_minuta:
        out = out[out["MINS"] >= min_minuta]

    out = out.sort_values("_rezultat", ascending=False).head(n)

    # Postotak za prikaz. NE skaliramo prema najboljem pogotku — time bi prvi
    # rezultat uvijek bio 100 % pa i kad nijedan igrač zapravo nije sličan.
    #   - α = 1 (kosinusne metrike): već su u [−1, 1], pa je 100·sim iskrena ljestvica,
    #   - euklidska: udaljenost se preslikava u (0, 100] preko exp(−d/√d_znacajki),
    #   - α < 1: rezultat nije omeđen (nazivnik ne poništava magnitudu do kraja),
    #     pa se skalira prema vlastitom rezultatu ciljnog igrača sim(a,a). Time je
    #     100 % = "jednako izražen kao sam cilj", uz moguć prelazak preko 100 %.
    if je_udaljenost:
        out["SLICNOST"] = 100 * np.exp(out["_rezultat"] / np.sqrt(len(prostor.znacajke)))
    elif a < 1.0:
        vlastiti = float(
            rezultati_slicnosti(prostor.X, idx, M=prostor.M if m else None, alpha=a)[idx]
        )
        out["SLICNOST"] = 100 * out["_rezultat"] / max(vlastiti, 1e-12)
    else:
        out["SLICNOST"] = 100 * np.clip(out["_rezultat"], 0, None)

    stupci = ["NAME"]
    if prostor.ima_lige:
        stupci.append("LEAGUE")
    stupci += ["ULOGA", "APPS", "MINS", "SLICNOST", "_rezultat"]
    return out[[c for c in stupci if c in out.columns]].reset_index(drop=True)


def usporedi_profile(prostor: ProstorIgraca, idx_a: int, idx_b: int) -> pd.DataFrame:
    """Razlaganje po značajkama: gdje se dva igrača poklapaju, a gdje razilaze."""
    za = prostor.X[idx_a]
    zb = prostor.X[idx_b]
    return pd.DataFrame(
        {
            "znacajka": [NAZIVI.get(f, f) for f in prostor.znacajke],
            "z_a": za.round(2),
            "z_b": zb.round(2),
            "razlika": np.abs(za - zb).round(2),
            "sirovo_a": prostor.df.iloc[idx_a][prostor.znacajke].to_numpy(),
            "sirovo_b": prostor.df.iloc[idx_b][prostor.znacajke].to_numpy(),
        }
    ).sort_values("razlika", ascending=False, ignore_index=True)
