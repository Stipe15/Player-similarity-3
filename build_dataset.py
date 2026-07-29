"""Izgradnja objedinjenog skupa podataka top-5 liga iz sirovih izvoza po ligi.

Ulaz  : podaci/{liga}_{kategorija}.csv  (4 kategorije x 5 liga = 20 datoteka)
Izlaz : podaci/top5_stats_combined.csv  (jedan redak po igracu, sa stupcem LEAGUE)

Skripta NECE zapisati izlaz ako validacija skale padne. Konkretno: svi brojivi
stupci moraju biti per-90, a ne sezonski zbrojevi. Izvoz Serie A obrane bio je
u zbrojevima (medijan TACKLES 24.0 naspram ~1.6 u ostale cetiri lige), sto bi
svakog igraca Serie A pretvorilo u obrambenog outliera. Provjera to hvata.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Windows konzola je po zadanom cp1250 i puca na imenima tipa "Nico González".
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PODACI = Path(__file__).parent / "podaci"
IZLAZ = PODACI / "top5_stats_combined.csv"

# datoteka -> citljivo ime lige
LIGE = {
    "pl": "Premier League",
    "laliga": "La Liga",
    "seriea": "Serie A",
    "bundesliga": "Bundesliga",
    "ligue1": "Ligue 1",
}

# kategorija -> prefiks stupca
KATEGORIJE = {
    "attacking": "ATT_",
    "carrying": "CAR_",
    "defending": "DEF_",
    "passing": "PAS_",
}

KLJUCEVI = ["NAME", "APPS", "MINS"]

# Najveci dopusteni omjer medijana jedne lige spram medijana svih liga.
# Zbrojevi umjesto per-90 daju omjer reda velicine 10-20x, pa je 3.0 siguran prag.
PRAG_SKALE = 3.0


def _u_broj(s: pd.Series) -> pd.Series:
    """Pretvori stupac u broj; podnosi postotke ('83.6%') i tisucice ('1,641')."""
    return pd.to_numeric(
        s.astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False),
        errors="coerce",
    )


def ucitaj_ligu(kljuc: str) -> pd.DataFrame:
    """Spoji 4 kategorijske datoteke jedne lige u jedan DataFrame."""
    spojeno = None
    for kategorija, prefiks in KATEGORIJE.items():
        put = PODACI / f"{kljuc}_{kategorija}.csv"
        if not put.exists():
            raise FileNotFoundError(f"Nedostaje {put}")
        df = pd.read_csv(put, thousands=",")
        df["NAME"] = df["NAME"].astype(str).str.strip()

        # Neki izvozi imaju PONOVLJENE RETKE ZAGLAVLJA usred podataka (redak u
        # kojem NAME doslovno pise "NAME"). Ako ostanu, outer merge ih spoji
        # svaki-sa-svakim: tri datoteke sa 7 takvih redaka daju 7^3 = 343
        # fantomska retka. Zato ih micemo prije spajanja, kao i eventualne
        # ponovljene igrace unutar iste kategorije.
        zaglavlja = (df["NAME"] == "NAME").sum()
        df = df[df["NAME"] != "NAME"]
        duplikati = df["NAME"].duplicated().sum()
        df = df.drop_duplicates(subset="NAME", keep="first")
        if zaglavlja or duplikati:
            print(
                f"  {put.name}: uklonjeno {zaglavlja} ponovljenih zaglavlja, "
                f"{duplikati} duplih igraca"
            )

        # APPS/MINS se izmedu kategorija znaju razlikovati (razlicit izvoz),
        # pa ih uzimamo samo iz 'attacking', a drugdje odbacujemo.
        statistike = [c for c in df.columns if c not in KLJUCEVI]
        df = df[(KLJUCEVI if kategorija == "attacking" else ["NAME"]) + statistike]
        df = df.rename(columns={c: prefiks + c for c in statistike})

        spojeno = df if spojeno is None else spojeno.merge(df, on="NAME", how="outer")

    nedostaje = spojeno["MINS"].isna().sum()
    if nedostaje:
        print(f"  upozorenje: {nedostaje} igraca bez MINS (nisu u 'attacking' izvozu) - izbaceni")
        spojeno = spojeno[spojeno["MINS"].notna()]

    spojeno.insert(1, "LEAGUE", LIGE[kljuc])
    return spojeno.reset_index(drop=True)


def provjeri_skalu(po_ligi: dict[str, pd.DataFrame]) -> list[str]:
    """Vrati popis problema: stupci u kojima jedna liga bitno odstupa od ostalih.

    Usporeduju se MEDIJANI po ligi. Per-90 vrijednosti su medu ligama vrlo
    slicne, pa svako odstupanje vece od PRAG_SKALE znaci da je ta liga izvezena
    u drugoj mjeri (tipicno sezonski zbroj umjesto per-90).
    """
    problemi = []
    prva = next(iter(po_ligi.values()))
    stupci = [c for c in prva.columns if c not in ("NAME", "LEAGUE", "APPS", "MINS")]

    for stupac in stupci:
        if "%" in stupac:  # postoci su vec normalizirani
            continue
        medijani = {lg: _u_broj(df[stupac]).median() for lg, df in po_ligi.items()}
        ukupni = pd.Series(medijani).median()
        if not ukupni or ukupni <= 0:
            continue
        for lg, vrijednost in medijani.items():
            omjer = vrijednost / ukupni
            if omjer > PRAG_SKALE or omjer < 1 / PRAG_SKALE:
                problemi.append(
                    f"{LIGE[lg]:<15} {stupac:<38} medijan {vrijednost:>8.2f} "
                    f"naspram {ukupni:.2f} ({omjer:.1f}x)"
                )
    return problemi


def spoji_duplikate(df: pd.DataFrame) -> pd.DataFrame:
    """Spoji igrace koji se pojavljuju u vise liga (zimski transferi).

    Statistike su per-90 stope, pa je ispravno objedinjavanje prosjek PONDERIRAN
    MINUTAMA, a ne obican prosjek: igrac s 1500 minuta u jednoj i 200 u drugoj
    ligi mora biti gotovo u cijelosti odreden prvom ligom.
    """
    duplikati = df["NAME"][df["NAME"].duplicated()].unique()
    if len(duplikati) == 0:
        return df

    stat_stupci = [c for c in df.columns if c not in ("NAME", "LEAGUE", "APPS", "MINS")]

    print(f"  spajam {len(duplikati)} igraca s vise od jedne lige (ponderirano minutama):")

    spojeni_redci = []
    for ime in duplikati:
        grupa = df[df["NAME"] == ime]
        tezine = grupa["MINS"].to_numpy(dtype=float)
        red = grupa.iloc[0].copy()
        red["LEAGUE"] = " / ".join(sorted(grupa["LEAGUE"]))
        red["APPS"] = grupa["APPS"].sum()
        red["MINS"] = grupa["MINS"].sum()
        for c in stat_stupci:
            vrijednosti = grupa[c].to_numpy(dtype=float)
            red[c] = (vrijednosti * tezine).sum() / tezine.sum()
        spojeni_redci.append(red)
        print(f"    - {ime} ({red['LEAGUE']}, {int(red['MINS'])} min)")

    ostali = df[~df["NAME"].isin(duplikati)]
    return pd.concat(
        [ostali, pd.DataFrame(spojeni_redci)], ignore_index=True
    ).reset_index(drop=True)


def main() -> int:
    print("Ucitavanje liga...")
    po_ligi = {}
    for kljuc in LIGE:
        po_ligi[kljuc] = ucitaj_ligu(kljuc)
        print(f"  {LIGE[kljuc]:<15} {len(po_ligi[kljuc]):>4} igraca")

    print("\nValidacija skale (per-90 naspram sezonskih zbrojeva)...")
    problemi = provjeri_skalu(po_ligi)
    if problemi:
        print("\n  NEUSPJEH - izlaz nije zapisan. Ovi stupci nisu u istoj mjeri:\n")
        for p in problemi:
            print("   ", p)
        print(
            "\n  Ponovno izvezi navedenu ligu/kategoriju u per-90 nacinu i pokreni skriptu opet."
        )
        return 1
    print("  u redu - sve lige su na istoj skali")

    df = pd.concat(po_ligi.values(), ignore_index=True)

    # Pretvorba u brojeve mora doci PRIJE spajanja duplikata, jer se ono racuna
    # ponderiranim prosjekom. Postoci se ujedno zapisuju kao brojevi (83.6), a
    # ne kao tekst ('83.6%'), pa aplikacija cita CSV bez dodatnog parsiranja.
    for c in df.columns:
        if c not in ("NAME", "LEAGUE"):
            df[c] = _u_broj(df[c])

    print("\nSpajanje duplikata...")
    df = spoji_duplikate(df)

    df = df.sort_values("NAME").reset_index(drop=True)
    df.to_csv(IZLAZ, index=False, encoding="utf-8")

    print(f"\nZapisano {IZLAZ}")
    print(f"  {len(df)} igraca, {len(df.columns)} stupaca")
    print(f"  lige: {df['LEAGUE'].nunique()} | minute: {df['MINS'].min():.0f}-{df['MINS'].max():.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
