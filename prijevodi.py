"""Prijevodi sučelja — hrvatski / engleski.

Jedan rječnik za cijelu aplikaciju (app.py, analitika.py, landing.html).
`similarity.py` namjerno ostaje isključivo hrvatski — to je istraživački/
modelski sloj koji dijeli i notebook, pa se ne dira; ovaj modul samo DODAJE
engleske inačice istih pojmova (statistike, metode) za prikaz u sučelju,
preko istih ključeva kakve već koristi `similarity.py` (nazivi stupaca,
ključevi presetova), tako da nema duple izvore istine.

Predlošci s `{...}` popunjavaju se pozivateljevim `.format(...)`.
"""

from __future__ import annotations

JEZICI = ("hr", "en")
ZADANI_JEZIK = "hr"

# ---------------------------------------------------------------------------
# Opći UI tekstovi
# ---------------------------------------------------------------------------
T: dict[str, dict[str, str]] = {
    # --- marka / naslov stranice -------------------------------------------------
    "app_page_title": {"hr": "Podudarnost — tražilica sličnih igrača", "en": "PlayerMatch — similar player finder"},
    "app_brand": {"hr": "⚽ Podudarnost", "en": "⚽ PlayerMatch"},
    "app_sidebar_subtitle": {"hr": "{n} igrača · top-5 europskih liga", "en": "{n} players · top-5 European leagues"},

    # --- prekidač jezika -----------------------------------------------------
    "lang_hr": {"hr": "HR", "en": "HR"},
    "lang_en": {"hr": "EN", "en": "EN"},

    # --- bočna traka -----------------------------------------------------------
    "label_igrac": {"hr": "Igrač", "en": "Player"},
    "help_igrac": {
        "hr": "Upiši dio imena za pretragu — lista se filtrira dok tipkaš.",
        "en": "Type part of a name to search — the list filters as you type.",
    },
    "label_metoda": {"hr": "Metoda", "en": "Method"},
    "label_sto_znace_metode": {"hr": "Što znače metode?", "en": "What do the methods mean?"},
    "label_napredno": {"hr": "Napredno", "en": "Advanced"},
    "label_stil_kolicina": {"hr": "Stil ↔ količina", "en": "Style ↔ volume"},
    "help_stil_kolicina": {
        "hr": (
            "Klizač bira što 'sličnost' znači. Skroz desno ('stil'): "
            "traži igrače koji igraju na isti način, bez obzira igraju "
            "li malo ili puno. Skroz lijevo ('količina'): traži igrače "
            "koji ostvaruju sličan BROJ istih akcija — golova, "
            "dodavanja — čak i ako im je stil drugačiji. Sredina je "
            "mješavina to dvoje."
        ),
        "en": (
            "The slider picks what 'similarity' means. All the way right "
            "('style'): finds players who play the same way, regardless of "
            "how much or how little they do it. All the way left "
            "('volume'): finds players who put up a similar NUMBER of the "
            "same actions — goals, passes — even if their style differs. "
            "The middle blends the two."
        ),
    },
    "label_uvazi_povezanost": {"hr": "Uvaži povezanost statistika", "en": "Account for stat correlation"},
    "help_uvazi_povezanost": {
        "hr": (
            "Kad je uključeno, model zna da su neke statistike "
            "prirodno povezane (npr. dodavanja i dodavanja u završnu "
            "trećinu) pa ih ne tretira kao potpuno nezavisne. Obično "
            "daje profinjenije rezultate."
        ),
        "en": (
            "When on, the model knows that some stats are naturally "
            "correlated (e.g. passes and final-third passes) and doesn't "
            "treat them as fully independent. Usually gives more refined "
            "results."
        ),
    },
    "caption_prilagodjeno": {
        "hr": "Prilagođeno — više ne prati odabranu metodu iznad.",
        "en": "Custom — no longer follows the method selected above.",
    },
    "label_filtri": {"hr": "Filtri", "en": "Filters"},
    "label_ista_uloga": {"hr": "Samo ista uloga", "en": "Same role only"},
    "label_izbaci_ligu": {"hr": "Izbaci istu ligu", "en": "Exclude same league"},
    "label_min_minuta": {"hr": "Najmanje minuta", "en": "Minimum minutes"},
    "label_broj_rezultata": {"hr": "Broj rezultata", "en": "Number of results"},

    # --- metode: nazivi (ključevi = similarity.PRESETI ključevi) ----------------
    "metoda_naziv_soft_cosine": {"hr": "Soft-kosinusna", "en": "Soft cosine"},
    "metoda_naziv_cosine": {"hr": "Kosinusna", "en": "Cosine"},
    "metoda_naziv_euclidean": {"hr": "Euklidska", "en": "Euclidean"},
    "metoda_naziv_dot_product": {"hr": "Skalarni produkt", "en": "Dot product"},

    # --- metode: kratki opis (caption ispod odabira) ----------------------------
    "metoda_opis_soft_cosine": {
        "hr": "Kao kosinusna, ali uvažava da su statistike međusobno povezane.",
        "en": "Like cosine, but accounts for statistics being correlated.",
    },
    "metoda_opis_cosine": {
        "hr": "Mjeri smjer profila — sličan stil bez obzira na razinu doprinosa.",
        "en": "Measures the direction of the profile — similar style regardless of output level.",
    },
    "metoda_opis_euclidean": {
        "hr": "Udaljenost u prostoru značajki — kažnjava svaku razliku, i u stilu i u razini.",
        "en": "Distance in feature space — penalizes every difference, in style and in level alike.",
    },
    "metoda_opis_dot_product": {
        "hr": "Nagrađuje iste akcije u sličnoj KOLIČINI — favorizira volumne igrače.",
        "en": "Rewards the same actions in similar VOLUME — favours high-output players.",
    },

    # --- metode: dugi opis (expander "Što znače metode?") -----------------------
    "metoda_objasnjenje_soft_cosine": {
        "hr": (
            "Zadano. Traži igrače sličnog STILA igre, a pritom uvažava da su "
            "neke statistike prirodno povezane (npr. tko puno dodaje, obično "
            "puno dodaje i u završnu trećinu) pa ih ne broji dvaput. "
            "Najuravnoteženiji izbor za većinu upita."
        ),
        "en": (
            "Default. Finds players of similar playing STYLE, while "
            "accounting for some stats being naturally correlated (e.g. "
            "whoever passes a lot usually also passes a lot into the final "
            "third) so it doesn't double-count them. The most balanced "
            "choice for most searches."
        ),
    },
    "metoda_objasnjenje_cosine": {
        "hr": (
            "Traži igrače sličnog STILA igre — gleda samo 'oblik' profila "
            "(u čemu je igrač relativno jak ili slab), a ne koliko je toga "
            "ukupno odradio. Dobro za usporedbu igrača s različitom minutažom."
        ),
        "en": (
            "Finds players of similar playing STYLE — looks only at the "
            "'shape' of the profile (where a player is relatively strong or "
            "weak), not how much they did overall. Good for comparing "
            "players with very different playing time."
        ),
    },
    "metoda_objasnjenje_euclidean": {
        "hr": (
            "Traži igrače koji su najbliži cilju u svemu odjednom — i po "
            "stilu i po razini doprinosa. Stroža mjera: kažnjava svaku "
            "razliku, veliku ili malu."
        ),
        "en": (
            "Finds players closest to the target in everything at once — "
            "style and output level both. A stricter measure: it penalizes "
            "every difference, big or small."
        ),
    },
    "metoda_objasnjenje_dot_product": {
        "hr": (
            "Traži igrače koji rade ISTE stvari u SLIČNOJ KOLIČINI (npr. "
            "slično puno golova i dodavanja). Favorizira igrače s visokom "
            "minutažom — dobro za 'jednako produktivnu' zamjenu, manje za "
            "sličan stil."
        ),
        "en": (
            "Finds players who do the SAME things in SIMILAR VOLUME (e.g. "
            "a similar amount of goals and passes). Favours high-minutes "
            "players — good for an 'equally productive' replacement, less "
            "so for a similar style."
        ),
    },

    # --- statistike (ključevi = similarity.ZNACAJKE nazivi stupaca) -------------
    "znacajka_ATT_GOALS": {"hr": "golovi", "en": "goals"},
    "znacajka_ATT_XG": {"hr": "xG", "en": "xG"},
    "znacajka_ATT_SHOTS": {"hr": "udarci", "en": "shots"},
    "znacajka_ATT_SOT": {"hr": "udarci u okvir", "en": "shots on target"},
    "znacajka_ATT_XG PER SHOT": {"hr": "xG po udarcu", "en": "xG per shot"},
    "znacajka_CAR_TOTAL (ALL CARRIES)": {"hr": "vođenja lopte", "en": "carries"},
    "znacajka_CAR_AVG (M) (ALL CARRIES)": {"hr": "prosj. duljina vođenja", "en": "avg. carry length"},
    "znacajka_CAR_TOTAL (PROGRESSIVE)": {"hr": "progresivna vođenja", "en": "progressive carries"},
    "znacajka_CAR_AVG (M) (PROGRESSIVE)": {"hr": "prosj. duljina progresije", "en": "avg. progressive length"},
    "znacajka_CAR_ENDED WITH SHOT": {"hr": "vođenja do udarca", "en": "carries to shot"},
    "znacajka_CAR_ENDED WITH GOAL": {"hr": "vođenja do gola", "en": "carries to goal"},
    "znacajka_CAR_ENDED WITH CHANCE": {"hr": "vođenja do prilike", "en": "carries to chance"},
    "znacajka_CAR_ENDED WITH ASSIST": {"hr": "vođenja do asistencije", "en": "carries to assist"},
    "znacajka_DEF_TACKLES": {"hr": "oduzimanja", "en": "tackles"},
    "znacajka_DEF_INTS": {"hr": "presijecanja", "en": "interceptions"},
    "znacajka_DEF_POS WON": {"hr": "osvojene lopte", "en": "possession won"},
    "znacajka_DEF_BLOCKS": {"hr": "blokovi", "en": "blocks"},
    "znacajka_DEF_CLEARANCES": {"hr": "izbijanja", "en": "clearances"},
    "znacajka_DEF_TOTAL (GROUND DUELS)": {"hr": "duel na tlu", "en": "ground duels"},
    "znacajka_DEF_% (GROUND DUELS)": {"hr": "% duela na tlu", "en": "% ground duels won"},
    "znacajka_DEF_TOTAL (AERIAL DUELS)": {"hr": "zračni dueli", "en": "aerial duels"},
    "znacajka_DEF_% (AERIAL DUELS)": {"hr": "% zračnih duela", "en": "% aerial duels won"},
    "znacajka_PAS_TOTAL (OPEN PLAY PASSES)": {"hr": "dodavanja", "en": "passes"},
    "znacajka_PAS_% (OPEN PLAY PASSES)": {"hr": "% dodavanja", "en": "% passes completed"},
    "znacajka_PAS_TOTAL (FINAL THIRD PASSES)": {"hr": "dodavanja u završnu trećinu", "en": "final-third passes"},
    "znacajka_PAS_% (FINAL THIRD PASSES)": {"hr": "% u završnu trećinu", "en": "% final-third completed"},
    "znacajka_PAS_TOTAL (CROSSES)": {"hr": "centaršutevi", "en": "crosses"},
    "znacajka_PAS_% (CROSSES)": {"hr": "% centaršuteva", "en": "% crosses completed"},
    "znacajka_PAS_THROUGH BALLS": {"hr": "ubačaji iza obrane", "en": "through balls"},

    # --- obitelji statistika (ključevi = analitika.KATEGORIJE vrijednosti) ------
    "kategorija_Napad": {"hr": "Napad", "en": "Attack"},
    "kategorija_Vođenje lopte": {"hr": "Vođenje lopte", "en": "Carrying"},
    "kategorija_Obrana": {"hr": "Obrana", "en": "Defending"},
    "kategorija_Dodavanje": {"hr": "Dodavanje", "en": "Passing"},
    "kategorija_Ostalo": {"hr": "Ostalo", "en": "Other"},
    "uloga_nizak_volumen": {"hr": "nizak volumen ({naziv})", "en": "low volume ({naziv})"},

    # --- opće riječi za male tablice/legende -------------------------------------
    "word_igrac": {"hr": "Igrač", "en": "Player"},
    "word_uloga": {"hr": "Uloga", "en": "Role"},
    "word_liga": {"hr": "Liga", "en": "League"},
    "word_min": {"hr": "Min.", "en": "Min."},
    "word_slicnost": {"hr": "Sličnost", "en": "Similarity"},

    # --- glavna stranica: cilj i profil ------------------------------------------
    "label_profil_znacajke": {"hr": "Profil — najizraženije značajke", "en": "Profile — most distinctive stats"},
    "btn_profil_i_grafovi": {"hr": "Profil i grafovi →", "en": "Profile & charts →"},
    "meta_nastupa_minuta": {"hr": "{apps} nastupa · {mins} minuta", "en": "{apps} appearances · {mins} minutes"},

    # --- rezultati ----------------------------------------------------------------
    "najslicnijih_igraca": {"hr": "{n} najsličnijih igrača", "en": "{n} most similar players"},
    "nacin_prilagodjeno": {"hr": "prilagođeno · α = {alpha:.2f}", "en": "custom · α = {alpha:.2f}"},
    "nijedan_igrac_filtri": {
        "hr": "Nijedan igrač ne zadovoljava odabrane filtre.",
        "en": "No player matches the selected filters.",
    },
    "col_igrac_uloga": {"hr": "Igrač / uloga", "en": "Player / role"},
    "btn_usporedi": {"hr": "Usporedi ↓", "en": "Compare ↓"},
    "btn_profil": {"hr": "Profil →", "en": "Profile →"},

    # --- panel usporedbe -----------------------------------------------------------
    "caption_standardizirane": {
        "hr": "Standardizirane (z) vrijednosti — 0 je prosjek lige na toj statistici.",
        "en": "Standardized (z) values — 0 is the league average for that stat.",
    },
    "najvece_razlike": {"hr": "Najveće razlike", "en": "Biggest differences"},
    "najblize_podudaranje": {"hr": "Najbliže podudaranje", "en": "Closest match"},

    # --- stranica igrača -----------------------------------------------------------
    "btn_natrag": {"hr": "← Natrag na pretragu", "en": "← Back to search"},
    "greska_igrac_nije_pronadjen": {
        "hr": "Igrač „{ime}” nije pronađen u skupu podataka.",
        "en": 'Player "{ime}" was not found in the dataset.',
    },
    "label_sve_statistike": {"hr": "Sve statistike", "en": "All stats"},
    "label_percentil_odnos": {"hr": "Percentil u odnosu na:", "en": "Percentile relative to:"},
    "opcija_sve_igrace": {"hr": "Sve igrače", "en": "All players"},
    "opcija_istu_ulogu": {"hr": "Istu ulogu ({uloga})", "en": "Same role ({uloga})"},
    "osnovica_svi_igraci": {"hr": "svi igrači", "en": "all players"},
    "osnovica_uloga": {"hr": "uloga: {uloga}", "en": "role: {uloga}"},
    "help_percentil_osnovica": {
        "hr": (
            "Postotak dodavanja stopera znači nešto drugo naspram napadača nego "
            "naspram drugih stopera — obje osnovice su korisne."
        ),
        "en": (
            "A centre-back's passing percentile means something different "
            "against strikers than against other centre-backs — both "
            "baselines are useful."
        ),
    },
    "col_statistika": {"hr": "Statistika", "en": "Stat"},
    "col_obitelj": {"hr": "Obitelj", "en": "Family"},
    "col_per90": {"hr": "Per 90", "en": "Per 90"},
    "col_zvrijednost": {"hr": "Z-vrijednost", "en": "Z-score"},
    "col_percentil": {"hr": "Percentil", "en": "Percentile"},

    "label_analiticki_grafovi": {"hr": "Analitički grafovi", "en": "Analysis charts"},
    "caption_grafovi_metoda": {
        "hr": "Grafovi sličnosti ispod koriste trenutno odabranu metodu na stranici za pretragu: **{metoda}**.",
        "en": "The similarity charts below use the method currently selected on the search page: **{metoda}**.",
    },
    "prilagodjeno_alpha": {"hr": "prilagođeno (α={alpha:.2f})", "en": "custom (α={alpha:.2f})"},

    "btn_generiraj_pct": {"hr": "📊 Generiraj profil percentila", "en": "📊 Generate percentile profile"},
    "caption_graf_pct": {
        "hr": (
            "Percentil po svakoj od 29 statistika, grupirano po obitelji. "
            "Isprekidana crta je medijan (50. percentil)."
        ),
        "en": (
            "Percentile across all 29 stats, grouped by family. "
            "The dashed line is the median (50th percentile)."
        ),
    },
    "btn_generiraj_mapa": {"hr": "🗺️ Generiraj mapu sličnosti", "en": "🗺️ Generate similarity map"},
    "label_2d_metoda": {"hr": "2D metoda", "en": "2D method"},
    "opcija_pca": {"hr": "PCA (brzo)", "en": "PCA (fast)"},
    "opcija_tsne": {"hr": "t-SNE (sporije, jasniji klasteri)", "en": "t-SNE (slower, clearer clusters)"},
    "caption_graf_mapa": {
        "hr": (
            "Svih {n} igrača u 2D prostoru u kojem model traži sličnost. "
            "Veća claret točka je cilj, žute točke su njegovih 10 najsličnijih."
        ),
        "en": (
            "All {n} players in the 2D space where the model looks for "
            "similarity. The larger claret dot is the target, yellow dots "
            "are his 10 closest matches."
        ),
    },
    "btn_generiraj_jed": {"hr": "🎯 Generiraj graf jedinstvenosti", "en": "🎯 Generate uniqueness chart"},
    "metric_jedinstvenost": {"hr": "Jedinstvenost", "en": "Uniqueness"},
    "caption_graf_jed": {
        "hr": (
            "Raspodjela sličnosti svih ostalih igrača prema cilju. Što je "
            "crta (prag top-10) dalje od gomile, to je cilj rjeđi profil — "
            "100 = najrjeđi u skupu, 0 = najzamjenjiviji."
        ),
        "en": (
            "Distribution of every other player's similarity to the "
            "target. The further the line (top-10 threshold) sits from the "
            "crowd, the rarer the target's profile — 100 = rarest in the "
            "set, 0 = most replaceable."
        ),
    },

    # --- podnožja ------------------------------------------------------------------
    "footer_glavna": {
        "hr": (
            "Podaci: https://theanalyst.com/, sezona 2025/26 "
            "(podaci/top5_stats_combined.csv). Uloge su izvedene K-Means "
            "klasteriranjem statistika — skup podataka nema stupac s "
            "pozicijom, pa 'uloga' opisuje stil igre, ne službenu poziciju."
        ),
        "en": (
            "Data: https://theanalyst.com/, 2025/26 season "
            "(podaci/top5_stats_combined.csv). Roles are derived via "
            "K-Means clustering of the stats — the dataset has no position "
            "column, so 'role' describes playing style, not an official "
            "position."
        ),
    },
    "footer_igrac_stranica": {
        "hr": (
            "Podaci: https://theanalyst.com/, sezona 2025/26. Uloge su "
            "izvedene K-Means klasteriranjem statistika."
        ),
        "en": (
            "Data: https://theanalyst.com/, 2025/26 season. Roles are "
            "derived via K-Means clustering of the stats."
        ),
    },

    # --- grafovi (analitika.py) ------------------------------------------------------
    "graf_pct_x_title": {"hr": "percentil ({osnovica})", "en": "percentile ({osnovica})"},
    "graf_pct_tt_statistika": {"hr": "statistika", "en": "stat"},
    "graf_pct_tt_per90": {"hr": "per 90", "en": "per 90"},
    "graf_pct_tt_percentil": {"hr": "percentil", "en": "percentile"},
    "graf_pct_tt_obitelj": {"hr": "obitelj", "en": "family"},
    "graf_mapa_naslov": {"hr": "Svi igrači u 2D prostoru ({metoda})", "en": "All players in 2D space ({metoda})"},
    "graf_jed_x_title": {"hr": "sličnost ({mjera})", "en": "similarity ({mjera})"},
    "graf_jed_y_title": {"hr": "broj igrača", "en": "number of players"},
    "graf_jed_tt_igraca": {"hr": "igrača", "en": "players"},
    "graf_jed_tt_prag": {"hr": "prag 10 najsličnijih", "en": "top-10 threshold"},

    # --- landing.html --------------------------------------------------------------
    "landing_eyebrow": {"hr": "Tražilica sličnih igrača", "en": "Similar player finder"},
    "landing_h1": {"hr": "Svaki igrač ima<br>svog dvojnika.", "en": "Every player has<br>a double."},
    "landing_lead": {
        "hr": (
            "Upiši ime igrača i odmah vidi tko u top 5 europskih liga igra "
            "na sličan način — po stilu, ne samo po poziciji. Bez skauta, "
            "bez tablica, par klikova."
        ),
        "en": (
            "Type in a player's name and instantly see who in the top 5 "
            "European leagues plays a similar way — by style, not just by "
            "position. No scouts, no spreadsheets, a couple of clicks."
        ),
    },
    "landing_cta_primary": {"hr": "Pokreni pretragu →", "en": "Start searching →"},
    "landing_cta_secondary": {"hr": "Kako to izgleda? ↓", "en": "What does it look like? ↓"},

    "landing_example_eyebrow": {"hr": "Primjer sučelja", "en": "Interface example"},
    "landing_example_h2": {"hr": "Tko igra kao Bruno Fernandes?", "en": "Who plays like Bruno Fernandes?"},
    "landing_example_lead": {
        "hr": "Ovako izgleda rezultat pretrage — profil igrača i njegovih najsličnijih susjeda.",
        "en": "This is what a search result looks like — a player's profile and his closest matches.",
    },
    "landing_role_playmaker": {"hr": "Kreator igre", "en": "Playmaker"},
    "landing_role_organizator": {"hr": "Vezni organizator", "en": "Deep-lying playmaker"},
    "landing_role_box2box": {"hr": "Box-to-box", "en": "Box-to-box"},
    "landing_example_meta": {
        "hr": "Premier League · 34 nastupa · 2.847 minuta",
        "en": "Premier League · 34 appearances · 2,847 minutes",
    },
    "landing_similar_header": {"hr": "5 najsličnijih igrača", "en": "5 most similar players"},
    "landing_similar_method": {"hr": "soft-kosinusna metoda", "en": "soft cosine method"},
    "landing_illustrative": {
        "hr": "Ilustrativni prikaz sučelja — stvarni rezultati ovise o odabranoj metodi i filtrima.",
        "en": "Illustrative preview — actual results depend on the selected method and filters.",
    },

    "landing_faq_eyebrow": {"hr": "Česta pitanja", "en": "Frequently asked questions"},
    "landing_faq_h2": {"hr": "Prije nego kreneš", "en": "Before you start"},
    "landing_faq_q1": {"hr": "Kako birete tko je sličan?", "en": "How do you decide who's similar?"},
    "landing_faq_a1": {
        "hr": (
            "Uspoređujemo stil igre po desecima statistika po 90 minuta — "
            "golovi, dodavanja, vođenja, obrana. Zadana metoda pametno "
            "prepoznaje da su neke od tih statistika prirodno povezane, "
            "pa ih ne broji dvaput."
        ),
        "en": (
            "We compare playing style across dozens of per-90 stats — "
            "goals, passes, carries, defending. The default method "
            "intelligently recognizes that some of these stats are "
            "naturally correlated, so it doesn't double-count them."
        ),
    },
    "landing_faq_q2": {
        "hr": "Igrač koji me zanima ima puno manje minuta — bitno je li to?",
        "en": "The player I'm interested in has far fewer minutes — does that matter?",
    },
    "landing_faq_a2": {
        "hr": (
            "Ne mora biti. Postoji način usporedbe koji gleda samo \"oblik\" "
            "profila, bez obzira igra li netko malo ili puno, a filtar za "
            "minimalnu minutažu čuva rezultate stabilnima."
        ),
        "en": (
            "Not necessarily. There's a comparison mode that looks only at "
            "the \"shape\" of the profile, regardless of how much or little "
            "someone plays, and the minimum-minutes filter keeps results "
            "stable."
        ),
    },
    "landing_faq_q3": {"hr": "Otkud dolaze podaci?", "en": "Where does the data come from?"},
    "landing_faq_a3": {
        "hr": "Statistike top 5 europskih liga za sezonu 2025/26, prikupljene s theanalyst.com.",
        "en": "Stats from the top 5 European leagues for the 2025/26 season, collected from theanalyst.com.",
    },
    "landing_faq_q4": {
        "hr": "Zašto piše \"uloga\" umjesto pozicije?",
        "en": "Why does it say \"role\" instead of position?",
    },
    "landing_faq_a4": {
        "hr": (
            "Podaci nemaju stupac s pozicijom, pa smo igrače grupirali po "
            "tome kako zaista igraju — pa \"uloga\" opisuje stil, a ne "
            "službenu poziciju s terena."
        ),
        "en": (
            "The data has no position column, so players are grouped by "
            "how they actually play — so \"role\" describes style, not an "
            "official on-field position."
        ),
    },
    "landing_faq_q5": {
        "hr": "Treba li mi znanje statistike da bih ovo koristio/la?",
        "en": "Do I need a statistics background to use this?",
    },
    "landing_faq_a5": {
        "hr": (
            "Ne. Upišeš ime, dobiješ listu. Napredne postavke su tu ako ih "
            "poželiš, ali zadane vrijednosti rade dobar posao same."
        ),
        "en": (
            "No. Type a name, get a list. Advanced settings are there if "
            "you want them, but the defaults do a good job on their own."
        ),
    },

    "landing_cta2_title": {"hr": "Spreman/na za usporedbu?", "en": "Ready to compare?"},
    "landing_cta2_lead": {
        "hr": "Upiši prvo ime koje ti padne na pamet — vidjet ćeš rezultate za par sekundi.",
        "en": "Type in the first name that comes to mind — you'll see results in seconds.",
    },
    "landing_footer": {
        "hr": (
            "Podaci: theanalyst.com, sezona 2025/26. Uloge su izvedene "
            "grupiranjem statistika — skup podataka nema stupac s pozicijom."
        ),
        "en": (
            "Data: theanalyst.com, 2025/26 season. Roles are derived by "
            "clustering the stats — the dataset has no position column."
        ),
    },
}


def t(kljuc: str, jezik: str, **kwargs) -> str:
    """Dohvati prijevod za `kljuc` na `jeziku`; nepoznat jezik pada na hrvatski.

    Preostali `kwargs` popunjavaju `{...}` mjesta u predlošku (`.format`).
    """
    unos = T.get(kljuc)
    if unos is None:
        return kljuc
    tekst = unos.get(jezik, unos.get(ZADANI_JEZIK, kljuc))
    return tekst.format(**kwargs) if kwargs else tekst


def znacajka(stupac: str, jezik: str) -> str:
    """Prijevod naziva statistike; nepoznat stupac vraća stupac kakav jest."""
    return t(f"znacajka_{stupac}", jezik) if f"znacajka_{stupac}" in T else stupac


def kategorija_naziv(kategorija: str, jezik: str) -> str:
    """Prijevod naziva obitelji statistika ('Napad', 'Obrana', …)."""
    kljuc = f"kategorija_{kategorija}"
    return t(kljuc, jezik) if kljuc in T else kategorija


def formatiraj_broj(n: int, jezik: str) -> str:
    """Tisućice: '.' za hrvatski (1.234), ',' za engleski (1,234)."""
    s = f"{n:,}"
    return s.replace(",", ".") if jezik == "hr" else s


_PREFIKS_NIZAK_VOLUMEN = "nizak volumen ("


def prevedi_ulogu(uloga: str, jezik: str) -> str:
    """Prevedi automatski generirani naziv uloge (similarity.izvedi_uloge).

    Uloge se grade JEDNOM, pri učitavanju podataka, uvijek na hrvatskom (spoj
    1-2 naziva iz similarity.NAZIVI, ili poseban oblik "nizak volumen (X)" za
    klastere bez izražene pozitivne značajke). Ovdje se ista, već izgrađena
    oznaka samo PRIKAZUJE na drugom jeziku — grupiranje igrača i dalje se
    oslanja na izvorni hrvatski niz kao ključ, ne na ovaj prijevod.
    """
    if jezik == "hr":
        return uloga

    import similarity as s  # lokalni uvoz da se izbjegne veza na vrhu modula

    if uloga.startswith(_PREFIKS_NIZAK_VOLUMEN) and uloga.endswith(")"):
        naziv_hr = uloga[len(_PREFIKS_NIZAK_VOLUMEN) : -1]
        kljuc = s.NAZIVI_OBRNUTO.get(naziv_hr)
        return t("uloga_nizak_volumen", jezik, naziv=znacajka(kljuc, jezik) if kljuc else naziv_hr)

    prevedeni = []
    for dio in uloga.split(" · "):
        kljuc = s.NAZIVI_OBRNUTO.get(dio)
        prevedeni.append(znacajka(kljuc, jezik) if kljuc else dio)
    return " · ".join(prevedeni)
