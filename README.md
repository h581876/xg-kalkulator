# xG-kalkulator

Webapp som estimerer sannsynligheten for at et skudd i fotball blir mål (expected goals, xG), basert på en maskinlæringsmodell trent på åpne data fra StatsBomb. 

- Applikasjon: https://xg-kalkulator.streamlit.app/
- Rapport: 

## Hva gjør appen?

Brukeren oppgir skuddets posisjon (x og y), kroppsdel, situasjon (åpent spill eller dødball), om skytteren er under press og om skuddet tas på første berøring. Appen viser sannsynligheten for mål og tegner skuddet på en bane.

## Prosjektstruktur


app.py               Streamlit-appen
src/load_data.py     Henter skudd fra StatsBomb Open Data
src/features.py      Rensing og features (avstand, vinkel osv.)
src/train.py         Trener og evaluerer modellene, lagrer valgt modell
models/model.joblib  Ferdigtrent modell som brukes av appen
report/              Rapport (PDF) og figurer
