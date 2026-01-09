# 📘 Manual de utilizare – Versiunea SaaS
## ETH Decision AI

Acest document este **manualul oficial al aplicației SaaS**, destinat să fie păstrat și versionat **împreună cu codul** (ex: `/docs/manual.md`).

---

## 1. Scopul aplicației

ETH Decision AI este o aplicație SaaS care oferă **decizii asistate de inteligență artificială** pentru gestionarea expunerii pe ETH.

Aplicația:
- analizează piața pe mai multe intervale de timp
- generează un scor agregat
- produce o decizie clară de tip investițional
- NU execută tranzacții automat

---

## 2. Public țintă

- investitori crypto
- traderi discreționari
- utilizatori care doresc disciplină și control al riscului

Nu este destinată scalping-ului sau tradingului ultra-frecvent.

---

## 3. Concepte-cheie

### 3.1 Symbol

- implicit: `ETHUSDT`
- extensibil pentru alte active

### 3.2 Timeframes analizate

| Timeframe | Rol |
|---------|----|
| 4h | Momentum pe termen scurt |
| 1d | Trend principal |
| 1w | Context macro |

---

## 4. Arhitectura logică (conceptual)

Fluxul de decizie:

1. Colectare date piață
2. Calcul scor per timeframe
3. Aplicare ponderi
4. Calcul scor final
5. Mapare scor → decizie
6. Calcul confidence

---

## 5. Scorurile

### 5.1 Scor per timeframe

- interval: `[-100, +100]`
- pozitiv = bias bullish
- negativ = bias bearish

### 5.2 Scor final (Weighted Score)

Scor final = sumă ponderată a scorurilor per timeframe.

Ponderi default:
- 4h: 25%
- 1d: 35%
- 1w: 40%

---

## 6. Deciziile posibile

| Interval scor final | Decizie |
|------------------|--------|
| > +20 | ACCUMULATE |
| -20 … +20 | HOLD |
| -20 … -45 | REDUCE |
| < -45 | EXIT |

Decizia este **deterministă**, bazată strict pe scor.

---

## 7. Confidence

### 7.1 Definiție

Confidence reprezintă **claritatea semnalului**, nu certitudinea direcției pieței.

### 7.2 Interval

| Confidence | Semnificație |
|----------|-------------|
| 0–30 | slab |
| 30–50 | moderat |
| 50–70 | puternic |
| 70+ | foarte puternic |

Recomandare implicită: nu se acționează agresiv sub 40.

---

## 8. Exposure & Step Percent

Aplicația nu recomandă modificări brute.

- expunerea este ajustată incremental
- există un prag minim (ex: `MIN_EXPOSURE = 30%`)

`step_percent` indică modificarea sugerată a expunerii.

---

## 9. Endpoint-uri principale (conceptual)

### 9.1 Health

Verifică starea aplicației.

### 9.2 Run Analysis

Rulează analiza completă pentru toate timeframe-urile.

### 9.3 Portfolio Plan

Primește decizia în funcție de expunerea curentă.

---

## 10. Workflow utilizator

1. Utilizatorul setează expunerea curentă
2. Rulează analiza
3. Primește:
   - decizie
   - confidence
   - scoruri
4. Ajustează manual portofoliul

---

## 11. Limitări asumate

- nu prezice prețul
- nu oferă timing exact
- nu garantează profit

Este un **sistem de suport decizional**, nu un bot de tranzacționare.

---

## 12. Principii de utilizare corectă

- evaluare pe serii de decizii
- disciplină
- evitarea overtrading-ului
- focus pe risk management

---

## 13. Versionare document

Acest manual trebuie:
- păstrat în repo
- versionat odată cu codul
- actualizat la fiecare schimbare de logică

---

## 14. Extensii viitoare (opțional)

- modul AUTO
- alerting
- istoric decizii
- multi-asset

---

**End of document**
