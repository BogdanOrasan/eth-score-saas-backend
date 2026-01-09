# 🔧 Manual Admin & Config
## ETH Decision AI – SaaS

Acest document descrie **configurarea și operarea sistemului** din perspectiva administratorului / operatorului.

---

## 1. Rolul Adminului

Adminul controlează:
- pragurile decizionale
- ponderile timeframe-urilor
- regulile de risk management

Adminul NU intervine manual în scoruri individuale.

---

## 2. Parametri principali

### 2.1 Threshold-uri decizionale

| Parametru | Descriere |
|---------|----------|
| ACCUMULATE_THRESHOLD | scor minim pentru acumulare |
| REDUCE_THRESHOLD | scor sub care se reduce |
| EXIT_THRESHOLD | scor critic pentru exit |

Valorile trebuie ajustate rar și testate istoric.

---

## 3. Ponderi Timeframe

| Timeframe | Weight |
|---------|-------|
| 4h | 0.25 |
| 1d | 0.35 |
| 1w | 0.40 |

Suma ponderilor trebuie să fie 1.0.

---

## 4. Risk Management

### 4.1 MIN_EXPOSURE

- definește expunerea minimă permisă
- previne ieșirile totale premature

### 4.2 Step Percent

- definește granularitatea ajustărilor
- recomandat: pași mici (5–10%)

---

## 5. Reguli operaționale

- nu se modifică parametri în timpul piețelor volatile
- orice schimbare trebuie versionată
- modificările se validează pe date istorice

---

## 6. Responsabilitate

Adminul este responsabil pentru:
- coerența logicii
- stabilitatea deciziilor
- comunicarea schimbărilor către utilizatori

---

**End of document**
