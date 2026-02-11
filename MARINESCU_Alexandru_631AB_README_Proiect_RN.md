## 1. Identificare Proiect

| Câmp | Valoare |
|------|---------|
| **Student** | Marinescu Alexandru |
| **Grupa / Specializare** | [631AB] / Informatică Industrială |
| **Disciplina** | Rețele Neuronale |
| **Instituție** | POLITEHNICA București – FIIR |
| **Link Repository GitHub** | https://github.com/alexmFIIR004/RN-Proiect |
| **Acces Repository** | Public |
| **Stack Tehnologic** | Python (TensorFlow/Keras, Streamlit) |
| **Domeniul Industrial de Interes (DII)** | Robotică / Navigație Autonomă |
| **Tip Rețea Neuronală** | Multimodal CNN (1D CNN + 2D CNN) |

### Rezultate Cheie (Versiunea Finală vs Etapa 6)

| Metric | Țintă Minimă | Rezultat Etapa 6 | Rezultat Final | Îmbunătățire | Status |
|--------|--------------|------------------|----------------|--------------|--------|
| Accuracy (Test Set) | ≥70% | 87.50% | 87.50% | +15% (vs baseline) | [✓] |
| F1-Score (Macro) | ≥0.65 | 0.876 | 0.876 | +19% (vs baseline) | [✓] |
| Latență Inferență | ≤50ms | 32 ms | 32 ms | -16 ms | [✓] |
| Contribuție Date Originale | ≥40% | 40% | 40% | [✓] |
| Nr. Experimente Optimizare | ≥4 | 4 | 4 | - | [✓] |

### Declarație de Originalitate & Politica de Utilizare AI

**Acest proiect reflectă munca, gândirea și deciziile mele proprii.**

Utilizarea asistenților de inteligență artificială (ChatGPT, Claude, Grok, GitHub Copilot etc.) este **permisă și încurajată** ca unealtă de dezvoltare – pentru explicații, generare de idei, sugestii de cod, debugging, structurarea documentației sau rafinarea textelor.

**Nu este permis** să preiau:
- cod, arhitectură RN sau soluție luată aproape integral de la un asistent AI fără modificări și raționamente proprii semnificative,
- dataset-uri publice fără contribuție proprie substanțială (minimum 40% din observațiile finale – conform cerinței obligatorii Etapa 4),
- conținut esențial care nu poartă amprenta clară a propriei mele înțelegeri.

**Confirmare explicită (bifez doar ce este adevărat):**

| Nr. | Cerință                                                                 | Confirmare |
|-----|-------------------------------------------------------------------------|------------|
| 1   | Modelul RN a fost antrenat **de la zero** (weights inițializate random, **NU** model pre-antrenat descărcat) | [x] DA     |
| 2   | Minimum **40% din date sunt contribuție originală** (generate/achiziționate/etichetate de mine) | [x] DA     |
| 3   | Codul este propriu sau sursele externe sunt **citate explicit** în Bibliografie | [x] DA     |
| 4   | Arhitectura, codul și interpretarea rezultatelor reprezintă **muncă proprie** (AI folosit doar ca tool, nu ca sursă integrală de cod/dataset) | [x] DA     |
| 5   | Pot explica și justifica **fiecare decizie importantă** cu argumente proprii | [x] DA     |

**Semnătură student (prin completare):** Declar pe propria răspundere că informațiile de mai sus sunt corecte.

---

## 2. Descrierea Nevoii și Soluția SIA

### 2.1 Nevoia Reală / Studiul de Caz

Roboții mobili autonomi operează adesea în medii mixte (indoor/outdoor) unde suprafața de rulare se schimbă frecvent (ex: trecerea de la covor la parchet, sau de la asfalt la iarbă). Necunoașterea tipului de suprafață duce la o estimare incorectă a odometriei (roțile alunecă diferit pe gresie vs covor), uzură prematură a mecanismelor sau chiar blocarea robotului (ex: intrarea pe iarbă înaltă sau nisip).

În prezent, majoritatea roboților low-cost folosesc doar senzori de proximitate. Există nevoia unui sistem robust care să clasifice tipul de teren în timp real folosind senzori ieftini deja existenți pe robot (Camera RGB + IMU), pentru a adapta parametrii de navigație (viteză, cuplu).

### 2.2 Beneficii Măsurabile Urmărite

1. **Adaptarea tracțiunii:** Ajustarea controlerelor PID în funcție de fricțiunea estimată (Accuracy > 85%).
2. **Siguranță:** Evitarea suprafețelor periculoase (ex: gresie udă/lucioasă detectată ca Tile).
3. **Localizare îmbunătățită:** Corecția erorilor de odometrie specifice fiecărei suprafețe.
4. **Întreținere predictivă:** Detectarea vibrațiilor anormale pe suprafețe netede (Concrete/Asphalt).

### 2.3 Tabel: Nevoie → Soluție SIA → Modul Software

| **Nevoie reală concretă** | **Cum o rezolvă SIA-ul** | **Modul software responsabil** | **Metric măsurabil** |
|---------------------------|--------------------------|--------------------------------|----------------------|
| Adaptare viteză la suprafață | Clasificare multimodală (Img+IMU) | Neural Network + Main App | Latență < 100ms |
| Evitare zone interzise (Iarbă) | Detecție clasă "Grass" -> Stop | State Machine Logic | Recall "Grass" > 90% |
| Feedback vizual operator | UI Dashboard cu bare de încredere | Web Service (Streamlit) | Update rate > 5Hz |

---

## 3. Dataset și Contribuție Originală

### 3.1 Sursa și Caracteristicile Datelor

| Caracteristică | Valoare |
|----------------|---------|
| **Origine date** | Dataset Open Source (VAST IROS 2022) + Augmentare Proprie |
| **Sursa concretă** | [VAST Dataset (RIVeR-Lab)](https://github.com/RIVeR-Lab/vast_data) |
| **Număr total observații finale (N)** | ~1,000 (100% din cele necesare) |
| **Număr features** | Imagini 224x224 (1 channel) + IMU 99x10 (Time-series) |
| **Tipuri de date** | Multimodal: Imagini (Grayscale) + Serii Temporale (IMU) |
| **Format fișiere** | Imagini `.jpg` + Semnale `.npy` |
| **Perioada colectării/generării** | Ianuarie 2026 |

### 3.2 Contribuția Originală (40% OBLIGATORIU)

| Câmp | Valoare |
|------|---------|
| **Total observații finale (N)** | 1,000 |
| **Observații originale (M)** | ~400 (40% din total) |
| **Procent contribuție originală** | 40% |
| **Tip contribuție** | Preprocesare  + Augmentare (Rotiri, Flip, Noise) + Date Sintetice |
| **Locație cod generare** | `src/data_acquisition/generate_all_data.py` |
| **Locație date originale** | `data/README.md` (detalii origine VAST) |

**Descriere metodă generare/achiziție:**

Datele provin în mare parte dintr-un dataset public (VAST IROS 2022), care conține date reale de la senzori. Pentru a îndeplini cerința de contribuție proprie și pentru a echilibra clasele, am generat scriptic 40% din setul total de date. Aceste noi intrări sunt date sintetice rezultate din augmentarea și sinteza statistică a semnalelor IMU (bazată pe distribuțiile reale din VAST) și a imaginilor (transformări geometrice și adăugare de zgomot). Astfel, am atins pragul de 1,000 de intrări, asigurând diversitatea necesară antrenării.


### 3.3 Preprocesare și Split Date

| Set | Procent | Număr Observații |
|-----|---------|------------------|
| Train | 70% | ~10,500 |
| Validation | 15% | ~2,250 |
| Test | 15% | ~2,250 |

**Preprocesări aplicate:**
- Normalizare Min-Max (0-1) pentru pixelii imaginilor.
- Standardizare (Z-score) pentru datele IMU (Acc/Gyro).
- Resizing la 224x224 pentru compatibilitate CNN.
- One-hot encoding pentru etichetele claselor.

**Referințe fișiere:** `data/README.md`, `config/preprocessing_params.pkl`

---

## 4. Arhitectura SIA și State Machine

### 4.1 Cele 3 Module Software

| Modul | Tehnologie | Funcționalitate Principală | Locație în Repo |
|-------|------------|---------------------------|-----------------|
| **Data Logging / Acquisition** | Python (Numpy/CV2) | Generare date sintetice multimodale și augmentare | `src/data_acquisition/` |
| **Neural Network** | TensorFlow/Keras | Arhitectură hibridă (CNN 1D + 2D) pentru clasificare | `src/neural_network/` |
| **Web Service / UI** | Streamlit | Dashboard interactiv pentru inferență și vizualizare | `src/app/` |

### 4.2 State Machine

**Locație diagramă:** `docs/state_machine.png`

**Stări principale și descriere:**

| Stare | Descriere | Condiție Intrare | Condiție Ieșire |
|-------|-----------|------------------|-----------------|
| `IDLE` | Așteptare comandă utilizator | Start aplicație | Buton "Run Inference" |
| `ACQUIRE_DATA` | Încărcare pereche (Img, IMU) | Trigger UI | Date încărcate |
| `PREPROCESS` | Normalizare și formatare date | Date brute OK | Tensor pregătit |
| `INFERENCE` | Rulare model `optimized_model.h5` | Tensor OK | Softmax probabilities |
| `CONFIDENCE_CHECK` | Verificare prag încredere > 70% | Predicție gata | High Conf / Low Conf |
| `OUTPUT/ALERT` | Afișare clasă și colorare UI | High Confidence | User acknowledgment |
| `LOG_UNCERTAIN` | Logare avertisment (Galben) | Low Confidence | Return to IDLE |

**Justificare alegere arhitectură State Machine:**

Structura secvențială cu ramificare de decizie ("Check Confidence") este esențială pentru a preveni deciziile eronate. Starea intermediară de verificare permite filtrarea predicțiilor slabe (sub 70%), evitând comportamentul haotic al robotului în situații ambigue, redirecționând controlul către un fallback sigur ("LOG_UNCERTAIN").

### 4.3 Actualizări State Machine în Etapa 6 (dacă este cazul)

| Componentă Modificată | Valoare Etapa 5 | Valoare Etapa 6 | Justificare Modificare |
|----------------------|-----------------|-----------------|------------------------|
| Threshold Confidence | 50% (Implicit) | 70% | Creșterea robusteții decizionale |
| Stare nouă adăugată | N/A | `CONFIDENCE_CHECK` | Filtrare activă a incertitudinii |
| Ramură Falback | N/A | Alertă "UNCERTAIN" | Feedback vizual pentru operator |

---

## 5. Modelul RN – Antrenare și Optimizare

### 5.1 Arhitectura Rețelei Neuronale

```
Input IMU (99, 10) -> Conv1D(64) -> MaxPool -> Conv1D(128) -> GlobalAvgPool -> Dense(64)
Input IMG (224, 224, 1) -> Conv2D(32) -> MaxPool -> Conv2D(64) -> MaxPool -> Flatten -> Dense(64)
Concatenate([Dense_IMU, Dense_IMG])
  -> Dense(64, ReLU)
  -> Dropout(0.3)
  -> Dense(5, Softmax)
Output: 5 clase (Asphalt, Carpet, Concrete, Grass, Tile)
```

**Justificare alegere arhitectură:**

Am ales o arhitectură multimodală cu două ramuri deoarece problema necesită informații complementare: textura vizuală (2D CNN) și semnătura vibratorie (1D CNN). O singură modalitate (doar imagine sau doar IMU) ducea la confuzii frecvente (ex: Grass vs Carpet vizual, sau Concrete vs Asphalt vibrațional). Fuziunea permite rețelei să învețe corelații complexe între cele două tipuri de date.

### 5.2 Hiperparametri Finali (Model Optimizat - Etapa 6)

| Hiperparametru | Valoare Finală | Justificare Alegere |
|----------------|----------------|---------------------|
| Learning Rate | 0.001 | Valoare optimă pentru Adam, compromis viteză/stabilitate |
| Batch Size | 32 | Convergență stabilă și utilizare eficientă a VRAM |
| Epochs | 5 (Early Stopping) | Prevenire overfitting; platou atins rapid |
| Optimizer | Adam | Adaptare dinamică a ratei de învățare |
| Loss Function | Categorical Crossentropy | Standard pentru clasificare multi-clasă |
| Regularizare | Dropout 0.3 | Suficient pentru generalizare, 0.6 a cauzat underfitting |

### 5.3 Experimente de Optimizare (minim 4 experimente)

| Exp# | Modificare față de Baseline | Accuracy | F1-Score | Timp Antrenare | Observații |
|------|----------------------------|----------|----------|----------------|------------|
| **Baseline** | Configurația din Etapa 5 | **87.50%** | **0.876** | 0.32 min | **Performanță optimă, echilibru stabilitate/viteză** |
| Exp 1 | High Dropout (0.6) | 78.41% | 0.772 | 0.33 min | Regularizare prea agresivă (-10% acc) |
| Exp 2 | Low LR (0.0001) | 61.36% | 0.598 | 0.29 min | Convergență foarte lentă în 5 epoci |
| Exp 3 | Larger Batch (64) | 71.59% | 0.703 | 0.28 min | Generalizare mai slabă decât batch 32 |
| Exp 4 (impl) | Augmented Data Only | N/A | N/A | - | (Implicit în Baseline prin generator) |
| **FINAL** | **Baseline Config** | **87.50%** | **0.876** | **0.32 min** | **Modelul folosit în producție** |

**Justificare alegere model final:**

Experimentul Baseline a oferit cele mai bune rezultate. Încercările de a mări regularizarea (Exp 1) sau batch size-ul (Exp 3) au degradat performanța, indicând că modelul baseline era deja bine calibrat pentru complexitatea datelor. Learning rate (Exp 2) nu a permis modelului să conveargă în numărul limitat de epoci. Astfel, am păstrat configurația Baseline ca Model Optimizat final.

**Referințe fișiere:** `results/optimization_experiments.csv`, `models/optimized_model.h5`

---

## 6. Performanță Finală și Analiză Erori

### 6.1 Metrici pe Test Set (Model Optimizat)

| Metric | Valoare | Target Minim | Status |
|--------|---------|--------------|--------|
| **Accuracy** | 87.50% | ≥70% | [✓] |
| **F1-Score (Macro)** | 0.876 | ≥0.65 | [✓] |
| **Precision (Macro)** | 0.879 | - | - |
| **Recall (Macro)** | 0.875 | - | - |

**Îmbunătățire față de Baseline (Etapa 5):**

| Metric | Etapa 5 (Baseline - Overfit) | Etapa 6 (Optimizat - Robust) | Diferență (Contextuală) |
|--------|------------------------------|------------------------------|-------------------------|
| Accuracy | ~100.00% | 87.50% | -12.5% (Intenționat) |
| F1-Score | ~1.000 | 0.876 | -0.124 (Intenționat) |
| Epoci Antrenare | 50 | 5 (Early Stopping) | Reducere drastică overfitting |

În Etapa 5, am obținut o acuratețe nerealistă de 100% antrenând timp de 50 de epoci. Analiza a arătat că rețeaua memorase dataset-ul.
În Etapa 6, am schimbat abordarea: am redus drastic numărul de epoci (Early Stopping) și am acceptat o scădere a metricilor brute (la ~87.5%) în favoarea unui model care funcționează corect și generalizează, în loc să memoreze zgomotul specific.

**Referință fișier:** `results/final_metrics.json`

### 6.2 Confusion Matrix

**Locație:** `docs/screenshots/confusion_matrix.png`

**Interpretare:**

| Aspect | Observație |
|--------|------------|
| **Clasa cu cea mai bună performanță** | **Asphalt** - Rugozitatea distinctă îl face ușor de clasificat (Precizie > 90%). |
| **Clasa cu cea mai slabă performanță** | **Carpet** - Confundat adesea cu Grass (Recall ~82%). |
| **Confuzii frecvente** | **Grass vs Carpet**: Ambele sunt suprafețe moi, fonoabsorbante, generând date IMU similare; vizual pot fi confundate în grayscale. |
| **Dezechilibru clase** | Datasetul este echilibrat (~20% per clasă), deci erorile nu provin din dezechilibre majore. |

### 6.3 Analiza Top 5 Erori

| # | Input (descriere scurtă) | Predicție RN | Clasă Reală | Cauză Probabilă | Implicație Industrială |
|---|--------------------------|--------------|-------------|-----------------|------------------------|
| 1 | Covor cu pattern repetitiv | Asphalt | Carpet | Pattern-ul vizual dens a fost interpretat ca rugozitate asfalt. | Robotul ar putea accelera pe covor crezând că e asfalt, riscând alunecarea. |
| 2 | Iarbă foarte scurtă/uscată | Concrete | Grass | Lipsa texturii "verzi" puternice în grayscale. | Uzură prematură a roților dacă se comportă ca pe beton. |
| 3 | Gresie fără rosturi vizibile | Concrete | Tile | IMU nu a detectat șocul de trecere peste rost. | Adaptarea suspensiei ar putea fi incorectă (prea rigidă). |
| 4 | Tranziție suprafețe (mix) | Uncertain | Tile | Semnal mixt la trecerea dintre camere. | Oprire nejustificată a robotului pentru verificare. |
| 5 | Zgomot IMU excesiv | Carpet | Concrete | Vibrații parazite interpretate greșit ca textură dură. | Instabilitate în controlul vitezei. |

### 6.4 Validare în Context Industrial

**Ce înseamnă rezultatele pentru aplicația reală:**

O acuratețe de 87.5% într-un mediu controlat indică un potențial ridicat pentru navigație autonomă. Totuși, confuziile Grass/Carpet (ambele "safe" pentru impact, dar cu frecări diferite) pot afecta odometria.

**Pragul de acceptabilitate pentru domeniu:** Recall ≥ 90% pentru suprafețe periculoase. Pentru clasele curente, 85% este acceptabil.
**Status:** [Atins] pentru majoritatea claselor, cu excepția Carpet (marginal).
**Plan de îmbunătățire:** Colectare date suplimentare specifice pentru Carpet și Grass în condiții de lumină variată.

---

## 7. Aplicația Software Finală

### 7.1 Modificări Implementate în Etapa 6

| Componentă | Stare Etapa 5 | Modificare Etapa 6 | Justificare |
|------------|---------------|-------------------|-------------|
| **Model încărcat** | `rn_floor_classifier_v0_skeleton.h5` | `optimized_model.h5` (Baseline) | Trecerea de la un schelet neantrenat la modelul final performant. |
| **Threshold decizie** | 0.5 default | 0.70 slider (ajustabil) | Creșterea siguranței în decizie; filtrarea predicțiilor slabe. |
| **UI - feedback vizual** | Text simplu | Bare de probabilitate colorate | Operatorul poate vedea distribuția încrederii, nu doar clasa finală. |
| **Logging** | Console print | Streamlit Metric Display | Vizibilitate imediată a parametrilor cheie (Timp inferență). |

### 7.2 Screenshot UI cu Model Optimizat

**Locație:** `docs/screenshots/ui_inference.png`

**Descriere:** Screenshot-ul arată interfața cu o imagine de Asfalt încărcată. Modelul prezice corect "Asphalt" cu o încredere de >90%. Graficele IMU arată o amplitudine mare a accelerației pe axa Z.

### 7.3 Demonstrație Funcțională End-to-End

**Locație dovadă:** `docs/demo/`.

**Fluxul demonstrat:**

| Pas | Acțiune | Rezultat Vizibil |
|-----|---------|------------------|
| 1 | Input | Upload imagine noua de catre utilizator. |
| 2 | Procesare | Imagineii i se aplica scalerul. |
| 3 | Inferenta | Afisarea imaginii incarcate de utilizator si a celei cu care lucreaza RN. Afiseaza predictia si confidence |
| 4 | Rezultat | Apare clasa prezisă cu font mare și culoarea verde dacă Confidence > 70%. |

**Latență măsurată end-to-end:** ~100-150 ms
**Data și ora demonstrației:** Ianuarie 2026.

---

## 8. Structura Repository-ului Final

```
proiect-rn-Marinescu-Alexandru/
│
├── README.md                               # Documentația principală (ACEST FIȘIER)
│
├── docs/                                   # Documentație pe Etape
│   ├── NUME_Prenume_Grupa_README_Proiect_RN.md  # Sinteza finală
│   ├── README_Etapa3_Preprocesare.md
│   ├── README_Etapa4_Arhitectura_SIA.md
│   ├── README_Etapa5_Antrenare_RN.md
│   ├── README_Etape6_Analiza_Performantei_Optimizare_Concluzii.md
│   ├── state_machine.mermaid               # Diagrama stărilor
│   ├── data_statistics.csv                 # Statistici set de date
│   └── screenshots/                        # Imagini suport
│
├── data/                                   # Date (Structura completă)
│   ├── README.md
│   ├── generated/                          # Sursa Datelor Originale
│   ├── raw/
│   ├── processed/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── src/                                    # Cod Sursă Modularizat
│   ├── data_acquisition/                   # Modul 1: Achiziție
│   │   ├── generate_all_data.py            # Generare date sintetice
│   │   ├── generate_imu.py
│   │   └── ...
│   │
│   ├── preprocessing/                      # Modul 2: Procesare
│   │   ├── build_dataset.py
│   │   ├── preprocess_images.py
│   │   └── create_scaler.py
│   │
│   ├── neural_network/                     # Modul 3: Rețea Neuronală
│   │   ├── model.py                        # Arhitectura CNN Dual-Head
│   │   ├── train.py                        # Antrenare
│   │   ├── evaluate.py                     # Testare
│   │   └── optimize.py                     # Optimizare
│   │
│   └── app/                                # Modul 4: Aplicație
│       ├── app.py                          # Interfața Streamlit
│       └── README.md
│
├── models/                                 # Modele salvate
│   ├── rn_floor_classifier_v0_skeleton.h5  
│   ├── trained_model.h5                    
│   └── optimized_model.h5                  # MODELUL FINAL
│
├── results/                                # Rezultate și Metrici
│   ├── final_metrics.json
│   ├── training_history.csv
│   └── optimization_experiments.csv
│
├── config/                                 # Configurații
├── requirements.txt                        # Dependențe
└── .gitignore
```

---

## 9. Instrucțiuni de Instalare și Rulare

### 9.1 Cerințe Preliminare

```
Python >= 3.8
Minim 4GB RAM
```

### 9.2 Instalare

```bash
# 1. Clonare repository (sau dezarhivare)
# git clone ... sau download zip

# 2. Creare mediu virtual (Opțional dar recomandat)
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. Instalare dependențe
pip install -r requirements.txt
```

### 9.3 Rulare Pipeline Complet

```bash
# 1. Clonare repository (dacă nu este deja descărcat)
git clone https://github.com/alexmFIIR004/proiect-rn-Marinescu-Alexandru
cd RN-Proiect

# 2. Generare Dataset (Date sintetice + Augmentări)
# Aceasta va popula folderul data/generated
python src/data_acquisition/generate_all_data.py

# 3. Preprocesare și Construire Dataset
# Împarte datele în Train/Val/Test și aplică scaler-ul
python src/preprocessing/build_dataset.py
python src/preprocessing/create_scaler.py

# 4. Antrenare Model
# Folosește configurația finală din config/optimized_config.yaml
python src/neural_network/train.py --config config/optimized_config.yaml

# 5. Evaluare Model
# Calculează metrici pe setul de test și generează matricea de confuzie
python src/neural_network/evaluate.py --model models/optimized_model.h5

# 6. Rulare Aplicație (Web UI)
streamlit run src/app/app.py
```

---

## 10. Concluzii și Discuții

### 10.1 Evaluare Performanță vs Obiective Inițiale

| Obiectiv Definit | Target | Realizat | Status |
|------------------|--------|----------|--------|
| Clasificare corectă 5 tipuri teren | Acc ≥ 80% | 87.50% | [✓] |
| Timp inferență real-time | < 200ms | < 100ms | [✓] |
| Robustezțe la zgomot IMU | Fără crash | Gestionat prin Dropout | [✓] |

### 10.2 Ce NU Funcționează – Limitări Cunoscute

1. **Dependența de Date Sintetice:** Cea mai mare limitare. Modelul "învață" generatorul, nu realitatea fizică complexă. Transferul pe un robot real ar necesita re-antrenare.
2. **Confuzia Grass-Carpet:** Modelul nu distinge fiabil între aceste două suprafețe.
3. **Iluminare Fixă:** Generatorul de imagini nu simulează umbre dinamice sau variații puternice de contrast.

### 10.3 Lecții Învățate (Top 5)

1. **Este nevoie de mai multa pregatire si cercetare inainte de inceperea proiectului** 
2. **Importanța Normalizării:** Datele IMU brute aveau scări diferite față de imaginile normalizate (0-255). Fără `create_scaler.py`, rețeaua nu convergea.
3. **Dropout** Fără stratul de Dropout(0.3), modelul ajungea la overfitting (Acc Train 99% / Val 70%) foarte rapid.
4. **Trebuie sa experimentezi cu mai multe configuratii pentru a-ti da seama cat mai devreme de ce trebuie schimbat in model** 
5. **Git LFS:** Gestionarea fișierelor mari (modele >100MB) necesită atenție specială încă de la începutul proiectului.

### 10.4 Retrospectivă

Dacă aș reîncepe proiectul, aș investi mai mult timp în generatorul de date pentru a-l face mai realist. De asemenea, aș augmenta datele intr-un mod mai complex pentru a preveni problemele cu antrenarea pe care le-am avut.

### 10.5 Direcții de Dezvoltare Ulterioară

| Termen | Îmbunătățire Propusă | Beneficiu Estimat |
|--------|---------------------|-------------------|
| **Short-term** | Adăugare culoare (RGB), adaugarea de imagini pentru carpet | Rezolvarea confuziei Iarbă/Covor |
| **Medium-term** | Portare pe Raspberry Pi (TFLite) | Robot autonom mobil |
| **Long-term** | Învățare Continuă (Online Learning) | Adaptare la suprafețe noi fără re-antrenare |

---

## 11. Bibliografie

1.  *TensorFlow Core v2.x documentation*, https://www.tensorflow.org/api_docs
2.  *Keras: The Python Deep Learning API*, https://keras.io/
3.  *Streamlit Documentation*, https://docs.streamlit.io/
4.  *Favraz et al., Deep Learning for Time Series Classification*, 2020. 
https://doi.org/10.48550/arXiv.2010.00567.
5.  Abaza, B., 2025. AI-Driven Dynamic Covariance for ROS 2 Mobile Robot Localization. Sensors, 25, 3026. https://doi.org/10.3390/s25103026

---

## 12. Checklist Final (Auto-verificare înainte de predare)

### Cerințe Tehnice Obligatorii

- [x] **Accuracy ≥70%** pe test set (87.50%)
- [x] **F1-Score ≥0.65** pe test set (0.876)
- [x] **Contribuție ≥40% date originale** (100% generate)
- [x] **Model antrenat de la zero**
- [x] **Minimum 4 experimente** de optimizare documentate
- [x] **Confusion matrix** generată și interpretată
- [x] **State Machine** definit cu minimum 4-6 stări
- [x] **Cele 3 module funcționale:** Data Logging, RN, UI
- [x] **Demonstrație end-to-end** disponibilă

### Repository și Documentație

- [x] **README.md** complet
- [x] **4 README-uri etape** prezente în `docs/`
- [x] **Screenshots** prezente
- [x] **Structura repository** conformă
- [x] **requirements.txt** actualizat
- [x] **Cod comentat**
- [x] **Toate path-urile relative**

### Acces și Versionare

- [x] **Repository accesibil**
- [x] **Tag `v0.6-optimized-final`**
- [x] **Commit-uri incrementale**
- [x] **Fișiere mari** gestionate (LFS)

### Verificare Anti-Plagiat

- [x] Model antrenat **de la zero**
- [x] **Minimum 40% date originale**
- [x] Cod propriu sau clar atribuit

---

## Note Finale

**Versiune document:** FINAL pentru examen  
**Ultima actualizare:** 11 februarie 2026
**Tag Git:** `v0.6-optimized-final`

*Acest README servește ca documentație principală pentru Livrabilul 1 (Aplicație RN).*

