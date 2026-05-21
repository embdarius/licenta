"""
Shared PMH (Past Medical History) vocabulary for Doctor v3 nurse — Change 1.

Maps PMH condition keywords (CHF, T2DM, COPD, etc.) to the 13 diagnosis_group
labels of Doctor v3 (catch-all "Symptoms, Signs, Ill-Defined" excluded — those
are presenting symptoms, not chronic conditions).

Used by:
    training/train_nurse_v3.py (training: PMH section of prior discharge notes
                                + free-text labels of prior ICD codes)
    tools/doctor_tool_v3.py    (inference: patient-reported chronic conditions
                                via the Nurse Agent's prior_history prompt)

Both sides must use the same vocabulary so training-time PMH flags and
inference-time PMH flags occupy the same feature space — the same parity
constraint that med_vocab.py enforces for medication flags.

Matching rules (parallel to med_vocab.py):
  - Keywords are matched with word boundaries to avoid spurious hits
    ("ckd" must not match "chickenpox", "afib" must not match "kafibola").
  - Plural forms via trailing optional `s` (e.g. "seizures" matches "seizure").
  - Negations like "no history of", "denies", "negative for", "ruled out",
    "no prior" are neutralized inside a 0-6 word window so they don't trip
    the bare keyword that follows.
"""

import re


# Stable list of the 13 diagnosis groups v3 uses as features (catch-all excluded).
# Order matches sorted(df["diagnosis_group"].unique()) at training time so the
# emitted feature columns are deterministic.
PMH_CATEGORIES = [
    "Blood and Blood-Forming Organs",
    "Circulatory",
    "Digestive",
    "Endocrine, Nutritional, Metabolic",
    "Genitourinary",
    "Infectious and Parasitic",
    "Injury and Poisoning",
    "Mental Disorders",
    "Musculoskeletal",
    "Nervous System and Sense Organs",
    "Other",
    "Respiratory",
    "Skin and Subcutaneous Tissue",
]


# ---------------------------------------------------------------------------
# PMH keyword -> diagnosis_group(s)
# ---------------------------------------------------------------------------
# A keyword value may be a single category string or a tuple of categories.
# Multi-category conditions: stroke is both Circulatory (cause) and Nervous
# System (effect); DVT is Circulatory but is also a Blood disorder context;
# cancer maps to "Other" (where Neoplasms lives in DIAGNOSIS_GROUP_MAP).
PMH_KEYWORD_MAP = {
    # ── Circulatory ──
    # Coronary / heart failure / structural heart disease
    "chf": "Circulatory",
    "congestive heart failure": "Circulatory",
    "heart failure": "Circulatory",
    "hfref": "Circulatory",
    "hfpef": "Circulatory",
    "systolic heart failure": "Circulatory",
    "diastolic heart failure": "Circulatory",
    "cardiomyopathy": "Circulatory",
    "coronary artery disease": "Circulatory",
    "cad": "Circulatory",
    "coronary disease": "Circulatory",
    "ischemic heart disease": "Circulatory",
    "myocardial infarction": "Circulatory",
    "prior mi": "Circulatory",
    "history of mi": "Circulatory",
    "stemi": "Circulatory",
    "nstemi": "Circulatory",
    "angina": "Circulatory",
    "cabg": "Circulatory",
    "coronary bypass": "Circulatory",
    "pci": "Circulatory",
    "stent": "Circulatory",
    "valvular heart disease": "Circulatory",
    "aortic stenosis": "Circulatory",
    "aortic regurgitation": "Circulatory",
    "mitral stenosis": "Circulatory",
    "mitral regurgitation": "Circulatory",
    "tricuspid regurgitation": "Circulatory",
    "pulmonary hypertension": "Circulatory",
    # Rhythm
    "atrial fibrillation": "Circulatory",
    "afib": "Circulatory",
    "a-fib": "Circulatory",
    "a fib": "Circulatory",
    "atrial flutter": "Circulatory",
    "aflutter": "Circulatory",
    "supraventricular tachycardia": "Circulatory",
    "svt": "Circulatory",
    "ventricular tachycardia": "Circulatory",
    "vtach": "Circulatory",
    "av block": "Circulatory",
    "heart block": "Circulatory",
    "sick sinus syndrome": "Circulatory",
    "pacemaker": "Circulatory",
    "icd implant": "Circulatory",
    # Hypertension
    "hypertension": "Circulatory",
    "htn": "Circulatory",
    "high blood pressure": "Circulatory",
    "hyperlipidemia": ("Circulatory", "Endocrine, Nutritional, Metabolic"),
    "hld": ("Circulatory", "Endocrine, Nutritional, Metabolic"),
    "hypercholesterolemia": ("Circulatory", "Endocrine, Nutritional, Metabolic"),
    "dyslipidemia": ("Circulatory", "Endocrine, Nutritional, Metabolic"),
    # Vascular
    "peripheral vascular disease": "Circulatory",
    "pvd": "Circulatory",
    "peripheral arterial disease": "Circulatory",
    "pad": "Circulatory",
    "aaa": "Circulatory",
    "aortic aneurysm": "Circulatory",
    "abdominal aortic aneurysm": "Circulatory",
    "thoracic aortic aneurysm": "Circulatory",
    "carotid stenosis": "Circulatory",
    "carotid artery disease": "Circulatory",
    # Stroke / CVA — also Nervous System
    "stroke": ("Circulatory", "Nervous System and Sense Organs"),
    "cva": ("Circulatory", "Nervous System and Sense Organs"),
    "cerebrovascular accident": ("Circulatory", "Nervous System and Sense Organs"),
    "transient ischemic attack": ("Circulatory", "Nervous System and Sense Organs"),
    "tia": ("Circulatory", "Nervous System and Sense Organs"),
    # Venous thromboembolism
    "dvt": "Circulatory",
    "deep vein thrombosis": "Circulatory",
    "pulmonary embolism": "Circulatory",
    "pe": "Circulatory",
    "vte": "Circulatory",
    "venous thromboembolism": "Circulatory",

    # ── Respiratory ──
    "copd": "Respiratory",
    "chronic obstructive pulmonary disease": "Respiratory",
    "emphysema": "Respiratory",
    "chronic bronchitis": "Respiratory",
    "asthma": "Respiratory",
    "obstructive sleep apnea": "Respiratory",
    "osa": "Respiratory",
    "sleep apnea": "Respiratory",
    "cpap": "Respiratory",
    "bipap": "Respiratory",
    "interstitial lung disease": "Respiratory",
    "ild": "Respiratory",
    "pulmonary fibrosis": "Respiratory",
    "idiopathic pulmonary fibrosis": "Respiratory",
    "ipf": "Respiratory",
    "sarcoidosis": "Respiratory",
    "bronchiectasis": "Respiratory",
    "cystic fibrosis": "Respiratory",
    "cf": "Respiratory",
    "chronic respiratory failure": "Respiratory",
    "home oxygen": "Respiratory",

    # ── Endocrine, Nutritional, Metabolic ──
    "diabetes": "Endocrine, Nutritional, Metabolic",
    "diabetes mellitus": "Endocrine, Nutritional, Metabolic",
    "dm": "Endocrine, Nutritional, Metabolic",
    "dmii": "Endocrine, Nutritional, Metabolic",
    "dmi": "Endocrine, Nutritional, Metabolic",
    "dm2": "Endocrine, Nutritional, Metabolic",
    "dm1": "Endocrine, Nutritional, Metabolic",
    "t2dm": "Endocrine, Nutritional, Metabolic",
    "t1dm": "Endocrine, Nutritional, Metabolic",
    "type 2 diabetes": "Endocrine, Nutritional, Metabolic",
    "type 1 diabetes": "Endocrine, Nutritional, Metabolic",
    "type ii diabetes": "Endocrine, Nutritional, Metabolic",
    "type i diabetes": "Endocrine, Nutritional, Metabolic",
    "iddm": "Endocrine, Nutritional, Metabolic",
    "niddm": "Endocrine, Nutritional, Metabolic",
    "hypothyroidism": "Endocrine, Nutritional, Metabolic",
    "hyperthyroidism": "Endocrine, Nutritional, Metabolic",
    "thyroid disease": "Endocrine, Nutritional, Metabolic",
    "graves": "Endocrine, Nutritional, Metabolic",
    "hashimoto": "Endocrine, Nutritional, Metabolic",
    "thyroidectomy": "Endocrine, Nutritional, Metabolic",
    "cushing": "Endocrine, Nutritional, Metabolic",
    "addison": "Endocrine, Nutritional, Metabolic",
    "adrenal insufficiency": "Endocrine, Nutritional, Metabolic",
    "obesity": "Endocrine, Nutritional, Metabolic",
    "morbid obesity": "Endocrine, Nutritional, Metabolic",
    "metabolic syndrome": "Endocrine, Nutritional, Metabolic",
    "gout": ("Endocrine, Nutritional, Metabolic", "Musculoskeletal"),
    "hyperparathyroidism": "Endocrine, Nutritional, Metabolic",
    "hypoparathyroidism": "Endocrine, Nutritional, Metabolic",
    "pituitary": "Endocrine, Nutritional, Metabolic",
    "malnutrition": "Endocrine, Nutritional, Metabolic",
    "vitamin d deficiency": "Endocrine, Nutritional, Metabolic",
    "b12 deficiency": ("Endocrine, Nutritional, Metabolic",
                        "Blood and Blood-Forming Organs"),

    # ── Mental Disorders ──
    "depression": "Mental Disorders",
    "major depressive disorder": "Mental Disorders",
    "mdd": "Mental Disorders",
    "anxiety": "Mental Disorders",
    "generalized anxiety disorder": "Mental Disorders",
    "gad": "Mental Disorders",
    "panic disorder": "Mental Disorders",
    "panic attacks": "Mental Disorders",
    "bipolar": "Mental Disorders",
    "bipolar disorder": "Mental Disorders",
    "schizophrenia": "Mental Disorders",
    "schizoaffective": "Mental Disorders",
    "ptsd": "Mental Disorders",
    "post-traumatic stress": "Mental Disorders",
    "post traumatic stress": "Mental Disorders",
    "ocd": "Mental Disorders",
    "obsessive compulsive": "Mental Disorders",
    "eating disorder": "Mental Disorders",
    "anorexia": "Mental Disorders",
    "bulimia": "Mental Disorders",
    "substance abuse": "Mental Disorders",
    "substance use disorder": "Mental Disorders",
    "sud": "Mental Disorders",
    "alcohol use disorder": "Mental Disorders",
    "alcohol abuse": "Mental Disorders",
    "alcoholism": "Mental Disorders",
    "etoh abuse": "Mental Disorders",
    "opioid use disorder": "Mental Disorders",
    "iv drug use": "Mental Disorders",
    "ivdu": "Mental Disorders",
    "polysubstance": "Mental Disorders",
    "dementia": ("Mental Disorders", "Nervous System and Sense Organs"),
    "alzheimer": "Nervous System and Sense Organs",
    "adhd": "Mental Disorders",
    "autism": "Mental Disorders",
    "personality disorder": "Mental Disorders",
    "suicide attempt": "Mental Disorders",
    "self harm": "Mental Disorders",
    "deliberate self harm": "Mental Disorders",

    # ── Genitourinary ──
    "chronic kidney disease": "Genitourinary",
    "ckd": "Genitourinary",
    "esrd": "Genitourinary",
    "end-stage renal disease": "Genitourinary",
    "end stage renal disease": "Genitourinary",
    "renal failure": "Genitourinary",
    "kidney failure": "Genitourinary",
    "hemodialysis": "Genitourinary",
    "peritoneal dialysis": "Genitourinary",
    "dialysis": "Genitourinary",
    "nephrolithiasis": "Genitourinary",
    "kidney stones": "Genitourinary",
    "renal stones": "Genitourinary",
    "polycystic kidney": "Genitourinary",
    "pkd": "Genitourinary",
    "iga nephropathy": "Genitourinary",
    "nephrotic syndrome": "Genitourinary",
    "glomerulonephritis": "Genitourinary",
    "bph": "Genitourinary",
    "benign prostatic hyperplasia": "Genitourinary",
    "prostate cancer": ("Genitourinary", "Other"),
    "prostatitis": "Genitourinary",
    "recurrent uti": "Genitourinary",
    "recurrent utis": "Genitourinary",
    "urinary tract infection": "Genitourinary",
    "neurogenic bladder": "Genitourinary",
    "urinary incontinence": "Genitourinary",
    "endometriosis": "Genitourinary",
    "ovarian cyst": "Genitourinary",
    "fibroids": "Genitourinary",
    "uterine fibroid": "Genitourinary",
    "renal transplant": "Genitourinary",
    "kidney transplant": "Genitourinary",

    # ── Digestive ──
    "gerd": "Digestive",
    "gastroesophageal reflux": "Digestive",
    "reflux disease": "Digestive",
    "peptic ulcer disease": "Digestive",
    "pud": "Digestive",
    "peptic ulcer": "Digestive",
    "gastritis": "Digestive",
    "gi bleed": "Digestive",
    "gib": "Digestive",
    "upper gi bleed": "Digestive",
    "lower gi bleed": "Digestive",
    "ugib": "Digestive",
    "lgib": "Digestive",
    "diverticulosis": "Digestive",
    "diverticulitis": "Digestive",
    "inflammatory bowel disease": "Digestive",
    "ibd": "Digestive",
    "crohn": "Digestive",
    "ulcerative colitis": "Digestive",
    "celiac": "Digestive",
    "ibs": "Digestive",
    "irritable bowel syndrome": "Digestive",
    "cirrhosis": "Digestive",
    "liver cirrhosis": "Digestive",
    "alcoholic cirrhosis": "Digestive",
    "hepatic cirrhosis": "Digestive",
    "fatty liver": "Digestive",
    "nafld": "Digestive",
    "nash": "Digestive",
    "hepatic failure": "Digestive",
    "liver failure": "Digestive",
    "pancreatitis": "Digestive",
    "chronic pancreatitis": "Digestive",
    "cholelithiasis": "Digestive",
    "cholecystitis": "Digestive",
    "gallstones": "Digestive",
    "biliary cirrhosis": "Digestive",
    "esophageal varices": "Digestive",
    "barrett": "Digestive",
    "achalasia": "Digestive",
    "bowel obstruction": "Digestive",
    "sbo": "Digestive",
    "colostomy": "Digestive",
    "ileostomy": "Digestive",
    # Hepatitis B/C are Infectious + Digestive (chronic liver dz)
    "hepatitis b": ("Digestive", "Infectious and Parasitic"),
    "hbv": ("Digestive", "Infectious and Parasitic"),
    "hepatitis c": ("Digestive", "Infectious and Parasitic"),
    "hcv": ("Digestive", "Infectious and Parasitic"),
    "chronic hepatitis": ("Digestive", "Infectious and Parasitic"),

    # ── Musculoskeletal ──
    "osteoarthritis": "Musculoskeletal",
    "oa": "Musculoskeletal",
    "rheumatoid arthritis": "Musculoskeletal",
    "ra": "Musculoskeletal",
    "psoriatic arthritis": "Musculoskeletal",
    "ankylosing spondylitis": "Musculoskeletal",
    "fibromyalgia": "Musculoskeletal",
    "lupus": "Musculoskeletal",
    "sle": "Musculoskeletal",
    "systemic lupus": "Musculoskeletal",
    "polymyalgia rheumatica": "Musculoskeletal",
    "pmr": "Musculoskeletal",
    "scleroderma": "Musculoskeletal",
    "sjogren": "Musculoskeletal",
    "spinal stenosis": "Musculoskeletal",
    "degenerative disc disease": "Musculoskeletal",
    "ddd": "Musculoskeletal",
    "herniated disc": "Musculoskeletal",
    "scoliosis": "Musculoskeletal",
    "kyphosis": "Musculoskeletal",
    "compression fracture": ("Musculoskeletal", "Injury and Poisoning"),
    "osteoporosis": "Musculoskeletal",
    "osteopenia": "Musculoskeletal",

    # ── Skin and Subcutaneous Tissue ──
    "psoriasis": "Skin and Subcutaneous Tissue",
    "eczema": "Skin and Subcutaneous Tissue",
    "atopic dermatitis": "Skin and Subcutaneous Tissue",
    "rosacea": "Skin and Subcutaneous Tissue",
    "recurrent cellulitis": "Skin and Subcutaneous Tissue",
    "chronic ulcer": "Skin and Subcutaneous Tissue",
    "venous stasis ulcer": "Skin and Subcutaneous Tissue",
    "decubitus ulcer": "Skin and Subcutaneous Tissue",
    "pressure ulcer": "Skin and Subcutaneous Tissue",
    "hidradenitis": "Skin and Subcutaneous Tissue",
    "vitiligo": "Skin and Subcutaneous Tissue",
    "mrsa colonization": ("Skin and Subcutaneous Tissue",
                           "Infectious and Parasitic"),

    # ── Nervous System and Sense Organs ──
    "seizure disorder": "Nervous System and Sense Organs",
    "seizures": "Nervous System and Sense Organs",
    "epilepsy": "Nervous System and Sense Organs",
    "migraine": "Nervous System and Sense Organs",
    "migraines": "Nervous System and Sense Organs",
    "parkinson": "Nervous System and Sense Organs",
    "parkinsons disease": "Nervous System and Sense Organs",
    "parkinson's disease": "Nervous System and Sense Organs",
    "multiple sclerosis": "Nervous System and Sense Organs",
    "ms": "Nervous System and Sense Organs",
    "myasthenia gravis": "Nervous System and Sense Organs",
    "als": "Nervous System and Sense Organs",
    "amyotrophic lateral sclerosis": "Nervous System and Sense Organs",
    "huntington": "Nervous System and Sense Organs",
    "neuropathy": "Nervous System and Sense Organs",
    "peripheral neuropathy": "Nervous System and Sense Organs",
    "trigeminal neuralgia": "Nervous System and Sense Organs",
    "bell's palsy": "Nervous System and Sense Organs",
    "cerebral palsy": "Nervous System and Sense Organs",
    "spinal cord injury": ("Nervous System and Sense Organs",
                            "Injury and Poisoning"),
    "guillain-barre": "Nervous System and Sense Organs",
    "vertigo": "Nervous System and Sense Organs",
    "meniere": "Nervous System and Sense Organs",
    "glaucoma": "Nervous System and Sense Organs",
    "macular degeneration": "Nervous System and Sense Organs",
    "diabetic retinopathy": ("Nervous System and Sense Organs",
                              "Endocrine, Nutritional, Metabolic"),
    "cataracts": "Nervous System and Sense Organs",
    "retinal detachment": "Nervous System and Sense Organs",
    "hearing loss": "Nervous System and Sense Organs",
    "tinnitus": "Nervous System and Sense Organs",

    # ── Blood and Blood-Forming Organs ──
    "anemia": "Blood and Blood-Forming Organs",
    "iron deficiency anemia": "Blood and Blood-Forming Organs",
    "ida": "Blood and Blood-Forming Organs",
    "anemia of chronic disease": "Blood and Blood-Forming Organs",
    "sickle cell": "Blood and Blood-Forming Organs",
    "sickle cell disease": "Blood and Blood-Forming Organs",
    "scd": "Blood and Blood-Forming Organs",
    "sickle cell crisis": "Blood and Blood-Forming Organs",
    "thalassemia": "Blood and Blood-Forming Organs",
    "hemophilia": "Blood and Blood-Forming Organs",
    "von willebrand": "Blood and Blood-Forming Organs",
    "itp": "Blood and Blood-Forming Organs",
    "immune thrombocytopenia": "Blood and Blood-Forming Organs",
    "thrombocytopenia": "Blood and Blood-Forming Organs",
    "hypercoagulable": "Blood and Blood-Forming Organs",
    "factor v leiden": "Blood and Blood-Forming Organs",
    "antiphospholipid": "Blood and Blood-Forming Organs",
    "myelodysplastic": ("Blood and Blood-Forming Organs", "Other"),
    "mds": ("Blood and Blood-Forming Organs", "Other"),
    "polycythemia": "Blood and Blood-Forming Organs",
    "essential thrombocytosis": "Blood and Blood-Forming Organs",

    # ── Infectious and Parasitic ──
    "hiv": "Infectious and Parasitic",
    "aids": "Infectious and Parasitic",
    "human immunodeficiency virus": "Infectious and Parasitic",
    "tuberculosis": "Infectious and Parasitic",
    "tb": "Infectious and Parasitic",
    "latent tb": "Infectious and Parasitic",
    "mrsa": "Infectious and Parasitic",
    "vre": "Infectious and Parasitic",
    "c diff": "Infectious and Parasitic",
    "c. diff": "Infectious and Parasitic",
    "clostridium difficile": "Infectious and Parasitic",
    "lyme disease": "Infectious and Parasitic",
    "syphilis": "Infectious and Parasitic",
    "herpes": "Infectious and Parasitic",
    "hsv": "Infectious and Parasitic",
    "shingles": "Infectious and Parasitic",
    "zoster": "Infectious and Parasitic",
    "recurrent infections": "Infectious and Parasitic",
    "immunocompromised": "Infectious and Parasitic",
    "neutropenia": ("Infectious and Parasitic",
                     "Blood and Blood-Forming Organs"),

    # ── Injury and Poisoning ──
    "tbi": "Injury and Poisoning",
    "traumatic brain injury": "Injury and Poisoning",
    "concussion": "Injury and Poisoning",
    "prior fracture": "Injury and Poisoning",
    "burn injury": "Injury and Poisoning",
    "amputation": "Injury and Poisoning",
    "trauma history": "Injury and Poisoning",
    "mva": "Injury and Poisoning",
    "motor vehicle accident": "Injury and Poisoning",
    "motor vehicle collision": "Injury and Poisoning",

    # ── Other (neoplasms / pregnancy / congenital — keep parity with
    # the "Other" bucket in DIAGNOSIS_GROUP_MAP) ──
    "cancer": "Other",
    "carcinoma": "Other",
    "malignancy": "Other",
    "neoplasm": "Other",
    "tumor": "Other",
    "metastatic": "Other",
    "metastasis": "Other",
    "lung cancer": "Other",
    "breast cancer": "Other",
    "colon cancer": "Other",
    "colorectal cancer": "Other",
    "pancreatic cancer": "Other",
    "ovarian cancer": "Other",
    "cervical cancer": "Other",
    "bladder cancer": "Other",
    "renal cell carcinoma": ("Other", "Genitourinary"),
    "melanoma": ("Other", "Skin and Subcutaneous Tissue"),
    "leukemia": ("Other", "Blood and Blood-Forming Organs"),
    "lymphoma": ("Other", "Blood and Blood-Forming Organs"),
    "multiple myeloma": ("Other", "Blood and Blood-Forming Organs"),
    "myeloma": ("Other", "Blood and Blood-Forming Organs"),
    "chemotherapy": "Other",
    "radiation therapy": "Other",
    "s/p chemo": "Other",
    "hospice": "Other",
    "ivf": "Other",
    "pregnancy": "Other",
    "g1p1": "Other",
    "g2p2": "Other",
    "previous c-section": "Other",
    "down syndrome": "Other",
    "trisomy": "Other",
    "congenital heart disease": ("Other", "Circulatory"),

    # ──────────────────────────────────────────────────────────────────
    # A2 VOCABULARY EXPANSION (2026-05-22)
    # ──────────────────────────────────────────────────────────────────
    # Targets gaps identified in the post-Change-1 audit (doctor-agent.md):
    # `s/p` surgical histories, abbreviation variants, brand-name drugs as
    # PMH condition proxies. Drug names are only matched inside the PMH
    # section (extract_pmh_section bounds them above the Medications block),
    # so they don't leak from Medications-on-Admission.
    # ──────────────────────────────────────────────────────────────────

    # ── Circulatory: s/p surgical histories and abbreviations ──
    "s/p cabg": "Circulatory",
    "s/p stent": "Circulatory",
    "s/p pci": "Circulatory",
    "s/p mi": "Circulatory",
    "s/p stemi": "Circulatory",
    "s/p nstemi": "Circulatory",
    "status post cabg": "Circulatory",
    "status post stent": "Circulatory",
    "status post mi": "Circulatory",
    "avr": "Circulatory",
    "aortic valve replacement": "Circulatory",
    "mvr": "Circulatory",
    "mitral valve replacement": "Circulatory",
    "valve replacement": "Circulatory",
    "valvuloplasty": "Circulatory",
    "aicd": "Circulatory",
    "automatic implantable defibrillator": "Circulatory",
    "ihd": "Circulatory",
    "ascvd": "Circulatory",
    "chf exacerbation": "Circulatory",
    "acute on chronic chf": "Circulatory",
    "decompensated chf": "Circulatory",

    # ── Circulatory: rhythm / conduction variants ──
    "paf": "Circulatory",
    "paroxysmal atrial fibrillation": "Circulatory",
    "chronic afib": "Circulatory",
    "persistent afib": "Circulatory",
    "permanent afib": "Circulatory",
    "chronic atrial fibrillation": "Circulatory",
    "first degree av block": "Circulatory",
    "second degree av block": "Circulatory",
    "third degree av block": "Circulatory",
    "complete heart block": "Circulatory",
    "chb": "Circulatory",
    "long qt": "Circulatory",
    "wolff-parkinson-white": "Circulatory",
    "wpw": "Circulatory",

    # ── Endocrine: diabetes variants ──
    "dm-ii": "Endocrine, Nutritional, Metabolic",
    "dm-i": "Endocrine, Nutritional, Metabolic",
    "dm type 2": "Endocrine, Nutritional, Metabolic",
    "dm type 1": "Endocrine, Nutritional, Metabolic",
    "diabetes type ii": "Endocrine, Nutritional, Metabolic",
    "diabetes type i": "Endocrine, Nutritional, Metabolic",
    "gestational diabetes": "Endocrine, Nutritional, Metabolic",
    "gdm": "Endocrine, Nutritional, Metabolic",
    "prediabetes": "Endocrine, Nutritional, Metabolic",
    "insulin dependent": "Endocrine, Nutritional, Metabolic",

    # ── Genitourinary: CKD staging ──
    "ckd 3": "Genitourinary",
    "ckd 4": "Genitourinary",
    "ckd 5": "Genitourinary",
    "ckd stage 3": "Genitourinary",
    "ckd stage 4": "Genitourinary",
    "ckd stage 5": "Genitourinary",
    "ckd iii": "Genitourinary",
    "ckd iv": "Genitourinary",
    "ckd v": "Genitourinary",
    "stage 3 ckd": "Genitourinary",
    "stage 4 ckd": "Genitourinary",
    "stage 5 ckd": "Genitourinary",
    "transplant recipient": "Genitourinary",

    # ── Respiratory: exacerbation patterns and home-O2 dependence ──
    "copd exacerbation": "Respiratory",
    "asthma exacerbation": "Respiratory",
    "pulmonary nodule": "Respiratory",
    "recurrent pneumonia": "Respiratory",
    "chronic cough": "Respiratory",
    "home o2": "Respiratory",
    "on oxygen": "Respiratory",

    # ── Digestive: chronic liver disease variants ──
    "chronic gerd": "Digestive",
    "liver disease": "Digestive",
    "chronic liver disease": "Digestive",
    "alcoholic liver disease": "Digestive",
    "non-alcoholic fatty liver disease": "Digestive",
    "esld": "Digestive",
    "end stage liver disease": "Digestive",
    "end-stage liver disease": "Digestive",

    # ── Mental: SI/HI, mood-disorder variants ──
    "mood disorder": "Mental Disorders",
    "borderline personality": "Mental Disorders",
    "bpd": "Mental Disorders",
    "suicidal ideation": "Mental Disorders",
    "chronic anxiety": "Mental Disorders",
    "chronic depression": "Mental Disorders",

    # ── Nervous System ──
    "chronic migraine": "Nervous System and Sense Organs",
    "chronic headache": "Nervous System and Sense Organs",
    "essential tremor": "Nervous System and Sense Organs",
    "restless legs": "Nervous System and Sense Organs",
    "rls": "Nervous System and Sense Organs",
    "diabetic neuropathy": ("Nervous System and Sense Organs",
                             "Endocrine, Nutritional, Metabolic"),

    # ── Musculoskeletal ──
    "chronic back pain": "Musculoskeletal",
    "chronic low back pain": "Musculoskeletal",
    "degenerative joint disease": "Musculoskeletal",
    "djd": "Musculoskeletal",
    "chronic pain syndrome": "Musculoskeletal",
    "chronic pain": "Musculoskeletal",

    # ── Injury (recurrent falls, prior trauma) ──
    "s/p fall": "Injury and Poisoning",
    "recurrent falls": "Injury and Poisoning",
    "history of trauma": "Injury and Poisoning",

    # ── Infectious (covid, hepA, on ART) ──
    "hepatitis a": "Infectious and Parasitic",
    "hav": "Infectious and Parasitic",
    "hiv positive": "Infectious and Parasitic",
    "on art": "Infectious and Parasitic",
    "on haart": "Infectious and Parasitic",
    "covid": "Infectious and Parasitic",
    "post-covid": "Infectious and Parasitic",
    "long covid": "Infectious and Parasitic",

    # ── Blood ──
    "hemolytic anemia": "Blood and Blood-Forming Organs",
    "aplastic anemia": "Blood and Blood-Forming Organs",
    "pernicious anemia": "Blood and Blood-Forming Organs",

    # ── Other (oncology / OB-GYN history) ──
    "history of cancer": "Other",
    "prior malignancy": "Other",
    "hospice care": "Other",
    "palliative care": "Other",
    "on chemo": "Other",
    "on chemotherapy": "Other",
    "prior c-section": "Other",
    "prior cesarean": "Other",
    "endometrial cancer": ("Other", "Genitourinary"),
    "thyroid cancer": ("Other", "Endocrine, Nutritional, Metabolic"),
    "head and neck cancer": "Other",

    # ── Brand-name drug proxies ──
    # Statins → cardiovascular + lipid disorder
    "atorvastatin": ("Circulatory", "Endocrine, Nutritional, Metabolic"),
    "lipitor": ("Circulatory", "Endocrine, Nutritional, Metabolic"),
    "simvastatin": ("Circulatory", "Endocrine, Nutritional, Metabolic"),
    "crestor": ("Circulatory", "Endocrine, Nutritional, Metabolic"),
    "rosuvastatin": ("Circulatory", "Endocrine, Nutritional, Metabolic"),
    # Diabetes
    "metformin": "Endocrine, Nutritional, Metabolic",
    "glipizide": "Endocrine, Nutritional, Metabolic",
    "insulin": "Endocrine, Nutritional, Metabolic",
    "ozempic": "Endocrine, Nutritional, Metabolic",
    "jardiance": "Endocrine, Nutritional, Metabolic",
    "januvia": "Endocrine, Nutritional, Metabolic",
    # Antihypertensives / cardiac
    "metoprolol": "Circulatory",
    "atenolol": "Circulatory",
    "carvedilol": "Circulatory",
    "lisinopril": "Circulatory",
    "losartan": "Circulatory",
    "amlodipine": "Circulatory",
    "hctz": "Circulatory",
    "hydrochlorothiazide": "Circulatory",
    "furosemide": "Circulatory",
    "lasix": "Circulatory",
    # Antiplatelet / anticoagulant
    "plavix": "Circulatory",
    "clopidogrel": "Circulatory",
    "eliquis": "Circulatory",
    "apixaban": "Circulatory",
    "xarelto": "Circulatory",
    "rivaroxaban": "Circulatory",
    "warfarin": "Circulatory",
    "coumadin": "Circulatory",
    # Psychiatric
    "prozac": "Mental Disorders",
    "fluoxetine": "Mental Disorders",
    "zoloft": "Mental Disorders",
    "sertraline": "Mental Disorders",
    "lexapro": "Mental Disorders",
    "escitalopram": "Mental Disorders",
    "citalopram": "Mental Disorders",
    "celexa": "Mental Disorders",
    "venlafaxine": "Mental Disorders",
    "effexor": "Mental Disorders",
    "duloxetine": "Mental Disorders",
    "cymbalta": "Mental Disorders",
    "trazodone": "Mental Disorders",
    "abilify": "Mental Disorders",
    "aripiprazole": "Mental Disorders",
    "olanzapine": "Mental Disorders",
    "zyprexa": "Mental Disorders",
    "quetiapine": "Mental Disorders",
    "seroquel": "Mental Disorders",
    "lithium carbonate": "Mental Disorders",
    # Antiepileptics / neuropathic pain agents (some dual-category)
    "valproate": ("Mental Disorders", "Nervous System and Sense Organs"),
    "depakote": ("Mental Disorders", "Nervous System and Sense Organs"),
    "lamictal": ("Mental Disorders", "Nervous System and Sense Organs"),
    "lamotrigine": ("Mental Disorders", "Nervous System and Sense Organs"),
    "keppra": "Nervous System and Sense Organs",
    "levetiracetam": "Nervous System and Sense Organs",
    "phenytoin": "Nervous System and Sense Organs",
    "dilantin": "Nervous System and Sense Organs",
    "gabapentin": ("Nervous System and Sense Organs", "Musculoskeletal"),
    "neurontin": ("Nervous System and Sense Organs", "Musculoskeletal"),
    "pregabalin": ("Nervous System and Sense Organs", "Musculoskeletal"),
    "lyrica": ("Nervous System and Sense Organs", "Musculoskeletal"),
    "levodopa": "Nervous System and Sense Organs",
    "sinemet": "Nervous System and Sense Organs",
    # GI (PPIs / H2 blockers)
    "omeprazole": "Digestive",
    "prilosec": "Digestive",
    "pantoprazole": "Digestive",
    "protonix": "Digestive",
    "esomeprazole": "Digestive",
    "nexium": "Digestive",
    "famotidine": "Digestive",
    "pepcid": "Digestive",
    # Thyroid
    "levothyroxine": "Endocrine, Nutritional, Metabolic",
    "synthroid": "Endocrine, Nutritional, Metabolic",
    "methimazole": "Endocrine, Nutritional, Metabolic",
    # Respiratory inhalers / leukotriene
    "albuterol": "Respiratory",
    "ventolin": "Respiratory",
    "advair": "Respiratory",
    "symbicort": "Respiratory",
    "spiriva": "Respiratory",
    "fluticasone": "Respiratory",
    "montelukast": "Respiratory",
    "singulair": "Respiratory",
}


# ---------------------------------------------------------------------------
# Compile one regex per category. Multi-word keys are matched as phrases
# (so "heart failure" matches the bigram, not "heart" alone).
# ---------------------------------------------------------------------------
# Build per-category keyword lists (inverting the {keyword: cat-or-tuple} map).
_cat_keywords: dict[str, list[str]] = {c: [] for c in PMH_CATEGORIES}
for kw, cat in PMH_KEYWORD_MAP.items():
    if isinstance(cat, str):
        _cat_keywords[cat].append(kw)
    else:
        for c in cat:
            _cat_keywords[c].append(kw)

# Longer phrases first so "heart failure" wins over "heart" if both were in
# the same category (defensive — we don't currently have such overlap).
for c in _cat_keywords:
    _cat_keywords[c].sort(key=lambda s: (-len(s), s))


def _kw_pattern(kw: str) -> str:
    """Word-boundary regex fragment for a single keyword.

    Special characters in the keyword (apostrophe, dot, hyphen, slash) are
    re.escaped. Trailing optional `s` handles plurals (seizures, fibroids,
    migraines, kidney stones).
    """
    return r'\b' + re.escape(kw) + r's?\b'


_CLASS_PATTERNS = {
    cat: re.compile(
        '|'.join(_kw_pattern(kw) for kw in kws),
        re.IGNORECASE,
    )
    for cat, kws in _cat_keywords.items()
    if kws
}


# Negation neutralizer. Within a 0-6 word window after a negation cue, replace
# the rest of the line with placeholder tokens so the bare keyword doesn't
# match. Common patterns in MIMIC PMH:
#   "no history of CHF"
#   "denies seizures"
#   "negative for HIV"
#   "ruled out for MI"
#   "no prior MI"
#   "without history of stroke"
_NEGATION_RE = re.compile(
    r'\b(?:no(?:\s+(?:prior|known|history|h/?o))?|denies|negative\s+for|'
    r'r/?o(?:\s+for)?|ruled\s+out(?:\s+for)?|without(?:\s+(?:history|h/?o))?|'
    r'absent\s+of|free\s+of)\b(?:\s+\S+){0,6}',
    re.IGNORECASE,
)


def flags_from_text(text) -> set:
    """Return the set of diagnosis_group flags matched in `text`.

    Negation phrases ("no history of X", "denies X") neutralize the next
    short window so the bare keyword doesn't trip. Match is case-insensitive
    with word boundaries.
    """
    if not isinstance(text, str) or not text:
        return set()

    # Replace negated windows with a non-matching placeholder string.
    neutralized = _NEGATION_RE.sub(' _NEG_ ', text.lower())

    flags = set()
    for cat, pattern in _CLASS_PATTERNS.items():
        if pattern.search(neutralized):
            flags.add(cat)
    return flags


# ---------------------------------------------------------------------------
# Discharge-note section extraction
# ---------------------------------------------------------------------------
# MIMIC-IV discharge notes have a stable set of section headers. The PMH
# section is bounded above by a header line containing "Past Medical
# History" / "PMH" / "PMHx", and bounded below by the next recognized
# header (Social History, Family History, Medications on Admission, etc.).
_PMH_SECTION_RE = re.compile(
    r'(?ims)'
    # Header line: "Past Medical History:", "PMH:", "PMHx:". The colon may
    # be followed by content on the same line (most common) or a newline
    # (also seen). `\s*` after `:` absorbs either.
    r'(?:^|\n)\s*(?:past\s+medical\s+history|pmh|pmhx)\s*:\s*'
    # Section body (non-greedy, DOTALL so it can span lines).
    r'(.+?)'
    # Stop at the next recognized header at line start, or end of text.
    r'(?=\n\s*(?:social\s+history|family\s+history|medications?\s+on\s+admission|'
    r'home\s+medications?|allergies|physical\s+exam(?:ination)?|'
    r'history\s+of\s+present\s+illness|hpi|chief\s+complaint|'
    r'review\s+of\s+systems|ros|brief\s+hospital\s+course|'
    r'assessment\s+and\s+plan|discharge\s+(?:diagnosis|medications?|condition|instructions?)|'
    r'pertinent\s+results)\s*:|\Z)',
)


def extract_pmh_section(note_text: str) -> str:
    """Return the PMH section of a MIMIC-IV discharge note, or '' if absent.

    Greedy enough to handle multi-line PMH (numbered/bulleted lists) but
    bounded by the next standard section header so we don't drag in Social
    History, etc.
    """
    if not isinstance(note_text, str) or not note_text:
        return ""
    m = _PMH_SECTION_RE.search(note_text)
    return m.group(1).strip() if m else ""
