"""
Shared medication vocabulary for Doctor v2 / v3 training and inference.

Provides a unified mapping of drug names and class keywords to 9 binary
medication category flags:
    has_cardiac_meds, has_diabetes_meds, has_psych_meds, has_respiratory_meds,
    has_opioid_meds, has_anticoagulant_meds, has_gi_meds, has_thyroid_meds,
    has_anticonvulsant_meds

Used by:
    training/train_nurse.py (training: medrecon.csv `name` + `etcdescription`)
    doctor_tool_v2 / v3     (inference: patient-reported medications)

Expanded based on the medication vocabulary audit (audit_med_vocab.py) that
found 39.3% of stays had mismatched flag sets between training and inference,
most severely `has_opioid_meds` at 44.4% Jaccard.

Matching rules:
  - Names are tokenized into alphabetic runs (handles hyphens, brackets, slashes
    in names like "fluticasone-salmeterol [advair diskus]").
  - Class keywords are matched with word boundaries to avoid spurious hits like
    "statin" in "nystatin" or "insulin" in "noninsulin".
  - "non-opioid" / "non-salicylate" negations are neutralized before matching.
"""

import re


# ---------------------------------------------------------------------------
# Drug name -> category. Generic + brand names. Any alphabetic token matching
# a key in this map flips the corresponding category flag.
# ---------------------------------------------------------------------------
# Values may be a single category string or a tuple of categories. Drugs
# that belong to two classes (e.g., benzodiazepines used as anticonvulsants,
# valproate/lamotrigine used for bipolar) must flag both, otherwise the
# training vocabulary (which reads the full etcdescription) will tag
# categories that inference cannot reach from the drug name alone.
DRUG_NAME_MAP = {
    # ── Cardiac ──
    # ACE inhibitors / ARBs
    "lisinopril": "has_cardiac_meds", "enalapril": "has_cardiac_meds",
    "ramipril": "has_cardiac_meds", "captopril": "has_cardiac_meds",
    "benazepril": "has_cardiac_meds", "quinapril": "has_cardiac_meds",
    "fosinopril": "has_cardiac_meds", "perindopril": "has_cardiac_meds",
    "losartan": "has_cardiac_meds", "valsartan": "has_cardiac_meds",
    "irbesartan": "has_cardiac_meds", "olmesartan": "has_cardiac_meds",
    "telmisartan": "has_cardiac_meds", "candesartan": "has_cardiac_meds",
    "azilsartan": "has_cardiac_meds", "eprosartan": "has_cardiac_meds",
    "cozaar": "has_cardiac_meds", "diovan": "has_cardiac_meds",
    "avapro": "has_cardiac_meds", "benicar": "has_cardiac_meds",
    # Statins / lipid-lowering
    "atorvastatin": "has_cardiac_meds", "simvastatin": "has_cardiac_meds",
    "rosuvastatin": "has_cardiac_meds", "pravastatin": "has_cardiac_meds",
    "lovastatin": "has_cardiac_meds", "fluvastatin": "has_cardiac_meds",
    "pitavastatin": "has_cardiac_meds",
    "lipitor": "has_cardiac_meds", "crestor": "has_cardiac_meds",
    "zocor": "has_cardiac_meds", "pravachol": "has_cardiac_meds",
    "fenofibrate": "has_cardiac_meds", "gemfibrozil": "has_cardiac_meds",
    "ezetimibe": "has_cardiac_meds", "zetia": "has_cardiac_meds",
    "niacin": "has_cardiac_meds",
    # Beta blockers
    "metoprolol": "has_cardiac_meds", "atenolol": "has_cardiac_meds",
    "propranolol": "has_cardiac_meds", "carvedilol": "has_cardiac_meds",
    "bisoprolol": "has_cardiac_meds", "nadolol": "has_cardiac_meds",
    "labetalol": "has_cardiac_meds", "sotalol": "has_cardiac_meds",
    "nebivolol": "has_cardiac_meds", "acebutolol": "has_cardiac_meds",
    "pindolol": "has_cardiac_meds",
    "toprol": "has_cardiac_meds", "coreg": "has_cardiac_meds",
    "tenormin": "has_cardiac_meds", "bystolic": "has_cardiac_meds",
    # Calcium channel blockers
    "amlodipine": "has_cardiac_meds", "nifedipine": "has_cardiac_meds",
    "diltiazem": "has_cardiac_meds", "verapamil": "has_cardiac_meds",
    "felodipine": "has_cardiac_meds", "nicardipine": "has_cardiac_meds",
    "isradipine": "has_cardiac_meds", "nisoldipine": "has_cardiac_meds",
    "norvasc": "has_cardiac_meds", "cardizem": "has_cardiac_meds",
    "plendil": "has_cardiac_meds",
    # Diuretics
    "hydrochlorothiazide": "has_cardiac_meds", "hctz": "has_cardiac_meds",
    "furosemide": "has_cardiac_meds", "torsemide": "has_cardiac_meds",
    "bumetanide": "has_cardiac_meds", "metolazone": "has_cardiac_meds",
    "spironolactone": "has_cardiac_meds", "chlorthalidone": "has_cardiac_meds",
    "triamterene": "has_cardiac_meds", "eplerenone": "has_cardiac_meds",
    "indapamide": "has_cardiac_meds", "amiloride": "has_cardiac_meds",
    "lasix": "has_cardiac_meds", "aldactone": "has_cardiac_meds",
    "demadex": "has_cardiac_meds",
    # Nitrates
    "nitroglycerin": "has_cardiac_meds", "isosorbide": "has_cardiac_meds",
    "nitrostat": "has_cardiac_meds", "imdur": "has_cardiac_meds",
    "nitro": "has_cardiac_meds",
    # Antiarrhythmics / other cardiac
    "amiodarone": "has_cardiac_meds", "digoxin": "has_cardiac_meds",
    "flecainide": "has_cardiac_meds", "dofetilide": "has_cardiac_meds",
    "ranolazine": "has_cardiac_meds", "ivabradine": "has_cardiac_meds",
    "dronedarone": "has_cardiac_meds", "mexiletine": "has_cardiac_meds",
    "propafenone": "has_cardiac_meds", "disopyramide": "has_cardiac_meds",
    "quinidine": "has_cardiac_meds",
    "ranexa": "has_cardiac_meds", "lanoxin": "has_cardiac_meds",
    # Alpha blockers / centrally acting
    "doxazosin": "has_cardiac_meds", "terazosin": "has_cardiac_meds",
    "prazosin": "has_cardiac_meds",
    "hydralazine": "has_cardiac_meds", "clonidine": "has_cardiac_meds",
    "minoxidil": "has_cardiac_meds", "methyldopa": "has_cardiac_meds",
    # Omega-3 (lipid adjunct) — training flags via antihyperlipidemic class
    "omega": "has_cardiac_meds",
    "lovaza": "has_cardiac_meds",  # prescription omega-3
    "krill": "has_cardiac_meds",
    # Bile-acid sequestrants (lipid-lowering)
    "cholestyramine": "has_cardiac_meds", "questran": "has_cardiac_meds",
    "colestipol": "has_cardiac_meds", "colestid": "has_cardiac_meds",
    "colesevelam": "has_cardiac_meds", "welchol": "has_cardiac_meds",
    # Timolol (beta-blocker, also used topically for glaucoma — training
    # flags cardiac regardless; matches its pharmacologic class)
    "timolol": "has_cardiac_meds",
    "cosopt": "has_cardiac_meds",  # dorzolamide-timolol eye drops
    # Carbonic anhydrase inhibitors used for hypertension/CHF
    "acetazolamide": "has_cardiac_meds", "diamox": "has_cardiac_meds",
    "methazolamide": "has_cardiac_meds",
    # Brand names for already-mapped generics
    "tricor": "has_cardiac_meds",  # fenofibrate
    "dyazide": "has_cardiac_meds",  # triamterene-hctz
    "maxzide": "has_cardiac_meds",
    "lopressor": "has_cardiac_meds",  # metoprolol

    # ── Diabetes ──
    "metformin": "has_diabetes_meds", "glucophage": "has_diabetes_meds",
    "insulin": "has_diabetes_meds", "lantus": "has_diabetes_meds",
    "humalog": "has_diabetes_meds", "novolog": "has_diabetes_meds",
    "humulin": "has_diabetes_meds", "novolin": "has_diabetes_meds",
    "levemir": "has_diabetes_meds", "tresiba": "has_diabetes_meds",
    "toujeo": "has_diabetes_meds", "basaglar": "has_diabetes_meds",
    "apidra": "has_diabetes_meds",
    "glipizide": "has_diabetes_meds", "glyburide": "has_diabetes_meds",
    "glimepiride": "has_diabetes_meds",
    "glucotrol": "has_diabetes_meds", "amaryl": "has_diabetes_meds",
    "januvia": "has_diabetes_meds", "sitagliptin": "has_diabetes_meds",
    "linagliptin": "has_diabetes_meds", "saxagliptin": "has_diabetes_meds",
    "alogliptin": "has_diabetes_meds",
    "tradjenta": "has_diabetes_meds", "onglyza": "has_diabetes_meds",
    "nesina": "has_diabetes_meds",
    "empagliflozin": "has_diabetes_meds", "dapagliflozin": "has_diabetes_meds",
    "canagliflozin": "has_diabetes_meds", "ertugliflozin": "has_diabetes_meds",
    "jardiance": "has_diabetes_meds", "farxiga": "has_diabetes_meds",
    "invokana": "has_diabetes_meds", "steglatro": "has_diabetes_meds",
    "ozempic": "has_diabetes_meds", "semaglutide": "has_diabetes_meds",
    "rybelsus": "has_diabetes_meds",
    "trulicity": "has_diabetes_meds", "dulaglutide": "has_diabetes_meds",
    "liraglutide": "has_diabetes_meds", "victoza": "has_diabetes_meds",
    "exenatide": "has_diabetes_meds", "byetta": "has_diabetes_meds",
    "bydureon": "has_diabetes_meds", "lixisenatide": "has_diabetes_meds",
    "pioglitazone": "has_diabetes_meds", "rosiglitazone": "has_diabetes_meds",
    "actos": "has_diabetes_meds", "avandia": "has_diabetes_meds",
    "repaglinide": "has_diabetes_meds", "nateglinide": "has_diabetes_meds",
    "prandin": "has_diabetes_meds", "starlix": "has_diabetes_meds",
    "acarbose": "has_diabetes_meds", "miglitol": "has_diabetes_meds",
    "precose": "has_diabetes_meds",
    "pramlintide": "has_diabetes_meds", "symlin": "has_diabetes_meds",

    # ── Psychiatric ──
    # SSRIs / SNRIs / atypicals
    "sertraline": "has_psych_meds", "zoloft": "has_psych_meds",
    "fluoxetine": "has_psych_meds", "prozac": "has_psych_meds",
    "escitalopram": "has_psych_meds", "lexapro": "has_psych_meds",
    "citalopram": "has_psych_meds", "celexa": "has_psych_meds",
    "paroxetine": "has_psych_meds", "paxil": "has_psych_meds",
    "fluvoxamine": "has_psych_meds", "luvox": "has_psych_meds",
    "vilazodone": "has_psych_meds", "viibryd": "has_psych_meds",
    "vortioxetine": "has_psych_meds", "trintellix": "has_psych_meds",
    "venlafaxine": "has_psych_meds", "effexor": "has_psych_meds",
    "duloxetine": "has_psych_meds", "cymbalta": "has_psych_meds",
    "desvenlafaxine": "has_psych_meds", "pristiq": "has_psych_meds",
    "levomilnacipran": "has_psych_meds", "fetzima": "has_psych_meds",
    "bupropion": "has_psych_meds", "wellbutrin": "has_psych_meds",
    "trazodone": "has_psych_meds", "desyrel": "has_psych_meds",
    "mirtazapine": "has_psych_meds", "remeron": "has_psych_meds",
    "nefazodone": "has_psych_meds",
    # Tricyclics
    "amitriptyline": "has_psych_meds", "nortriptyline": "has_psych_meds",
    "imipramine": "has_psych_meds", "doxepin": "has_psych_meds",
    "desipramine": "has_psych_meds", "clomipramine": "has_psych_meds",
    "elavil": "has_psych_meds", "pamelor": "has_psych_meds",
    "tofranil": "has_psych_meds", "anafranil": "has_psych_meds",
    # Benzodiazepines (clonazepam and diazepam are also formally classified
    # as anticonvulsants in MIMIC's etcdescription)
    "alprazolam": "has_psych_meds", "xanax": "has_psych_meds",
    "lorazepam": "has_psych_meds", "ativan": "has_psych_meds",
    "diazepam": ("has_psych_meds", "has_anticonvulsant_meds"),
    "valium": ("has_psych_meds", "has_anticonvulsant_meds"),
    "clonazepam": ("has_psych_meds", "has_anticonvulsant_meds"),
    "klonopin": ("has_psych_meds", "has_anticonvulsant_meds"),
    "temazepam": "has_psych_meds", "restoril": "has_psych_meds",
    "oxazepam": "has_psych_meds", "chlordiazepoxide": "has_psych_meds",
    "librium": "has_psych_meds",
    "midazolam": "has_psych_meds", "triazolam": "has_psych_meds",
    # Other anxiolytics
    "buspirone": "has_psych_meds", "buspar": "has_psych_meds",
    "hydroxyzine": "has_psych_meds", "vistaril": "has_psych_meds",
    "atarax": "has_psych_meds",
    # Antipsychotics
    "quetiapine": "has_psych_meds", "seroquel": "has_psych_meds",
    "risperidone": "has_psych_meds", "risperdal": "has_psych_meds",
    "olanzapine": "has_psych_meds", "zyprexa": "has_psych_meds",
    "aripiprazole": "has_psych_meds", "abilify": "has_psych_meds",
    "ziprasidone": "has_psych_meds", "geodon": "has_psych_meds",
    "clozapine": "has_psych_meds", "clozaril": "has_psych_meds",
    "paliperidone": "has_psych_meds", "invega": "has_psych_meds",
    "lurasidone": "has_psych_meds", "latuda": "has_psych_meds",
    "asenapine": "has_psych_meds", "saphris": "has_psych_meds",
    "cariprazine": "has_psych_meds", "vraylar": "has_psych_meds",
    "brexpiprazole": "has_psych_meds", "rexulti": "has_psych_meds",
    "iloperidone": "has_psych_meds",
    "haloperidol": "has_psych_meds", "haldol": "has_psych_meds",
    # Prochlorperazine/compazine are phenothiazine antipsychotics that are
    # also heavily used as antiemetics — flag both categories.
    "prochlorperazine": ("has_psych_meds", "has_gi_meds"),
    "compazine": ("has_psych_meds", "has_gi_meds"),
    "chlorpromazine": "has_psych_meds", "thorazine": "has_psych_meds",
    "fluphenazine": "has_psych_meds", "perphenazine": "has_psych_meds",
    "thioridazine": "has_psych_meds", "thiothixene": "has_psych_meds",
    "loxapine": "has_psych_meds", "molindone": "has_psych_meds",
    # Mood stabilizers (lithium; anticonvulsants used for bipolar are
    # also flagged under has_anticonvulsant_meds)
    "lithium": "has_psych_meds",
    # Stimulants / ADHD / narcolepsy
    "adderall": "has_psych_meds", "methylphenidate": "has_psych_meds",
    "ritalin": "has_psych_meds", "concerta": "has_psych_meds",
    "metadate": "has_psych_meds", "daytrana": "has_psych_meds",
    "vyvanse": "has_psych_meds", "lisdexamfetamine": "has_psych_meds",
    "dexmethylphenidate": "has_psych_meds", "focalin": "has_psych_meds",
    "dextroamphetamine": "has_psych_meds", "dexedrine": "has_psych_meds",
    "atomoxetine": "has_psych_meds", "strattera": "has_psych_meds",
    "guanfacine": "has_psych_meds", "intuniv": "has_psych_meds",
    "modafinil": "has_psych_meds", "provigil": "has_psych_meds",
    "armodafinil": "has_psych_meds", "nuvigil": "has_psych_meds",
    # Hypnotics
    "ambien": "has_psych_meds", "zolpidem": "has_psych_meds",
    "eszopiclone": "has_psych_meds", "lunesta": "has_psych_meds",
    "zaleplon": "has_psych_meds", "sonata": "has_psych_meds",
    "suvorexant": "has_psych_meds", "belsomra": "has_psych_meds",
    "ramelteon": "has_psych_meds", "rozerem": "has_psych_meds",

    # ── Respiratory ──
    "albuterol": "has_respiratory_meds", "salbutamol": "has_respiratory_meds",
    "proair": "has_respiratory_meds", "proventil": "has_respiratory_meds",
    "ventolin": "has_respiratory_meds",
    "levalbuterol": "has_respiratory_meds", "xopenex": "has_respiratory_meds",
    "fluticasone": "has_respiratory_meds", "flovent": "has_respiratory_meds",
    "advair": "has_respiratory_meds", "symbicort": "has_respiratory_meds",
    "budesonide": "has_respiratory_meds", "formoterol": "has_respiratory_meds",
    "salmeterol": "has_respiratory_meds",
    "mometasone": "has_respiratory_meds", "asmanex": "has_respiratory_meds",
    "ciclesonide": "has_respiratory_meds", "alvesco": "has_respiratory_meds",
    "beclomethasone": "has_respiratory_meds", "qvar": "has_respiratory_meds",
    "flonase": "has_respiratory_meds", "nasonex": "has_respiratory_meds",
    "nasacort": "has_respiratory_meds",
    "pulmicort": "has_respiratory_meds", "flexhaler": "has_respiratory_meds",
    "serevent": "has_respiratory_meds", "diskus": "has_respiratory_meds",
    "anoro": "has_respiratory_meds", "ellipta": "has_respiratory_meds",
    "dulera": "has_respiratory_meds",
    "foradil": "has_respiratory_meds", "aerolizer": "has_respiratory_meds",
    "arformoterol": "has_respiratory_meds", "brovana": "has_respiratory_meds",
    "maxair": "has_respiratory_meds", "pirbuterol": "has_respiratory_meds",
    "wixela": "has_respiratory_meds", "perforomist": "has_respiratory_meds",
    "montelukast": "has_respiratory_meds", "singulair": "has_respiratory_meds",
    "zafirlukast": "has_respiratory_meds", "accolate": "has_respiratory_meds",
    "zileuton": "has_respiratory_meds", "zyflo": "has_respiratory_meds",
    "tiotropium": "has_respiratory_meds", "spiriva": "has_respiratory_meds",
    "ipratropium": "has_respiratory_meds", "atrovent": "has_respiratory_meds",
    "combivent": "has_respiratory_meds", "duoneb": "has_respiratory_meds",
    "aclidinium": "has_respiratory_meds", "tudorza": "has_respiratory_meds",
    "glycopyrrolate": "has_respiratory_meds", "seebri": "has_respiratory_meds",
    "umeclidinium": "has_respiratory_meds", "incruse": "has_respiratory_meds",
    "vilanterol": "has_respiratory_meds", "breo": "has_respiratory_meds",
    "olodaterol": "has_respiratory_meds", "stiolto": "has_respiratory_meds",
    "theophylline": "has_respiratory_meds", "theo": "has_respiratory_meds",
    "roflumilast": "has_respiratory_meds", "daliresp": "has_respiratory_meds",
    "omalizumab": "has_respiratory_meds", "xolair": "has_respiratory_meds",
    "mepolizumab": "has_respiratory_meds", "nucala": "has_respiratory_meds",
    "benralizumab": "has_respiratory_meds", "fasenra": "has_respiratory_meds",
    "dupilumab": "has_respiratory_meds", "dupixent": "has_respiratory_meds",
    "cromolyn": "has_respiratory_meds",

    # ── Opioid ──
    "oxycodone": "has_opioid_meds", "percocet": "has_opioid_meds",
    "oxycontin": "has_opioid_meds", "roxicodone": "has_opioid_meds",
    "hydrocodone": "has_opioid_meds", "vicodin": "has_opioid_meds",
    "norco": "has_opioid_meds", "lortab": "has_opioid_meds",
    "zohydro": "has_opioid_meds",
    "morphine": "has_opioid_meds", "roxanol": "has_opioid_meds",
    "kadian": "has_opioid_meds", "avinza": "has_opioid_meds",
    "codeine": "has_opioid_meds",
    "tramadol": "has_opioid_meds", "ultram": "has_opioid_meds",
    "fentanyl": "has_opioid_meds", "duragesic": "has_opioid_meds",
    "actiq": "has_opioid_meds", "sublimaze": "has_opioid_meds",
    "hydromorphone": "has_opioid_meds", "dilaudid": "has_opioid_meds",
    "exalgo": "has_opioid_meds",
    "methadone": "has_opioid_meds", "dolophine": "has_opioid_meds",
    "buprenorphine": "has_opioid_meds", "suboxone": "has_opioid_meds",
    "subutex": "has_opioid_meds", "butrans": "has_opioid_meds",
    "belbuca": "has_opioid_meds", "zubsolv": "has_opioid_meds",
    "tapentadol": "has_opioid_meds", "nucynta": "has_opioid_meds",
    "meperidine": "has_opioid_meds", "demerol": "has_opioid_meds",
    "oxymorphone": "has_opioid_meds", "opana": "has_opioid_meds",
    "levorphanol": "has_opioid_meds",
    # Note: butalbital / Fioricet / Fiorinal WITHOUT codeine are barbiturate
    # combos classified as "Non-Opioid" in MIMIC — deliberately NOT mapped
    # here. Their codeine-containing variants flag via the codeine token.

    # ── Anticoagulant / antiplatelet ──
    "warfarin": "has_anticoagulant_meds", "coumadin": "has_anticoagulant_meds",
    "jantoven": "has_anticoagulant_meds",
    "apixaban": "has_anticoagulant_meds", "eliquis": "has_anticoagulant_meds",
    "rivaroxaban": "has_anticoagulant_meds", "xarelto": "has_anticoagulant_meds",
    "dabigatran": "has_anticoagulant_meds", "pradaxa": "has_anticoagulant_meds",
    "edoxaban": "has_anticoagulant_meds", "savaysa": "has_anticoagulant_meds",
    "betrixaban": "has_anticoagulant_meds",
    "aspirin": "has_anticoagulant_meds", "ecotrin": "has_anticoagulant_meds",
    "bayer": "has_anticoagulant_meds", "bufferin": "has_anticoagulant_meds",
    "ascriptin": "has_anticoagulant_meds",
    # Low-dose aspirin brands tokenize as "aspir" (e.g. "aspir-81", "aspir-low"),
    # "halfprin", "aspridrox", "asperdrink". Map the bare stems.
    "aspir": "has_anticoagulant_meds", "halfprin": "has_anticoagulant_meds",
    "aspridrox": "has_anticoagulant_meds", "asperdrink": "has_anticoagulant_meds",
    "excedrin": "has_anticoagulant_meds",  # contains aspirin
    "anacin": "has_anticoagulant_meds",
    # Other salicylates
    "salsalate": "has_anticoagulant_meds", "disalcid": "has_anticoagulant_meds",
    "diflunisal": "has_anticoagulant_meds", "dolobid": "has_anticoagulant_meds",
    # Platelet aggregation inhibitor not already listed
    "anagrelide": "has_anticoagulant_meds", "agrylin": "has_anticoagulant_meds",
    "clopidogrel": "has_anticoagulant_meds", "plavix": "has_anticoagulant_meds",
    "prasugrel": "has_anticoagulant_meds", "effient": "has_anticoagulant_meds",
    "ticagrelor": "has_anticoagulant_meds", "brilinta": "has_anticoagulant_meds",
    "ticlopidine": "has_anticoagulant_meds", "ticlid": "has_anticoagulant_meds",
    "cangrelor": "has_anticoagulant_meds", "kengreal": "has_anticoagulant_meds",
    "dipyridamole": "has_anticoagulant_meds", "persantine": "has_anticoagulant_meds",
    "aggrenox": "has_anticoagulant_meds",
    "cilostazol": "has_anticoagulant_meds", "pletal": "has_anticoagulant_meds",
    "heparin": "has_anticoagulant_meds",
    "enoxaparin": "has_anticoagulant_meds", "lovenox": "has_anticoagulant_meds",
    "dalteparin": "has_anticoagulant_meds", "fragmin": "has_anticoagulant_meds",
    "tinzaparin": "has_anticoagulant_meds", "innohep": "has_anticoagulant_meds",
    "fondaparinux": "has_anticoagulant_meds", "arixtra": "has_anticoagulant_meds",
    "argatroban": "has_anticoagulant_meds",
    "bivalirudin": "has_anticoagulant_meds", "angiomax": "has_anticoagulant_meds",
    "lepirudin": "has_anticoagulant_meds", "desirudin": "has_anticoagulant_meds",

    # ── GI ──
    # PPIs
    "omeprazole": "has_gi_meds", "prilosec": "has_gi_meds",
    "pantoprazole": "has_gi_meds", "protonix": "has_gi_meds",
    "esomeprazole": "has_gi_meds", "nexium": "has_gi_meds",
    "lansoprazole": "has_gi_meds", "prevacid": "has_gi_meds",
    "dexlansoprazole": "has_gi_meds", "dexilant": "has_gi_meds",
    "rabeprazole": "has_gi_meds", "aciphex": "has_gi_meds",
    # H2 blockers
    "ranitidine": "has_gi_meds", "zantac": "has_gi_meds",
    "famotidine": "has_gi_meds", "pepcid": "has_gi_meds",
    "cimetidine": "has_gi_meds", "tagamet": "has_gi_meds",
    "nizatidine": "has_gi_meds", "axid": "has_gi_meds",
    # Antiemetics
    "ondansetron": "has_gi_meds", "zofran": "has_gi_meds",
    "granisetron": "has_gi_meds", "dolasetron": "has_gi_meds",
    "promethazine": "has_gi_meds", "phenergan": "has_gi_meds",
    "meclizine": "has_gi_meds", "antivert": "has_gi_meds",
    "metoclopramide": "has_gi_meds", "reglan": "has_gi_meds",
    "scopolamine": "has_gi_meds",
    "aprepitant": "has_gi_meds", "emend": "has_gi_meds",
    "dronabinol": "has_gi_meds", "marinol": "has_gi_meds",
    # Laxatives / stool softeners
    "docusate": "has_gi_meds", "colace": "has_gi_meds",
    "senna": "has_gi_meds", "sennosides": "has_gi_meds", "senokot": "has_gi_meds",
    "bisacodyl": "has_gi_meds", "dulcolax": "has_gi_meds",
    "polyethylene": "has_gi_meds", "miralax": "has_gi_meds",
    "glycol": "has_gi_meds",
    "lactulose": "has_gi_meds", "kristalose": "has_gi_meds",
    "psyllium": "has_gi_meds", "metamucil": "has_gi_meds",
    "methylcellulose": "has_gi_meds", "citrucel": "has_gi_meds",
    "linaclotide": "has_gi_meds", "linzess": "has_gi_meds",
    "lubiprostone": "has_gi_meds", "amitiza": "has_gi_meds",
    "magnesium": "has_gi_meds",  # milk of magnesia / citrate; common enough
    # Antacids / calcium-based neutralizers
    "tums": "has_gi_meds", "maalox": "has_gi_meds", "mylanta": "has_gi_meds",
    "gaviscon": "has_gi_meds", "rolaids": "has_gi_meds",
    "bicarbonate": "has_gi_meds",  # sodium bicarbonate as antacid
    # Fleet enema / bowel prep
    "fleet": "has_gi_meds", "peg": "has_gi_meds",
    # Antidiarrheals / other
    "loperamide": "has_gi_meds", "imodium": "has_gi_meds",
    "diphenoxylate": "has_gi_meds", "lomotil": "has_gi_meds",
    "sucralfate": "has_gi_meds", "carafate": "has_gi_meds",
    "mesalamine": "has_gi_meds", "asacol": "has_gi_meds",
    "sulfasalazine": "has_gi_meds", "azulfidine": "has_gi_meds",
    "dicyclomine": "has_gi_meds", "bentyl": "has_gi_meds",
    "hyoscyamine": "has_gi_meds", "levsin": "has_gi_meds",

    # ── Thyroid / calcium regulation (grouped for training parity) ──
    "levothyroxine": "has_thyroid_meds", "synthroid": "has_thyroid_meds",
    "levoxyl": "has_thyroid_meds", "levothroid": "has_thyroid_meds",
    "unithroid": "has_thyroid_meds", "tirosint": "has_thyroid_meds",
    "euthyrox": "has_thyroid_meds",
    # Desiccated / natural thyroid ("thyroid (pork)") — tokenizes as
    # "thyroid" + "pork". Map both.
    "thyroid": "has_thyroid_meds", "throid": "has_thyroid_meds",
    "westhroid": "has_thyroid_meds", "naturethroid": "has_thyroid_meds",
    "armour": "has_thyroid_meds",
    "liothyronine": "has_thyroid_meds", "cytomel": "has_thyroid_meds",
    "liotrix": "has_thyroid_meds", "thyrolar": "has_thyroid_meds",
    "methimazole": "has_thyroid_meds", "tapazole": "has_thyroid_meds",
    "propylthiouracil": "has_thyroid_meds", "ptu": "has_thyroid_meds",
    # Calcitriol / calcimimetics — training groups these here (description
    # contains thyroid/parathyroid-related terms). Kept for training parity.
    "calcitriol": "has_thyroid_meds", "rocaltrol": "has_thyroid_meds",
    "doxercalciferol": "has_thyroid_meds", "hectorol": "has_thyroid_meds",
    "paricalcitol": "has_thyroid_meds", "zemplar": "has_thyroid_meds",
    "cinacalcet": "has_thyroid_meds", "sensipar": "has_thyroid_meds",
    "teriparatide": "has_thyroid_meds", "forteo": "has_thyroid_meds",

    # ── Anticonvulsant ──
    "gabapentin": "has_anticonvulsant_meds", "neurontin": "has_anticonvulsant_meds",
    "gralise": "has_anticonvulsant_meds",
    "pregabalin": "has_anticonvulsant_meds", "lyrica": "has_anticonvulsant_meds",
    "levetiracetam": "has_anticonvulsant_meds", "keppra": "has_anticonvulsant_meds",
    "phenytoin": "has_anticonvulsant_meds", "dilantin": "has_anticonvulsant_meds",
    "fosphenytoin": "has_anticonvulsant_meds", "cerebyx": "has_anticonvulsant_meds",
    # Carbamazepine / oxcarbazepine / valproate / lamotrigine are classic
    # mood stabilizers for bipolar disorder AND anticonvulsants. Training
    # flags both because the etcdescription names mention both roles
    # ("Anticonvulsant - Phenyltriazine Derivatives" matches `phenyltriazine`
    # in the psych class list). Inference must flag both too.
    "carbamazepine": ("has_anticonvulsant_meds", "has_psych_meds"),
    "tegretol": ("has_anticonvulsant_meds", "has_psych_meds"),
    "carbatrol": ("has_anticonvulsant_meds", "has_psych_meds"),
    "equetro": ("has_anticonvulsant_meds", "has_psych_meds"),
    "oxcarbazepine": ("has_anticonvulsant_meds", "has_psych_meds"),
    "trileptal": ("has_anticonvulsant_meds", "has_psych_meds"),
    "oxtellar": ("has_anticonvulsant_meds", "has_psych_meds"),
    "topiramate": "has_anticonvulsant_meds", "topamax": "has_anticonvulsant_meds",
    "trokendi": "has_anticonvulsant_meds", "qudexy": "has_anticonvulsant_meds",
    "valproic": ("has_anticonvulsant_meds", "has_psych_meds"),
    "valproate": ("has_anticonvulsant_meds", "has_psych_meds"),
    "divalproex": ("has_anticonvulsant_meds", "has_psych_meds"),
    "depakote": ("has_anticonvulsant_meds", "has_psych_meds"),
    "depakene": ("has_anticonvulsant_meds", "has_psych_meds"),
    "lamotrigine": ("has_anticonvulsant_meds", "has_psych_meds"),
    "lamictal": ("has_anticonvulsant_meds", "has_psych_meds"),
    "phenobarbital": ("has_anticonvulsant_meds", "has_psych_meds"),
    "luminal": ("has_anticonvulsant_meds", "has_psych_meds"),
    "zonisamide": "has_anticonvulsant_meds", "zonegran": "has_anticonvulsant_meds",
    "lacosamide": "has_anticonvulsant_meds", "vimpat": "has_anticonvulsant_meds",
    "primidone": "has_anticonvulsant_meds", "mysoline": "has_anticonvulsant_meds",
    "ethosuximide": "has_anticonvulsant_meds", "zarontin": "has_anticonvulsant_meds",
    "felbamate": "has_anticonvulsant_meds", "felbatol": "has_anticonvulsant_meds",
    "tiagabine": "has_anticonvulsant_meds", "gabitril": "has_anticonvulsant_meds",
    "vigabatrin": "has_anticonvulsant_meds", "sabril": "has_anticonvulsant_meds",
    "rufinamide": "has_anticonvulsant_meds", "banzel": "has_anticonvulsant_meds",
    "perampanel": "has_anticonvulsant_meds", "fycompa": "has_anticonvulsant_meds",
    "eslicarbazepine": "has_anticonvulsant_meds", "aptiom": "has_anticonvulsant_meds",
    "clobazam": "has_anticonvulsant_meds", "onfi": "has_anticonvulsant_meds",
    "brivaracetam": "has_anticonvulsant_meds", "briviact": "has_anticonvulsant_meds",
}


# ---------------------------------------------------------------------------
# Class / free-text keywords -> category. Used on etcdescription (training)
# and patient free-text (inference: "blood thinner", "anxiety pill").
# Matched with word boundaries; "non-opioid" / "non-salicylate" negations
# are neutralized first.
# ---------------------------------------------------------------------------
MED_CLASS_KEYWORDS = {
    "has_cardiac_meds": [
        "statin", "beta blocker", "beta-blocker",
        "ace inhibitor", "calcium channel", "angiotensin",
        "diuretic", "aldosterone", "nitrate", "coronary vasodilator",
        "antihyperlipidemic", "antiarrhythmic", "alpha-beta blocker",
        "cardiac glycoside", "digitalis glycoside",
        "alpha-1 receptor", "alpha-2 receptor",
        "direct acting vasodilator", "vasodilator",
        "antianginal",
        "niacin",
        "blood pressure", "cholesterol", "heart medication",
    ],
    "has_diabetes_meds": [
        "biguanide", "insulin", "sulfonylurea", "antihyperglycemic",
        "sglt", "glp", "dpp",
        "blood sugar", "diabetes",
    ],
    "has_psych_meds": [
        "benzodiazepine", "antidepressant", "ssri", "snri", "ndri",
        "antipsychotic", "antianxiety", "sedative-hypnotic", "sari",
        "nassa", "tricyclic", "bipolar", "adhd",
        "phenyltriazine",     # lamotrigine class
        "amphetamine", "narcolepsy therapy",
        "anxiety", "depression", "sleeping pill", "sleep aid",
    ],
    "has_respiratory_meds": [
        "asthma", "copd", "inhaled corticosteroid", "nasal corticosteroid",
        "bronchodilator", "beta 2-adrenergic", "beta-2 adrenergic",
        "leukotriene", "inhaler", "breathing",
    ],
    "has_opioid_meds": [
        "opioid", "oxycodone",
        "narcotic", "pain killer", "painkiller",
    ],
    "has_anticoagulant_meds": [
        "anticoagulant", "coumarin",
        "platelet aggregation inhibitor", "thienopyridine",
        "salicylate",             # covers "Salicylate Analgesics" (aspirin)
        "factor xa",
        "heparin",
        "thrombin inhibitor",
        "blood thinner",
    ],
    "has_gi_meds": [
        "proton pump", "ppi", "h2-receptor", "h2 receptor",
        "gastric acid", "peptic ulcer",
        "laxative", "stool softener",
        "antiemetic", "antiemesis",
        "antacid",
        "antidiarrheal",
        "antispasmodic",
        "prokinetic",
        "colonic acidifier", "ammonia inhibitor",
        "cytoprotective",
        "inflammatory bowel",
        "5-ht3",                  # ondansetron class
        "acid reflux", "stomach",
    ],
    "has_thyroid_meds": [
        "thyroid", "antithyroid",
        "calcimimetic",
        "vitamin d analog",
        "parathyroid",
        # Note: plain "Vitamins - D Derivatives" (cholecalciferol / vit D3
        # supplements) is intentionally NOT flagged — those are OTC
        # vitamins, not thyroid therapy. Prescription analogs (doxercalciferol,
        # paricalcitol, calcitriol) are covered via DRUG_NAME_MAP.
    ],
    "has_anticonvulsant_meds": [
        "anticonvulsant",
        "gaba analog",
        "seizure", "epilepsy",
    ],
}

MED_CATEGORIES = list(MED_CLASS_KEYWORDS.keys())


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------

# Replace "non-opioid" / "non opioid" / "non-salicylate" etc. with a placeholder
# so the bare keyword inside them doesn't get matched.
_NEGATION_RE = re.compile(
    r'\bnon[-\s]+(opioid|salicylate|steroidal|insulin)s?\b',
    re.IGNORECASE,
)

# Split a drug name into lowercase alphabetic tokens. Handles hyphens, brackets,
# slashes, punctuation uniformly (e.g., "fluticasone-salmeterol [advair diskus]"
# -> ["fluticasone", "salmeterol", "advair", "diskus"]).
_NAME_TOKEN_RE = re.compile(r'[a-z]+')

# Pre-compile one regex per category: any keyword with a trailing optional
# `s` so plural forms match too. Without this, keyword "benzodiazepine"
# fails to match "Benzodiazepines" (word boundary between d|s is a non-
# boundary), "nasal corticosteroid" misses "Nasal Corticosteroids",
# "heparin" misses "Heparins", etc. Word-boundary at the start is kept so
# "statin" still does not match "nystatin".
_CLASS_PATTERNS = {
    cat: re.compile(
        '|'.join(r'\b' + re.escape(kw) + r's?\b' for kw in kws),
        re.IGNORECASE,
    )
    for cat, kws in MED_CLASS_KEYWORDS.items()
}


def _add_cat(flags: set, cat) -> None:
    """Accept either a single-category string or a tuple/list of categories."""
    if isinstance(cat, str):
        flags.add(cat)
    else:
        flags.update(cat)


def flags_from_name(name) -> set:
    """Return category flags matched from a drug-name string.

    Tokenizes into alphabetic runs and looks up each token in DRUG_NAME_MAP.
    A single token may flag multiple categories (e.g., clonazepam maps to
    both has_psych_meds and has_anticonvulsant_meds).
    """
    if not isinstance(name, str) or not name:
        return set()
    flags = set()
    for tok in _NAME_TOKEN_RE.findall(name.lower()):
        cat = DRUG_NAME_MAP.get(tok)
        if cat is not None:
            _add_cat(flags, cat)
    return flags


def flags_from_text(text) -> set:
    """Return category flags matched from a class description or free-text string.

    Word-boundary keyword match. Negations like "non-opioid" are neutralized
    first so they don't trigger "opioid".
    """
    if not isinstance(text, str) or not text:
        return set()
    text_norm = _NEGATION_RE.sub(' _NEG_ ', text.lower())
    flags = set()
    for cat, pattern in _CLASS_PATTERNS.items():
        if pattern.search(text_norm):
            flags.add(cat)
    return flags


def flags_from_row(name, etcdescription) -> set:
    """Training-time: union of name-map flags and class-keyword flags."""
    return flags_from_name(name) | flags_from_text(etcdescription)
