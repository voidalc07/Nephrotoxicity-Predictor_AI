from __future__ import annotations

# -------------------------------------------------------------------------
# Named SMARTS Alert Library
# These alerts are lightweight medicinal-chemistry heuristics rather than a
# full mechanistic toxicology ontology. They exist to provide transparent
# structural cues in the live explanation stack, complementing statistical
# predictions with interpretable toxicophore-style matches described in the
# dissertation's chemistry-facing discussion.
# -------------------------------------------------------------------------
ALERT_DEFINITIONS: list[dict[str, str]] = [
    {
        "name": "Methyl phosphate",
        "smarts": "[#6]OP(=O)(O[#6])O[#6]",
        "reference": "Phosphate-containing motifs can alter renal handling and accumulation risk.",
    },
    {
        "name": "Glycoside",
        "smarts": "[#6]1-O-[#6](-O-[#6])-[#6](-O)-[#6](-O)-[#6]-1-O",
        "reference": "Glycoside-rich antibiotics are often discussed in nephrotoxicity screening contexts.",
    },
    {
        "name": "Fluoroquinolone",
        "smarts": "n1cc(C(=O)O)c(=O)c2cc(F)c(N3CCNCC3)cc12",
        "reference": "Fluoroquinolone-like motifs are monitored due to known renal adverse-effect literature.",
    },
    {
        "name": "Beta-lactam",
        "smarts": "N1C(=O)CC1",
        "reference": "Beta-lactam substructures are retained as clinically familiar structural alerts.",
    },
    {
        "name": "Cephalosporin",
        "smarts": "O=C1N2C(=C(CS[C@H]2[C@H]1)C)C(=O)O",
        "reference": "Cephalosporin-like scaffolds are included as named medicinally relevant alerts.",
    },
    {
        "name": "Tetrazole",
        "smarts": "[c,C]1=NN=NN1",
        "reference": "Tetrazole motifs are flagged because they can shift acidity and transporter interactions.",
    },
    {
        "name": "Phenylsulfonylacetic acid",
        "smarts": "c1ccccc1S(=O)(=O)CC(=O)O",
        "reference": "Sulfonyl-acid motifs are retained from the project's existing alert library.",
    },
    {
        "name": "Pyridinecarboxamide",
        "smarts": "c1ccncc1C(=O)N",
        "reference": "Pyridinecarboxamide patterns are kept as medicinal-chemistry interpretation cues.",
    },
    {
        "name": "Purine",
        "smarts": "n1cnc2c1ncnc2",
        "reference": "Purine-like heterocycles are relevant for nucleoside and antiviral analogues.",
    },
    {
        "name": "Chlorobenzene",
        "smarts": "c1ccccc1Cl",
        "reference": "Aromatic halogen motifs are included as simple lipophilic structural flags.",
    },
]
