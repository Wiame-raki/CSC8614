# Évaluation RAG - 3 Questions Sélectionnées

## Question 1 (Q3): "Quel est la date limite pour s'inscrire au TOEIC du 17 janvier ?"

**Source attendue:** email

**Réponse générée (avant):**
```
Information manquante :
1. Non, aucun sujeto supplémentario pour les PFE n'est indiqué dans le contexte.
2. Non, aussi aucun sujeto supplémentaire pour les PFE n'est mentionné.
ANSWER: Information insuffisante. 2 informations manquantes.
```

**Sources récupérées:**
- doc_1: (admin_pdf) Reglement_Interieur_TSP
- doc_2: (admin_pdf) Reglement_Interieur_TSP
- doc_3: (email) TOEIC-related file
- doc_4: (admin_pdf) Reglement_Interieur_TSP
- doc_5: (email) TOEIC-related file

**Analyse:**
- ✅ Française
- ❌ Pas de citation [doc_i]
- ✅ Reconnaissance de l'insuffisance (format correct "Information insuffisante.")
- ❌ Réponse incohérente avec la question (devrait être simple : "12 janvier minuit")
- ❌ Contient du charabia ("sujeto supplémentario")

**SCORE: 0**
Raison: Hallucination - la réponse est totalement hors sujet (parle de PFE au lieu de TOEIC), mal formée, et contient du bruit malgré que TOEIC soit en doc_5.

---

## Question 2 (Q2): "Comment valider une UE ?"

**Source attendue:** admin_pdf

**Réponse générée:**
```
**RÉponse :**

Pour valider une UE, l'élève doit suivre les critères stipulés dans le document "Validation des acquis 
de l'expérience (VAE)" du rapport [doc_4]. L'évaluation est effectuée en considering les qualifications, 
l'experiencé professionnelle et la performance. Le jury peut awarder ou non une licencée selon ces critères. 
Si un UE ne est pas validé, il peut être retourné pour se révalider dans l'année scolaire suivante. 
Il est advised à utiliser les documents de validation proposés et cited [doc_4] pour faire la réponse actionnable.
```

**Sources récupérées:**
- doc_1-4: (admin_pdf) Reglement_Interieur_TSP
- doc_5: (admin_pdf) Reglement_scolarite_FISA

**Analyse:**
- ✅ Française (avec fautes mineures : "RÉponse", "considering", "awarder", "licencée" hors contexte)
- ✅ Citations présentes: [doc_4]
- ⚠️ Réponse partiellement correcte mais avec erreurs et anglicismes
- ✅ Contient des détails (jury, révaluation)
- ❌ Qualité linguistique dégradée (mélange en/fr, accords incorrects)

**SCORE: 1**
Raison: Partiellement correct avec citations, mais nombreuses fautes linguistiques et formulation incohérente malgré l'effort de sourçage.

---

## Question 3 (Précédente): "Comment valider une UE ?" (2e test)

**Source attendue:** admin_pdf

**Réponse générée:** (identique à Q2 ci-dessus)

**SCORE: 1** (même analyse)

---

## Résumé Scores

| Question | Score | Raison |
|----------|-------|--------|
| Q3 (TOEIC deadline) | **0** | Hallucination complète, réponse hors-sujet |
| Q2 (UE validation) | **1** | Citations présentes mais qualité linguistique faible |
| Q6 (ENG 8601) | À tester | Non exécuté |

## Recommandations

1. **Améliorer le prompt** : Ajouter des contraintes strictes en français
2. **Post-traitement** : Nettoyer la sortie (supprimer les balises non-FR, corriger les accords)
3. **RAG refinement** : Vérifier que les bons documents sont rangés en top-k (notamment pour q3)
4. **Validation** : Tester avec plus d'exemples et des métriques ROUGE/BLEU
