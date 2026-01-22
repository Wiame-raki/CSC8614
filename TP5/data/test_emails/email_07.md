---
email_id: E07
from: "Marc Lemoine <marc.lemoine@external-partner.com>"
date: "2026-01-18"
subject: "Question"
---

CORPS:
<<<
Salut,

Est-ce que tu as pu regarder le truc que je t'ai envoyé ? On doit valider ça rapidement sinon ça va bloquer pour la suite.

Dis-moi si c'est bon.

A+
Marc
>>>

ATTENDU:
- intent: ask_clarification
- note: Demande ambiguë ("le truc"), demander de quel dossier/fichier il s'agit.