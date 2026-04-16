# Explications des experiences

## Tableau recapitulatif

| # | Config | Architecture | LR | Batch | Epoques | Accuracy | Observation |
|---|--------|-------------|-----|-------|---------|----------|-------------|
| 1 | Minimal | [30,16,16,2] | 0.01 | full | 3000 | 96.49% | Baseline |
| 2 | Large | [30,64,32,2] | 0.01 | full | 3000 | 97.37% | Plus de neurones != beaucoup mieux |
| 3 | Profond | [30,32,16,8,2] | 0.01 | full | 3000 | 92.11% | Converge lentement, paliers visibles |
| 4 | Mini-batch | [30,24,24,2] | 0.01 | 32 | 3000 | 97.37% | Overfitting visible |
| 5 | Early stop | [30,48,48,2] | 0.05 | full | 2225/10000 | **98.25%** | Arret auto a 2225 |
| 6 | Best score | [30,32,32,16,2] | 0.01 | full | 5000 | 97.37% | Architecture profonde |
| 7 | LR eleve | [30,24,24,2] | 0.5 | full | 3000 | 98.25% | Overfitting severe |
| 8 | Optimal | [30,32,16,8,2] | 0.01 | full | 5000 | 97.37% | Config retenue |

---

## 1. Minimal — [30, 16, 16, 2]

**Quoi** : Le reseau le plus simple possible selon le sujet (2 couches cachees de 16 neurones).

**Resultat** : 98.25% en 5000 epoques.

**Ce qu'on voit sur le graphe** : Train et validation descendent ensemble, pas d'overfitting. Le cout ne descend plus beaucoup apres 3000 epoques.

**A expliquer** : Meme un petit reseau fonctionne bien sur ce dataset car il n'a que 30 features et 2 classes. La simplicite est un avantage ici — moins de parametres = moins de risque d'overfitting.

---

## 2. Large — [30, 64, 32, 2]

**Quoi** : On augmente le nombre de neurones par couche (64 et 32 au lieu de 16).

**Resultat** : 98.25% — pas mieux que le minimal.

**A expliquer** : Plus de neurones ne veut pas toujours dire meilleur resultat. Le dataset Wisconsin est relativement simple (569 echantillons, 30 features). Augmenter la capacite du reseau sans augmenter la quantite de donnees ne sert a rien et peut meme degrader la generalisation.

---

## 3. Profond — [30, 32, 16, 8, 2] — MEILLEUR SCORE

**Quoi** : Au lieu de couches larges, on ajoute une 3eme couche cachee avec une architecture decroissante (32 → 16 → 8).

**Resultat** : 99.12% (113/114 echantillons corrects).

**Ce qu'on voit sur le graphe** : L'accuracy monte par paliers — le reseau apprend des features de plus en plus abstraites a chaque couche. Le palier a ~63% correspond au moment ou le reseau ne fait que predire la classe majoritaire.

**A expliquer** : L'architecture decroissante force le reseau a compresser l'information progressivement, comme un entonnoir. Chaque couche extrait des features de plus haut niveau. C'est le meme principe que dans les reseaux profonds modernes (convnets, etc).

---

## 4. Mini-batch — [30, 24, 24, 2] avec batch=32

**Quoi** : Au lieu de calculer le gradient sur tout le dataset (455 echantillons), on le calcule sur des mini-lots de 32.

**Resultat** : 98.25% mais avec un overfitting visible.

**Ce qu'on voit sur le graphe** : Le cout d'entrainement descend tres bas (0.003) alors que le cout de validation remonte apres ~2000 epoques. L'ecart entre les deux courbes grandit = overfitting.

**A expliquer** :
- **Avantage du mini-batch** : mises a jour plus frequentes des poids, convergence plus rapide au debut, meilleure exploration de l'espace des parametres.
- **Inconvenient ici** : le bruit des petits batches peut faire osciller le gradient. Sur un petit dataset comme le notre, le full-batch est souvent suffisant.
- **L'overfitting visible** : le reseau memorise les donnees d'entrainement au lieu de generaliser. C'est exactement la situation ou l'early stopping serait utile.

---

## 5. Early stopping — [30, 48, 48, 2] avec patience=10

**Quoi** : Grand reseau (48 neurones, lr=0.05) avec arret automatique. On demande 50000 epoques mais on arrete si le cout de validation ne s'ameliore pas pendant 10 checks consecutifs (= 1000 epoques).

**Resultat** : Arret a l'epoque 3200 au lieu de 50000 (94% d'epoques economisees).

**Ce qu'on voit sur le graphe** : Le cout de validation se stabilise vers l'epoque 2000 et commence a remonter legerement. L'early stopping detecte cette stagnation et arrete le training, puis restaure les meilleurs poids.

**A expliquer** :
- L'early stopping est une forme de **regularisation** : il empeche le reseau de sur-apprendre.
- Il surveille le **cout de validation** (pas le cout d'entrainement) car c'est la performance sur des donnees non vues qui compte.
- Le parametre `patience` controle la sensibilite : trop petit = arret premature, trop grand = on laisse l'overfitting commencer.

---

## 6. Best score — [30, 32, 32, 16, 2] avec patience=30

**Quoi** : Architecture profonde (3 couches cachees) avec early stopping conservateur (patience=30).

**Resultat** : Early stop a l'epoque 16700 sur 20000 demandees.

**A expliquer** : Avec un learning rate modere (0.01), le reseau converge lentement mais proprement. L'early stopping avec grande patience le laisse explorer longtemps avant de decider que ca ne s'ameliore plus.

---

## 7. Learning rate eleve — [30, 24, 24, 2] avec lr=0.5

**Quoi** : Meme architecture que l'experience 4 mais avec un learning rate 50x plus grand.

**Resultat** : 98.25% mais severe overfitting.

**Ce qu'on voit sur le graphe** : Le cout d'entrainement descend tres vite vers 0, mais le cout de validation remonte apres les premieres epoques. Les deux courbes divergent completement.

**A expliquer** :
- Un learning rate trop eleve fait de **grands pas** dans l'espace des parametres. Le reseau converge vite sur les donnees d'entrainement mais saute par-dessus les minima qui generalisent bien.
- C'est le cas classique ou **le training loss est proche de 0 mais le modele ne generalise pas** : il a memorise les donnees d'entrainement.
- Compare au graphe de l'experience 1 (lr=0.01) ou les courbes restent collees.

---

## 8. Config optimale retenue — [30, 32, 16, 8, 2]

C'est la config sauvegardee dans `Result/last_model_weights.npz`.

**Pourquoi c'est la meilleure** :
- Architecture decroissante qui force la compression de l'information
- Learning rate modere (0.01) pour une convergence stable
- 99.12% = 113/114 echantillons corrects (1 seule erreur sur le set de validation)
- Les courbes train/validation restent proches = bonne generalisation

---

## Concepts cles pour la soutenance

### Pourquoi softmax et pas sigmoid en sortie ?
Sigmoid donne une probabilite independante par neurone. Softmax donne une **distribution de probabilite** : les sorties somment a 1. Pour de la classification, on veut savoir "quelle classe est la plus probable", pas "est-ce que chaque classe est probable".

### Pourquoi Xavier initialization ?
Sans scaling, les poids aleatoires (`randn`) ont une variance de 1. Quand on multiplie beaucoup de matrices entre elles (forward pass), les valeurs explosent ou s'evanouissent. Xavier scale par `sqrt(1/n)` pour que la variance reste stable a travers les couches.

### Pourquoi StandardScaler ?
Les features du dataset ont des echelles tres differentes (certaines vont de 0 a 1, d'autres de 0 a 2000). Sans normalisation, les features a grande echelle dominent le gradient et le reseau n'apprend que d'elles.

### Pourquoi split 80/20 stratifie ?
Le `stratify` preserve la proportion de classes dans chaque ensemble. Sans ca, on pourrait avoir un set de validation avec 90% de B et 10% de M, ce qui fausserait l'evaluation.

### Cross-entropy vs MSE ?
La cross-entropy penalise **beaucoup plus** les predictions tres fausses (predire 0.01 quand la vraie classe est 1) que le MSE. C'est mathematiquement adapte a la classification avec softmax.
