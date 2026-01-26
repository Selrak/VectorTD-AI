# VectorTD — Espace d’action **Discrete(K)** + candidatisation **par type** (multi-maps)

Convention d’écriture dans ce document : tout code inline est entre backticks (ex. `foo()`), tout bloc de code est entre triples backticks.

---

## Sommaire

0. État des lieux (run_016 à ~2.98M steps) : ce que cela suggère sur la “santé” de l’apprentissage  
1. Objectif et contraintes (SB3 / MaskablePPO / multi-maps)  
2. Pourquoi l’espace d’action actuel (≈2338) casse (et pourquoi le masquage n’est pas une solution magique)  
3. Vue d’ensemble de la solution Discrete(K) + candidats dynamiques “par type”  
4. Définition exacte de l’espace d’action `gym.spaces.Discrete(K)`  
5. Calcul des candidats `cells[t]` (par map) et choix concrets de `Kcells[t]` (par type de tour)  
6. Gestion des tours existantes via `Ktower` “slots” stables (upgrade / sell / set_mode)  
7. Format exact `F` (features par candidat) : compact + multi-maps  
8. Action masking : règles exactes, garanties “au moins 1 action valide”, et instrumentation anti-Simplex  
9. Plan d’implémentation pas à pas (fichiers, fonctions, assertions, tests)

---

## 0) État des lieux (run_016 à ~2.98M steps) : lecture minimale

D’après les fichiers de run (`run_016/run_summary.json` + `monitor/*.monitor.csv`) :

- Les épisodes restent **tous perdus** : les rewards observés sont dans la bande ~`[-2052, -1535]` (max = -1535).  
  Avec une pénalité de défaite de l’ordre de `-1000` et une pénalité par vie perdue de l’ordre de `-50`, cela correspond typiquement à des runs où **beaucoup de vies** sont perdues (ex. -2000 ≈ défaite + ~20 vies perdues).  
- Il y a un **signal faible d’amélioration** : certains épisodes plus tardifs ont des rewards un peu moins négatifs (donc moins de vies perdues), et des longueurs d’épisode plus grandes (épisodes qui “tiennent” plus longtemps).  
- Le scale de reward est **très grand** (ordre de grandeur `~ -2000` par épisode), ce qui rend l’optimisation (et la calibration des hyperparams PPO) plus délicate.

Conclusion utile pour la suite : il y a vraisemblablement un signal, mais il est noyé dans (1) un espace d’action très grand + masquage “massif”, et (2) un reward scale trop large. La suite du document s’attaque à (1) en priorité, et donne aussi la structure pour “resserrer” le reward à `~[-10,+10]` ensuite (sans intégrer de reward “dégâts infligés” pour l’instant).

---

## 1) Objectif et contraintes (SB3 / MaskablePPO / multi-maps)

### Objectif

Construire une API d’action stable et factorisée, adaptée à :

- l’entraînement sur **Switchback** (d’abord),
- puis la généralisation sur **plusieurs maps** (variabilité des cellules constructibles, des chemins, etc.),
- tout en restant dans le cadre SB3 `MaskablePPO`.

### Contraintes principales

- SB3 et SB3-contrib attendent un **action space de taille fixe** : `Discrete(N)` ou `MultiDiscrete([...])`.  
  La variabilité “intrinsèque” (carte, tours disponibles, slots de tours, etc.) doit donc être absorbée par :
  - une **candidatisation** (on ne propose qu’un sous-ensemble d’actions),
  - un **mask** (on désactive dynamiquement ce qui n’est pas valide).
- L’objectif est de réduire massivement l’espace d’action actuel, qui ressemble à :
  - `PLACE(tower_type, cell)` pour **tous** les couples `(type, cell)`,
  - + `UPGRADE(tower_id)` / `SELL(tower_id)` / `SET_MODE(tower_id, mode)`.

Le placement dans le SWF est “cell-based” avec snap 25 px + contraintes de validité (zone, hitTest chemin, pas de tour déjà présente). Référence Flash : `scripts/DefineSprite_146_placer/frame_1/DoAction.as`.

---

## 2) Pourquoi l’espace d’action actuel (≈2338) casse

### 2.1 Symptôme : erreur “Simplex()” sur la distribution Categorical masquée

L’exception typique est :

- `ValueError: Expected parameter probs ... to satisfy the constraint Simplex(), but found invalid values`

Dans MaskablePPO, cela survient souvent quand, à un moment de l’entraînement :

- un masque rend **trop** d’actions invalides (logits “écrasés”),
- et/ou certaines lignes de batch se retrouvent avec **zéro action valide** (masque tout à `False`),
- et/ou les logits masqués produisent des `NaN` ou des distributions numériquement incohérentes.

La littérature SB3/SB3-contrib montre que le couple “grand espace discret + masquage lourd” est un point fragile : de nombreux tickets et discussions mentionnent des erreurs Simplex/MaskableCategorical sur des tailles d’action très grandes.

### 2.2 Cause racine (pragmatique) : masquage massif + très grande dimension

Même si “en théorie” le masquage rend le problème équivalent à une distribution sur le sous-ensemble valide, “en pratique” :

- Le réseau doit produire des logits sur **toutes** les actions, même celles presque toujours masquées.
- Le masquage transforme des logits en valeurs extrêmement négatives (ou `-inf`), et la softmax dans une dimension très grande devient plus fragile.
- Les bugs ou incohérences de masques (un seul step mal masqué) coûtent très cher (crash).

Donc la stratégie est : réduire `N` et réduire la pression sur les masques.

---

## 3) Vue d’ensemble de la solution Discrete(K) + candidats “par type”

Idée centrale :

- Fixer un `Discrete(K)` **modéré** (ex. `K ~ 600–900` au lieu de 2338).
- Chaque action correspond à un “slot” abstrait (ex. “placer une `tower_green` sur le candidat #7 pour cette map”).
- Pour chaque map, on construit **à l’initialisation** (ou à `reset()`) un tableau de candidats par type : `cells[t] = [cell_0, ..., cell_{Kcells[t]-1}]`.
- Pendant l’épisode, un masque désactive ce qui est impossible (pas assez d’argent, cellule non valide, pas de slot de tour, etc.).

Ce découplage a deux effets :

1) Robustesse numérique : moins de logits, moins de chances de masquage catastrophique.  
2) Généralisation multi-map : `cells[t]` est recalculé à partir de la géométrie de la map, donc l’agent voit toujours la même “API” d’action.

---

## 4) Définition exacte de l’espace d’action `gym.spaces.Discrete(K)`

### 4.1 Notations

- `T` : liste des types de tours (bank + éventuellement buff)  
  Exemples (towers SWF) :
  - bank : `tower_green`, `tower_green2`, `tower_green3`, `tower_pink1`, `tower_pink2`, `tower_pink3`, `tower_brown`, `tower_blue`, `tower_bash`, `tower_swarm`, `tower_red`
  - buffs : `tower_buff1` (buffD), `tower_buff2` (buffR)

- `Kcells[t]` : nombre de candidats de cellules pour le type `t` (fixé globalement, mais la liste `cells[t]` est map-dépendante).
- `Ktower` : nombre maximum de “slots” de tours existantes exposés à l’agent (ex. 32).

### 4.2 Liste d’actions et indexation

Action space : `Discrete(K)` avec une table `ActionSpec[idx] -> (op, args...)` :

1) `idx=0` : `NOOP`  
2) `idx=1` : `START_WAVE`

3) Bloc “PLACER” (tours bank + buffs) :  
   Pour chaque type `t` dans un ordre fixe, pour chaque `k in [0..Kcells[t)-1]` :  
   - `PLACE(t, slot=k)`  
   Ici `slot` renvoie à `cells[t][k]` (cellule concrète de la map).

4) Bloc “UPGRADE/SELL/SET_MODE” sur `Ktower` slots :  
   - `UPGRADE(slot=i)` pour `i in [0..Ktower-1]`  
   - `SELL(slot=i)` idem  
   - `SET_MODE(slot=i, mode=m)` pour `m` dans une liste fixe `MODES = [closest, weakest, hardest, fastest, random]`

Important : le masque invalide les modes non-supportés par un type de tour (ex. `tower_swarm` a un ciblage `random` en SWF, `tower_blue`/`tower_bash` ciblent “fastest”).

### 4.3 Taille K typique (ordre de grandeur)

Avec les choix concrets de `Kcells[t]` ci-dessous (section 5), on obtient :

- placement : ≈ 500 actions
- opérations tours (si `Ktower=32`) : `32 (upgrade) + 32 (sell) + 32*5 (set_mode) = 224`
- total : `K ≈ 2 + 500 + 224 = 726`

---

## 5) Calcul des candidats `cells[t]` et choix concrets de `Kcells[t]`

### 5.1 Source de vérité : buildable cells

Le placement SWF impose :

- snap sur grille (pas 25 px),
- zone de validité `[0..550) × [0..450)`,
- collision chemin via `hitTest` sur `_map.hit`,
- absence de tour déjà présente à la même coordonnée.

Référence : `scripts/DefineSprite_146_placer/frame_1/DoAction.as`.


**Extrait SWF (placement + constructibilité + paiement)** — `important_scripts/scripts/DefineSprite_146_placer/frame_1/DoAction.as` :

```actionscript
_X = int((_root._xmouse - _width / 2) / 25) * 25;
_Y = int((_root._ymouse - 8 - _height / 2) / 25) * 25 - 75;

if(_X >= 0 && _Y >= 0 && _X < 550 && _Y < 450)
{
   ...
   if(_root._game._map.hit.hitTest(_X + 25,_Y + 100,1))
   {
      OK = 0;
   }
   else
   {
      OK = 1;
   }
   if(OK == 1)
   {
      i = 0;
      while(i < _root._game.towerArray.length)
      {
         if(_root._game.towerArray[i]._x == _X && _root._game.towerArray[i]._y == _Y)
         {
            OK = 0;
         }
         i++;
      }
   }
   ...
}

onRelease = function()
{
   if(OK == 1)
   {
      if(Type != "tower")
      {
         _root._game.ups--;
      }
      else
      {
         _root._game.bank -= cost;
      }
      ...
      _root._game.towerArray.push(T);
      ...
   }
   _root._game.doBuffs();
};
```

Côté Python, l’environnement doit déjà avoir l’équivalent de `can_place_tower(t, cell)`.

Le set initial “buildable cells” de la map est donc défini par :  
`B = {cell | can_place_tower(tower_green, cell) == True en l’absence de tours}`  
(Le type n’importe pas pour la constructibilité brute, sauf si plus tard certaines maps restreignent par type.)

### 5.2 But de `cells[t]` : “réduire sans heuristique de gameplay”

`cells[t]` ne doit pas “jouer à la place” de l’agent.  
Mais `cells[t]` doit éliminer les cases qui, pour un type donné, sont structurellement inutiles :

- Une tour de portée ~3 cases posée très loin du chemin est quasi toujours inactive.
- Une tour de portée ~6 cases (RED ROCKETS) peut rester utile plus loin.

Le bon compromis (pour multi-map) : candidatisation basée uniquement sur la géométrie (chemins/markers) et la portée.

### 5.3 Score géométrique recommandé

Pré-calcul map-dépendant, type-dépendant :

1) Construire une polyline de chemin “dense” : échantillonnage des segments du chemin à pas fixe (ex. 10 px).  
   Appelons-la `P = [p0, p1, ...]`.

2) Pour une cellule candidate `c` (centre en pixels), et une portée `R_t` (pixels) :

- `coverage_t(c) = (# points p ∈ P tels que dist(p,c) <= R_t) / |P|`
- `dist_to_path(c) = min_{p ∈ P} dist(p,c)`

Score simple :
- `score_t(c) = coverage_t(c)  - 0.15 * clamp(dist_to_path(c) / R_t, 0, 1)`

Intuition :
- On préfère couvrir une grande fraction de chemin (coverage),
- et on décourage légèrement les cellules trop loin (sans interdire, ce qui laisse les RED ROCKETS “loin mais couvrantes”).

3) Diversité spatiale : éviter 20 candidats quasi identiques au même endroit.  
   Méthode simple : “greedy with minimum distance”  
   - trier par score desc,
   - prendre le meilleur,
   - puis ignorer tout candidat à distance < `dmin_t` du déjà-pris (ex. `dmin_t = 1 cell` ou `2 cells` selon la portée),
   - continuer jusqu’à `Kcells[t]`.

### 5.4 Choix concrets de `Kcells[t]` (à partir de `towers.md`)

Données utiles (coût / portée) : voir `towers.md`.  
Portées SWF typiques :
- `70–80 px` (≈3 cases),
- `100 px` (≈4 cases),
- `150 px` (≈6 cases).

Table recommandée (compromis “compact + multi-map”):

| type `t` | range (px) | `Kcells[t]` | remarque |
|---|---:|---:|---|
| `tower_green` / `green2` / `green3` | 70 | 32 | portée courte → candidats centrés sur zones proches du chemin |
| `tower_blue` | 70 | 32 | idem |
| `tower_brown` | 80 | 32 | idem |
| `tower_bash` | 80 | 32 | idem |
| `tower_swarm` | 80 | 32 | idem |
| `tower_pink1` / `pink2` / `pink3` | 100 | 48 | un peu plus large → plus de diversité utile |
| `tower_red` | 150 | 64 | longue portée → plus de spots pertinents, y compris plus loin |
| `tower_buff1` (buffD) | 80 (aura effective <100) | 32 | buff = placement “près d’un cluster” ; on reste compact |
| `tower_buff2` (buffR) | 80 (aura effective <100) | 32 | idem |

Remarque buff : dans le SWF, l’effet s’applique si `dist < 100` et il s’agit de +25% dégâts ou +25% portée (accumulable), avec recalcul `buffedRange/buffedDamage`.


**Extrait SWF (buffs : rayon effectif + pourcentages)** — `important_scripts/scripts/DefineSprite_525/frame_1/DoAction.as` :

```actionscript
function doBuffs()
{
   i = 0;
   while(i < towerArray.length)
   {
      towerArray[i].damageBuff = 0;
      towerArray[i].rangeBuff = 0;
      i++;
   }
   b = 0;
   while(b < towerArray.length)
   {
      buff(towerArray[b],1);
      b++;
   }
   buff(_root._game.towers.newt.placer,0);
}

function buff(tower, OK)
{
   if(tower.Type == "buffD" || tower.Type == "buffR")
   {
      ...
      dist = Math.sqrt(Math.pow(tower._x - towerArray[i]._x,2) + Math.pow(tower._y - towerArray[i]._y,2));
      if(dist < 100)
      {
         if(tower.Type == "buffD")
         {
            if(OK == 1) { towerArray[i].damageBuff += 25; }
         }
         else
         {
            if(OK == 1) { towerArray[i].rangeBuff += 25; }
         }
         towerArray[i].buffedRange = towerArray[i].range + towerArray[i].range / 100 * towerArray[i].rangeBuff;
         towerArray[i].buffedDamage = towerArray[i].damage + towerArray[i].damage / 100 * towerArray[i].damageBuff;
      }
   }
}
```

La candidatisation peut donc utiliser `R=100` pour buffD/buffR, même si `range=80` apparaît ailleurs (ex. affichage radar).

### 5.5 Recalcul multi-map

Pour chaque map `M` :

- Calculer `B(M)` (cells constructibles).
- Calculer `cells_M[t]` en appliquant le score + diversité sur `B(M)`.
- Conserver `cells_M[t]` fixe pendant tout l’épisode (et en général pendant tout l’entraînement sur cette map).

Ainsi, l’agent voit la même API, quelle que soit la map.

---

## 6) Tours existantes via `Ktower` slots stables

### 6.1 Pourquoi des “slots” ?

Les actions `UPGRADE(tower_id)` / `SELL(tower_id)` / `SET_MODE(tower_id, mode)` imposent un identifiant de tour.

Pour garder `Discrete(K)` fixe :

- On expose `Ktower` slots : `slot 0..Ktower-1`.
- Chaque slot pointe vers **une tour existante** ou est vide.

### 6.2 Règle d’assignation stable (recommandée)

Option simple (stable, déterministe) :

- À chaque état, l’environnement maintient une liste `tower_array` dans l’ordre de création (comme SWF `towerArray.push(T)`).
- Slot `i` correspond à `tower_array[i]` si `i < len(tower_array)`, sinon vide.
- Lors d’un `SELL`, la tour est retirée ; tous les slots “au-dessus” se décalent.

Avantage :
- comportement identique au SWF (où les tours sont gérées via un tableau).

Inconvénient :
- l’identité d’un slot peut “glisser” après un sell (mais c’est gérable, et SB3 apprend souvent ce pattern).

Alternative (plus stable) :
- donner un `tower_uid` monotone et garder les slots triés par `tower_uid`, en laissant des trous.  
  Cela stabilise l’identité d’un slot mais nécessite plus de bookkeeping.

### 6.3 Modes de ciblage : liste et contraintes SWF

Observations ActionScript :

- Beaucoup de tours (green/pink/brown/red) implémentent explicitement : `closest`, `weakest`, `hardest` (via `if(targetMode == "...")`).  
- `tower_blue` et `tower_bash` ciblent “fastest” (ils comparent `C.speed > qualifier`).  
- `tower_swarm` est “random” (sélection aléatoire dans `targetArray`).

Donc :

- Définir `MODES = ["closest","weakest","hardest","fastest","random"]`.
- Masquer `SET_MODE` si le mode n’est pas supporté par ce type (ex. une green ne doit pas “random”, une swarm ne doit pas “hardest”, etc.).


**Extrait SWF (modes de ciblage supportés)** — `important_scripts/scripts/DefineSprite_525/frame_1/DoAction.as` :

```actionscript
if(n.targetMode == "random")        { _ui.sidebar.button_firemode.gotoAndStop(2); }
else if(n.targetMode == "fastest")  { _ui.sidebar.button_firemode.gotoAndStop(3); }
else if(n.targetMode == "closest")  { ... m1 ... }
else if(n.targetMode == "hardest")  { ... m2 ... }
else                                { ... m3 ... } // correspond à "weakest" ailleurs
```

---

## 7) Format exact `F` (features par candidat) : compact + multi-maps

### 7.1 Principe

Pour que l’agent puisse choisir parmi `PLACE(t, k)` sans avoir “en mémoire” la géométrie brute de la carte, il est utile d’exposer, dans l’observation, des features **par candidat**.

But :

- rester compact (éviter une grande “image” de la map),
- rester map-agnostique (features normalisées),
- rester informatif (couverture du chemin, coût, portée, etc.).

### 7.2 Définition de `F` pour un candidat de placement

Pour chaque type `t`, et chaque slot candidat `k` (0..`Kcells[t]-1`), on expose un vecteur `F_place[t,k]` de taille **12** :

`F_place[t,k] = [`

0. `valid` : 1.0 si la cellule est utilisable maintenant, sinon 0.0  
   (validité dynamique : pas déjà occupée, constructible, dans la zone, etc.)

1. `affordable` : 1.0 si `bank >= cost(t)` (ou `ups >= 1` pour buff), sinon 0.0

2. `cost_norm` : `log1p(cost(t)) / log1p(cost_ref)` (ex. `cost_ref=5000`)

3. `range_norm` : `range_px(t) / 200.0` (200 px = borne “au-dessus” de la plus grande portée actuelle)

4. `coverage` : `coverage_t(c)` (section 5.3), dans `[0,1]`

5. `dist_to_path_norm` : `min_dist_to_path(c) / 200.0` clamp `[0,1]`

6. `near_spawn_norm` : distance du candidat au premier point du chemin, / 600 clamp `[0,1]`  
   (indication “early coverage”)

7. `near_end_norm` : distance au dernier point du chemin, / 600 clamp `[0,1]`  
   (indication “late defense”)

8. `local_density` : fraction des candidats “buildables” dans un disque de rayon 2 cases autour de `c`  
   (proxy de “zone de cluster”)

9. `tower_count_in_buff_radius` : nombre de tours existantes dans un rayon 100 px (normalisé / `Ktower`)  
   (utile surtout pour buffs)

10. `is_buffD` : 1.0 si `t == tower_buff1`, sinon 0.0

11. `is_buffR` : 1.0 si `t == tower_buff2`, sinon 0.0

`]`

Notes :

- Les composantes 4–8 peuvent être **pré-calculées** à `reset(map)` et stockées pour accélérer.
- La composante 9 est dynamique (dépend des tours existantes).
- `valid` et `affordable` sont aussi dynamiques et doivent matcher le masque d’action.

### 7.3 Autres features (hors `F`) nécessaires en observation

En plus des `F_place`, il faut des features globales minimales :

- `bank_norm`, `ups_norm`, `interest_norm`, `lives_norm`, `wave_norm`, `score_norm`, `baseHP_norm`, etc.  
  (le gameplay SWF montre que bank/interest/ups/lives sont structurants : `DefineSprite_525.setup()`)


**Extrait SWF (setup économie / vies / bank)** — `important_scripts/scripts/DefineSprite_525/frame_1/DoAction.as` :

```actionscript
bank = 250;
interest = 3;
ups = 0;
lives = 20;
score = 0;
```

Et des features par tour slot :

- `tower_present` (0/1),
- `tower_type_id` (one-hot ou id normalisé),
- `level_norm`,
- `damage_norm`, `range_norm`, `damageBuff_norm`, `rangeBuff_norm`,
- `mode_id` (one-hot ou id),
- `cooldown_norm` (si modèle le suit).

---

## 8) Action masking : règles exactes et instrumentation anti-Simplex

### 8.1 Règle d’or : garantir `mask.any() == True`

À chaque appel de `get_action_mask()` :

- vérifier (assert/log) qu’au moins une action est valide.
- sinon, forcer `NOOP` valide (fallback), et logguer l’état (pour débug).

Exemple :

```python
mask = np.zeros((K,), dtype=bool)
mask[NOOP] = True  # fallback
...
assert mask.any()
```

### 8.2 Masque “placer”

Pour `PLACE(t, k)` :

Valide ssi :

1) `cells[t][k]` existe (sinon slot de padding)  
2) `can_place_tower(t, cells[t][k]) == True` (constructibilité + cellule libre)  
3) ressource ok :
   - si type “tower” : `bank >= cost(t)`
   - si buff : `ups >= 1`

Ces règles reflètent le SWF : paiement `bank -= cost` si `Type == "tower"`, sinon `ups--`.

### 8.3 Masque “upgrade/sell/set_mode”

Pour un slot `i` :

- `UPGRADE(i)` valide ssi slot occupé ET `bank >= int(baseCost/2)` (logique SWF `upgrade(n)`).
- `SELL(i)` valide ssi slot occupé.
- `SET_MODE(i,m)` valide ssi slot occupé ET `m` supporté par le type de tour.

### 8.4 Instrumentation pour prévenir les retours de masques incohérents

À activer au moins en debug :

- log d’un résumé du masque : `n_valid`, répartition par blocs.
- log si un bloc entier devient invalide (ex. plus aucune action PLACE possible alors que bank est haut).

En cas de crash “Simplex” : écrire les 3 diagnostics suivants dans les logs (en une seule ligne JSON) :

- `mask_any`: bool  
- `mask_valid_count`: int  
- `mask_place_valid_count`: int  
- `mask_towerop_valid_count`: int  
- `bank`, `ups`, `len(towers)`  
- `wave`, `tick`

---

## 9) Plan d’implémentation pas à pas

La liste ci-dessous est conçue pour être appliquée sans tout réécrire en une fois.

### Étape 1 — Introduire un “Action Table” stable

1) Créer un module (ex.) `vectortd/ai/action_space/discrete_k.py` contenant :

- une dataclass `ActionSpec(op: str, t: int|None, k: int|None, slot: int|None, mode: int|None)`
- une classe `DiscreteKActionTable` :
  - `__init__(tower_types, Kcells_by_type, Ktower, modes)`
  - `build_for_map(map_data) -> (action_space: Discrete, table: list[ActionSpec], cells_by_type: dict[t, list[cell]])`

2) Test unitaire local (sans RL) :
- vérifier que `len(table) == action_space.n`
- vérifier l’ordre et la stabilité (mêmes paramètres ⇒ même table).

### Étape 2 — Implémenter la candidatisation géométrique (section 5.3)

1) Créer une fonction `compute_path_samples(map_data, step_px=10) -> np.ndarray[(N,2)]`.
2) Créer `score_cells_for_type(buildable_cells, path_samples, range_px) -> list[(cell, score)]`.
3) Créer `select_top_k_diverse(scored, K, dmin_px) -> list[cell]`.

Test à introduire “au fil de l’eau” :
- sur Switchback, vérifier que `len(cells[t]) == Kcells[t]` pour tous `t`, ou bien que les slots “manquants” sont bien paddés et masqués.

### Étape 3 — Modifier l’environnement Gym

1) Dans l’Env :
- remplacer `action_space` actuel par le `Discrete(K)` fourni par `DiscreteKActionTable`.
- stocker `self._action_table` et `self._cells_by_type`.

2) Implémenter `step(action_idx)` :
- décoder via `spec = self._action_table[action_idx]`
- exécuter :
  - `NOOP` : rien
  - `START_WAVE` : appel moteur
  - `PLACE` : `place_tower(spec.t, cells[t][spec.k])`
  - `UPGRADE/SELL/SET_MODE` : résoudre le tower slot ⇒ tour ⇒ appel moteur

Test :
- en mode “replay” / step manuel, vérifier que chaque action fait exactement ce qui est attendu.

### Étape 4 — Implémenter `get_action_mask()`

1) Calculer `mask = np.zeros((K,), dtype=bool)`.
2) Mettre `mask[NOOP] = True` en fallback.
3) Parcourir les specs :
- pour PLACE : `can_place_tower` + ressources
- pour ops tours : slot occupé + règles

Test :
- à chaque `reset()` et `step()`, vérifier `mask.any()` et logguer `n_valid`.

### Étape 5 — Ajouter les features `F` dans l’observation (section 7)

1) Ajouter dans l’observation un bloc “candidats” :
- `F_place` flatten en un vecteur 1D (si policy MLP),
- ou exposé en `Dict` si MultiInputPolicy.

2) Pré-calculer les champs statiques (coverage, dist_to_path, near_spawn/end, density) et les stocker.

Test :
- vérifier que les features restent dans des bornes raisonnables `[0,1]` (sauf log norm).
- vérifier qu’un candidat masqué a `valid=0` et souvent `affordable=0` si non payable.

### Étape 6 — Migration incrémentale (Switchback → multi-map)

Palier recommandé (qui n’est pas un détour) :

1) Switchback d’abord :
- candidatisation “par type” en place,
- `Kcells[t]` comme ci-dessus,
- features `F` actives.

2) Puis multi-map :
- calcul automatique de `cells_M[t]` à `reset(map_id)`,
- normalisation des features indépendante de la map (bornes fixes),
- éventuellement curriculum (maps faciles → difficiles).

---

## Annexe A — Rappels “tours / coûts / portées” (extrait)

Référence synthétique : `towers.md`.

- `tower_green` : cost=100, range=70  
- `tower_green2` : cost=400, range=70  
- `tower_green3` : cost=2000, range=70  
- `tower_pink1` : cost=300, range=100  
- `tower_pink2` : cost=900, range=100  
- `tower_pink3` : cost=2800, range=100  
- `tower_brown` : cost=200, range=80  
- `tower_blue` : cost=300, range=70  
- `tower_bash` : cost=500, range=80  
- `tower_swarm` : cost=800, range=80  
- `tower_red` : cost=2500, range=150  
- buffs : `tower_buff1`/`tower_buff2` coût=1 ups, range=80, effet si dist < 100




---

## Annexe B — Reward : densifier + rescaler vers ~[-10, +10] (sans reward “dégâts infligés”)

Objectif : éviter des épisodes à `~ -2000` (difficiles à calibrer) et obtenir un signal fréquent mais sobre.

### B.1 Principes

1) **Terminal** : doit dominer le “sens” (gagner/perdre).  
2) **Densité** : injecter un signal régulier lié au progrès (vagues franchies) et à la survie (vies perdues).  
3) **Scale** : viser un total typique par épisode dans `[-10, +10]` (ou `[-20,+20]` si PPO est stable, mais commencer plus petit).

### B.2 Proposition simple (robuste multi-maps)

Variables :
- `L = lives` (0..20)
- `W = wave_index` (0..Wmax)
- `done` : fin d’épisode
- `win` : bool

Définir à chaque transition :

1) **Perte de vies** (signal immédiat)  
   `r_life = -0.5 * max(0, L_prev - L_now)`  
   Interprétation : perdre toutes les vies ⇒ `-10` (scale OK).

2) **Progrès par vague** (signal densifié)  
   Quand `W_now > W_prev` (une vague vient d’être validée) :  
   `r_wave = +0.2 * (W_now - W_prev)`  
   Exemple : 40 vagues ⇒ `+8` si tout est fini (sans bonus terminal).

3) **Terminal**  
   Si `done` :
   - si `win` : `r_terminal = +2.0`
   - sinon : `r_terminal = -2.0`

Total (par épisode) :
- victoire “propre” : ~`+8..+10`
- défaite par 20 vies perdues : `-10..-12`
- cas intermédiaires : la réduction des pertes de vies est directement visible.

### B.3 Implémentation (pas à pas)

1) Dans l’Env, garder `self._prev_lives`, `self._prev_wave`.  
2) À chaque `step()` après l’avancement du moteur :
   - calculer `delta_lives = prev - now`
   - calculer `delta_wave = now - prev`
   - construire `reward = r_life + r_wave + r_terminal`
3) Mettre à jour `prev_*`.  
4) Ajouter un log (debug) : `{"r_life":..., "r_wave":..., "r_terminal":..., "L":..., "W":...}`.

Test minimal (à écrire immédiatement pendant l’étape 1) :
- simuler un épisode où 1 vie est perdue : vérifier `reward == -0.5` (à epsilon près)
- simuler passage de vague : vérifier `+0.2`
- simuler `done/win` : `+2`, sinon `-2`.

### B.4 Ajustements possibles (si signal trop faible/fort)

- Si l’agent “stagne” : augmenter `r_wave` à `0.25` et/ou `r_terminal` à `±3`.  
- Si le critic explose : réduire `r_terminal` et `r_life`, ou augmenter `vf_coef` (sans changer l’ordre de grandeur total).
