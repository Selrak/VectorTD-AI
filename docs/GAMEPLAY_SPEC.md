# docs/GAMEPLAY_SPEC.md

# VectorTD (version simplifiée) – GAMEPLAY_SPEC
Ce document spécifie la reproduction Python de la logique du SWF, en se basant sur les scripts ActionScript extraits. L’objectif est double :
1) recréer un gameplay fidèle (au moins sur Switchback),
2) fournir une simulation stable et déterministe pour entraîner une IA (puis généraliser aux autres maps du jeu, puis à d’autres Tower Defense).

Sources principales (ActionScript) :
- Core : `scripts/DefineSprite_525/frame_1/DoAction.as`
- Creep : `scripts/DefineSprite_39_creep/frame_1/DoAction.as`
- Placement : `scripts/DefineSprite_146_placer/frame_1/DoAction.as`
- Map Switchback : `scripts/DefineSprite_325/frame_1/DoAction.as`

---

## 1) Conventions “monde” (coordonnées et dimensions)
Références dans le code :
- Les tours considèrent un creep “dans la map” si `0 < C._x < 550` et `0 < C._y < 450` (vu dans les scripts de tour).
- Le placer borne la zone : `_X >= 0 && _Y >= 0 && _X < 550 && _Y < 450`.
- La grille de placement utilise un pas de 25 : `_X = int(.../25)*25` et `_Y = int(.../25)*25 - 75`.

Spécification Python (propre et stable) :
- Coordonnées monde en pixels “logiques”.
- Dimensions de la zone jouable : largeur 550, hauteur 450 (comme le SWF).
- Grille de placement : pas 25.
- On isole l’offset UI Flash (ex: `-75`) : en Python, on définit simplement une grille (x,y) cohérente dans [0..550)×[0..450).

---

## 2) Données de map : marqueurs + chemins (Switchback)
Dans le SWF, le pathfinding n’est pas un A* : c’est un déplacement sur une liste de points.
- Les chemins sont encodés sous forme de listes d’indices, et on vise des marqueurs `_map["m" + idx]`.

Switchback (map actuelle) :
- Fichier : `scripts/DefineSprite_325/frame_1/DoAction.as`
- Données :
  - `paths = [[1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33],
             [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34]]`
  - `spawnDir = "up"`

Point crucial :
- Les positions (x,y) de `m1..m34` ne sont pas dans ce `.as`.
- Il faut les extraire du SWF (placements/instances) ou les reconstruire manuellement pour la version simplifiée.
- Format cible recommandé pour Python :
  - `markers: dict[int, (x,y)]`
  - `paths: list[list[int]]`
  - `spawn_dir: str`

---

## 3) État global (init) – d’après `setup()`
Source : `scripts/DefineSprite_525/frame_1/DoAction.as`, fonction `setup()`.

Variables initiales (valeurs observables dans le code) :
- `bank = 250`
- `interest = 3`
- `ups = 0`
- `level = 0` (numéro de vague, minuscule)
- `lives = 20`
- `score = 0`
- `bonusEvery = 5`
- `Paused = 0`
- `baseWorth = 3`
- `baseHP = 550` si `Level < 5`, sinon `650`
- `Level = 1` pour Switchback (index de map, majuscule)

Table de vagues :
- `levels = [2,1,2,3,7,4,2,5,2,7,2,3,2,4,7,5,2,1,2,7,2,4,2,5,7,1,2,3,2,7,4,2,5,2,7,5,2,1,2,8]`
Interprétation :
- `level` (minuscule) indexe la vague courante.
- `levels[level-1]` donne le “Type” de creep à spawner pour cette vague.
- `Type == 8` déclenche une logique de vague “mixée” (voir spawn).

---

## 4) Déroulé d’une vague (wave / waveB / spawn)
Sources : `wave()`, `waveB()`, `spawn()` dans `DefineSprite_525/frame_1/DoAction.as`.

### 4.1 Déclenchement `wave()`
Précondition :
- `Paused == 0`

Effets :
- `level++`
- Gain d’intérêt avant spawn :
  - `bank += int(bank / 100 * interest)`
  - `score += int(bank / 100 * interest)`
- Sélection de type de creep :
  - `Type = levels[level - 1]`
- Récupère les données de map :
  - `spawnDir = _map.spawnDir`
  - `paths = _map.paths`

### 4.2 Spawn par chemin
Boucle :
- Pour chaque chemin `P = paths[i]` :
  - point de base : `(X,Y) = position du marqueur m[ P[0] ]`
  - spawn 14 creeps (`v = 1 .. 14`) en décalant la position selon `spawnDir`

Switchback :
- `spawnDir = "up"` donc pour chaque creep :
  - `Y -= 20`
  - `waveB(X,Y,P,Type)`

Donc, sur Switchback :
- 2 chemins × 14 creeps = 28 creeps par vague.

### 4.3 Bonus creep (waveB)
Source : `waveB(X, Y, P, Type)`.

Règle :
- Dernier creep de la vague (ici `v == 14`), dernière lane (`i == paths.length - 1`), et tous les `bonusEvery` niveaux :
  - si `level / bonusEvery == int(level / bonusEvery)` alors `t = 6`

HP de base par vague :
- `bHP = baseHP`
- si `t == 6 || t == 7` alors `bHP = baseHP * 1.5`
- puis `spawn(X,Y,P,t)`

### 4.4 Création du creep (spawn)
Source : `spawn(X,Y,Path,Type)`.

Cas `Type == 8` (vague “mixée”) :
- `t = int(cc / 5) + 1`
- si `t == 6` alors `t = 7`
- (cc est un compteur incrémenté à chaque spawn, remis à 1 au début de `wave()`)

Attributs fixés sur le creep :
- `C._x = X`, `C._y = Y`
- `C.creepType.gotoAndStop(t)` (graphique Flash ; en Python : type_id = t)
- `C.path = Path` (liste d’indices de marqueurs)
- `C.pathPoint = 0`
- `C.worth = baseWorth`
- Vitesse :
  - si `t == 4` alors `C.speed = 2` sinon `C.speed = 1`
  - `C.maxSpeed = C.speed`
- Points de vie :
  - si `t != 6` : `C.hp = bHP`
  - sinon : `C.hp = bHP * 4`
  - `C.maxhp = C.hp`
- Target courant :
  - `C.targ = _map["m" + Path[0]]`
  - direction initiale via atan2

Fin de vague :
- Après spawn de tous les creeps, le code augmente la difficulté :
  - `baseHP += int(baseHP / 6)` si `Level < 5`, sinon `int(baseHP / 5)`
  - `baseWorth += 1`

---

## 5) Mouvement d’un creep (tick)
Source : `scripts/DefineSprite_39_creep/frame_1/DoAction.as`.

Tick (si `Paused == 0`) :
- Avancement :
  - `_X += Xval * speed`
  - `_Y += Yval * speed`
- Accélération légère vers `maxSpeed` :
  - `if(speed < maxSpeed) speed += 0.01`
- Détection arrivée sur cible (boîte autour du point) :
  - si `abs(_X - targ._x) < speed` et `abs(_Y - targ._y) < speed` :
    - snap : `_X = targ._x; _Y = targ._y`
    - `pathPoint++`

Quand `pathPoint == path.length` :
- `g.lives--`
- `g.score -= worth * 2` puis clamp à 0
- si `lives < 1` : `gameOver()`
- puis reset :
  - `pathPoint = 1`
  - position remise au startTarg (premier point)
  - `targ = m[path[1]]`

Conséquence gameplay (importante) :
- Un creep non tué peut “boucler” et faire perdre plusieurs vies au fil du temps.
- La simulation Python doit reproduire ce comportement (ou décider explicitement de s’en écarter, mais alors ce n’est plus VectorTD).

---

## 6) Tours : placement, ciblage, tir, dégâts
### 6.1 Placement (placer)
Source : `scripts/DefineSprite_146_placer/frame_1/DoAction.as`.

- Snap :
  - x sur pas 25
  - y sur pas 25 (avec offsets UI Flash)
- Validité (OK=1) :
  - dans la zone [0..550)×[0..450)
  - pas de collision avec le chemin (hitTest sur `_map.hit`)
  - pas déjà une tour à la même coordonnée (scan `towerArray`)
- Paiement :
  - si `Type == "tower"` alors `bank -= cost`
  - sinon `ups--`
- Instanciation :
  - `attachMovie(sprite, ...)` puis `towerArray.push(T)`
  - set `T.active = 1`, `T.Type = Type`

Python (API interne recommandée) :
- `can_place_tower(tower_type, cell)` → bool + raison
- `place_tower(tower_type, cell)` → mutation d’état

### 6.2 Buffs (doBuffs / buff)
Source : `doBuffs()` et `buff()` dans `DefineSprite_525`.

- Avant recalcul :
  - remise à 0 de `damageBuff` et `rangeBuff` sur toutes les tours
- Un buff tower (Type "buffD" ou "buffR") applique (si dist < 100) :
  - +25% damageBuff (buffD)
  - +25% rangeBuff (buffR)
- Recalcule :
  - `buffedRange = range + range/100 * rangeBuff`
  - `buffedDamage = damage + damage/100 * damageBuff`

Python :
- Soit recalcul à chaque tick,
- soit recalcul uniquement à chaque changement (placement/sell/upgrade/move preview), comme le SWF le fait souvent.

### 6.3 Tir et kill
Source : `fire(...)` et `kill(C)` dans `DefineSprite_525`.

- `fire(...)` : plusieurs “Types” (lasers / effets / buffs). Pour la version simplifiée, on peut :
  - implémenter d’abord “dégâts instantanés” : `target.hp -= damage` puis si `hp <= 0` → `kill(target)`
  - ignorer les effets visuels (traits, explosions décoratives)

- `kill(C)` :
  - retire `C` de `creepArray`
  - `bank += int(C.worth)`
  - `score += int(C.worth)`
  - si `C.Type == 6` : `ups++`
  - si `creepArray` vide et `autoLevel==1` : déclenche `wave()`

---

## 7) Upgrade / Sell
Source : `upgrade(n)` et `sell(n)` dans `DefineSprite_525`.

Upgrade (si `bank >= int(baseCost/2)`) :
- `level += 1`
- `damage += int(baseDamage / 2.2)`
- `range += int(baseRange / 20)`
- `cost += int(baseCost / 2)`
- `bank -= int(baseCost / 2)`
- recalc `buffedRange/buffedDamage`

Sell :
- `bank += int(cost * 0.75)`
- retire la tour (`removeTower`) + vide l’info UI (`blankInfo`)

---

## 8) Interface IA (objectif futur, mais à prévoir dès maintenant)
Objectif : l’IA jouera Switchback puis les autres maps, puis d’autres TD.
Donc l’engine Python doit exposer une interface stable “environnement” :

Observation minimale (à chaque step) :
- Map : `markers`, `paths`, `spawn_dir`, collision grid / cells buildables
- État : `bank`, `interest`, `ups`, `lives`, `score`, `wave_index`, `baseHP`, `baseWorth`
- Creeps : position, hp/maxhp, speed/maxSpeed, type_id, pathPoint
- Tours : type, position, range/damage, buffs, targetMode, level, cooldown

Actions minimales :
- `start_wave()`
- `place_tower(tower_type, cell)`
- `upgrade_tower(tower_id)`
- `sell_tower(tower_id)`
- `set_target_mode(tower_id, mode)`

Note :
- Même si l’UI Flash est ignorée, les handlers SWF montrent quelles actions existent et comment elles mutent l’état. L’objectif est d’être “data-driven” : changer de map ne doit pas changer le code de simulation, uniquement les données.
