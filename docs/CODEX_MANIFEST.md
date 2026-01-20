# docs/CODEX_MANIFEST.md

# VectorTD (SWF) – CODEX_MANIFEST
But : fournir à Codex (dans VS Code) une carte claire de “quoi lire, quand, et pour quoi faire”, afin de recréer VectorTD (version simplifiée) en Python à partir des scripts ActionScript extraits du SWF.

Portée immédiate :
- Implémentation jouable et simulable (sans effets visuels Flash) centrée sur la map **SWITCHBACK**.
- Les autres maps seront implémentées ensuite (elles sont déjà encodées par frames dans les scripts), et l’IA entraînée devra jouer à **toutes** les maps (et ensuite à d’autres Tower Defense via une interface d’environnement stable).

Principe d’usage pour Codex :
- Ne pas “charger tout le repo dans le prompt”.
- À la place : lire en priorité le **noyau** (Core) + la **map courante**, puis lire à la demande les tours/handlers UI nécessaires à la feature du moment.

---

## 0) Contexte map : comment “Switchback” est sélectionnée dans le SWF
Ces scripts montrent le mapping mapName/Level/frame :
- `scripts/DefineSprite_491/frame_1/PlaceObject2_469_1/CLIPACTIONRECORD on(release).as`
  - Sur clic : `_root._game._map.gotoAndStop(1); _root._game.Level = 1; _root._game.mapName = "SWITCHBACK";`

Cela justifie :
- Switchback == `Level = 1`
- Switchback == `_map` frame 1
- Donc données de chemins Switchback == `DefineSprite_325/frame_1/DoAction.as`

---

## 1) CORE (à lire presque à chaque fois)
### 1.1 Contrôleur principal du gameplay
- `scripts/DefineSprite_525/frame_1/DoAction.as`
Rôle :
- Initialise et maintient l’état global du jeu.
- Contient la logique des vagues, du spawn, du kill, de l’économie, des buffs, upgrade/sell, et (dans le SWF) une partie de l’UI.
Symboles et fonctions structurantes à extraire en priorité :
- Variables d’état : `bank`, `interest`, `ups`, `lives`, `score`, `level` (numéro de vague), `Level` (index de map), `baseHP`, `baseWorth`, `bonusEvery`, `Paused`, `autoLevel`
- Table de vagues : `levels = [ ... ]` (séquence des types de creeps par vague)
- Fonctions : `setup()`, `wave()`, `waveB()`, `spawn()`, `kill()`, `doBuffs()`, `buff()`, `upgrade()`, `sell()`
- Fonction centrale de “tir” : `fire(fromX, fromY, too, Type, Damage)` (utile même si on supprime les effets visuels)

Pourquoi c’est “core” :
- La plupart des constantes et conventions (dimensions, économie, bonusEvery, etc.) vivent ici.
- L’implémentation Python doit reproduire en premier lieu ce fichier (en séparant logique vs rendu).

### 1.2 Déplacement d’un creep (runtime)
- `scripts/DefineSprite_39_creep/frame_1/DoAction.as`
Rôle :
- Mouvement “sur rail” le long d’une liste de marqueurs `m1..mN` dans `_map`.
- Logique de progression `pathPoint`, interpolation directionnelle (atan2), accélération simple.
- Gestion du “tour” : quand `pathPoint == path.length`, décrémente `lives` et réinitialise le creep au début (convention importante de VectorTD).

Pourquoi c’est “core” :
- À reproduire fidèlement dans la simulation : c’est ce qui définit “quand le joueur perd des vies”.

### 1.3 Placement de tour (ghost + validation + instanciation)
- `scripts/DefineSprite_146_placer/frame_1/DoAction.as`
Rôle :
- Snap sur grille de pas 25, bornes map (0..550/0..450), validation collision via `_map.hit.hitTest(...)`, vérification d’occupation (pas 2 tours sur la même case).
- Paiement : soit via `bank` (Type "tower"), soit via `ups` (Type != "tower"), puis `attachMovie(...)` et push dans `towerArray`.
- Déclenche `doBuffs()` régulièrement.

Pourquoi c’est “core” :
- Pour une IA : l’action la plus importante est “placer une tour”, donc la logique de validité doit être codée proprement côté Python.

---

## 2) MAP (Switchback en priorité)
### 2.1 Données de chemins Switchback
- `scripts/DefineSprite_325/frame_1/DoAction.as`
Contenu :
- `paths = [[1,3,5,...,33],[2,4,6,...,34]];`
- `spawnDir = "up";`
Notes :
- Les indices (1..34) référencent des marqueurs nommés `m1..m34` dans le clip `_map`. Les positions (x,y) ne sont pas dans ce `.as` : elles sont dans les placements du SWF.

### 2.2 Autres maps (à implémenter ensuite)
Même sprite, autres frames :
- `scripts/DefineSprite_325/frame_2/DoAction.as` … `frame_8/DoAction.as`
Chacune définit son `paths` + `spawnDir`.
Objectif :
- Mettre en place un format de données “map” en Python/JSON qui permet d’importer Switchback puis les 7 autres sans changer l’engine.
- L’IA devra être entraînée sur plusieurs maps (généralisation intra-jeu), puis extrapoler vers d’autres TD (généralisation inter-jeux).

---

## 3) TOURS (à lire à la demande)
Chaque tour est un “agent” avec scan des creeps, choix de cible, cadence, puis appel à `_root._game.fire(...)`.

Tours importantes (attaque principale) :
- `scripts/DefineSprite_106_tower_green/frame_1/DoAction.as` (GREEN LASER 1)
- `scripts/DefineSprite_110_tower_green2/frame_1/DoAction.as`
- `scripts/DefineSprite_114_tower_green3/frame_1/DoAction.as`
- `scripts/DefineSprite_92_tower_blue/frame_1/DoAction.as`
- `scripts/DefineSprite_133_tower_red/frame_1/DoAction.as`
- `scripts/DefineSprite_99_tower_brown/frame_1/DoAction.as`
- `scripts/DefineSprite_121_tower_pink1/frame_1/DoAction.as`
- `scripts/DefineSprite_125_tower_pink2/frame_1/DoAction.as`
- `scripts/DefineSprite_129_tower_pink3/frame_1/DoAction.as`
- `scripts/DefineSprite_137_tower_swarm/frame_1/DoAction.as`
- `scripts/DefineSprite_88_tower_bash/frame_1/DoAction.as`

Tours “buff” (l’engine des buffs est surtout dans DefineSprite_525) :
- `scripts/DefineSprite_141_tower_buff2/frame_1/DoAction.as`
- `scripts/DefineSprite_145_tower_buff1/frame_1/DoAction.as`

Conseil de progression (pour Codex) :
- Implémenter d’abord 1 seule tour (ex: GREEN LASER 1) pour valider le pipeline : spawn → move → target → damage → kill → économie.
- Puis ajouter les autres types (modes de ciblage, effets, buffs, etc.).

---

## 4) HANDLERS UI (optionnels si on remplace par une interface Python)
Ces scripts montrent comment le SWF déclenche les fonctions core, utile pour concevoir l’API d’action de l’IA :
- Lancer la vague / auto :
  - `scripts/DefineSprite_386/frame_1/PlaceObject2_371_1/CLIPACTIONRECORD on(release).as`
  - `scripts/DefineSprite_386/frame_1/PlaceObject2_371_3/CLIPACTIONRECORD on(release).as`
- Choix du mode de ciblage :
  - `scripts/DefineSprite_442/frame_1/PlaceObject2_430_2/CLIPACTIONRECORD on(release).as` (closest)
  - `scripts/DefineSprite_442/frame_1/PlaceObject2_435_5/CLIPACTIONRECORD on(release).as` (hardest)
  - `scripts/DefineSprite_442/frame_1/PlaceObject2_438_8/CLIPACTIONRECORD on(release).as` (weakest)
- Upgrade / Sell :
  - `scripts/DefineSprite_447/frame_1/PlaceObject2_413_85/CLIPACTIONRECORD on(release).as` (upgrade)
  - `scripts/DefineSprite_424/frame_2/PlaceObject2_423_12/CLIPACTIONRECORD on(release).as` (sell)

---

## 5) “À ignorer” pour l’engine Python (sauf si besoin ultérieur)
- UI générique `mx/*`, composants de scrollbar, skins, menus, etc.
- Effets de rendu Flash (draw lines, explosions décoratives) si on fait une simulation headless.


---

## 6) Tests (definitions JSON)
Quand un nouveau test est fourni sous forme de définition JSON dans `data/tests` :
- Créer ou mettre à jour un test Python dans `src/vectortd/tests` qui charge ce JSON et exécute le runner (pattern de `src/vectortd/tests/test_green_laser.py`).
- Mettre à jour `.vscode/launch.json` → `inputs.testName.options` pour ajouter l'option du nouveau test.
Notes :
- Le GUI en mode test lit uniquement `data/tests`.
- Le nom attendu par `--test` est le **nom de fichier sans extension** (ex: `data/tests/foo.json` → `--test foo`), pas le champ `"name"`.

## 7) Infos diverses

- Des infos sur les précédentes exécutions sont présents dans runs/N où N est un compteur qui s'incrémente à chaque exécution :
  - Des screenshots sont disponibles dans screenshots . Aller les regarder dès que nécessaire (question de GUI, de rendu, etc.)
  - Un log du terminal est présent, le lire dès que pertinent.
- Des notes de TODO liées à l'IA sont dans `TODO/` (ex: `TODO/TRAIN_WHILE_EVAL.md`).

- Le jeu est représenté en deux parties : l'écran de jeu (game screen) à gauche et la barre de contrôle (control bar or sidebar) à droite.

## 8) Développement IA

Chaque fois que nous ferons des découvertes concernant l'apprentissage IA (pas des bugs de code ou d'environnement mais vraiment des faiblesses ou impasses de l'apprentissages, ou des améliorations ou modifications dans le retour/la récompense, ou des changements dans l'archi de l'IA ou l'agorithme d'apprentissage) tous les choix et leurs raisons (et les observations qui ont mené à tout ça) devront être documentées dans un paragrpahe de AI_LESSONS_LEARNED.md Exemple :

Observation : l'agent place des tours dans des endroits éloignés des ennemis et qui restent inactives.
Conséquence : argent perdu
Solution : Mettre une récompense pour chaque cycle qu'une tour aura été active pendant une vague. Plus les tours placées sont actives, plus elles génèrent de récompense.

ATTENTION c'est un exemple, ça ne veut pas dire que c'est une bonne idée !

SMP fera référence à "Stable Baselines 3 (Contrib) MaskablePPO".

On n'oubliera pas, lors de développement de nouvelles fonctioannlités d'entraînement ou autres modifications (logging, etc.) d'ajouter à run_metadata ou run_summary ce qui est pertinent de l'être.
