# docs/MAP_SWITCHBACK_MARKERS_TODO.md

# Switchback – extraction des marqueurs m1..m34 (SWF → données Python/JSON)

## 0) Objet du document
La map **Switchback** encode ses chemins sous forme de listes d’indices (1..34) qui référencent des **marqueurs** nommés `m1`, `m2`, … `m34` dans le clip `_map`.  
Les fichiers ActionScript donnent **l’ordre des marqueurs**, mais **pas leurs coordonnées**. Pour reconstruire la map en Python (simulation + IA), il faut extraire la position (x,y) de chaque marqueur depuis le SWF (via JPEXS/FFDec ou autre).

Ce travail est à faire **une fois par map**. Le même pipeline sera ensuite appliqué aux autres maps (frame 2..8), afin que l’engine et l’IA puissent jouer toutes les maps de VectorTD, puis d’autres TD via la même interface d’environnement.

---

## 1) Références ActionScript (preuve de la dépendance aux marqueurs)
### 1.1 Switchback : chemins et direction de spawn
Source : `scripts/DefineSprite_325/frame_1/DoAction.as`
- `spawnDir = "up"`
- `paths = [[1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33],
            [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34]]`

Interprétation :
- Il y a **34 marqueurs** utilisés (`m1..m34`).
- Il y a **2 chemins** (2 lanes) de longueur 17 chacun.

### 1.2 Le moteur “vise” explicitement `_map["m" + idx]`
Source : `scripts/DefineSprite_525/frame_1/DoAction.as`
- Lors du spawn : position de départ issue de `m[path[0]]`
- `C.targ = _map["m" + Path[0]]` au départ

Source : `scripts/DefineSprite_39_creep/frame_1/DoAction.as`
- À chaque changement de waypoint : `targ = _parent._map["m" + path[pathPoint]]`

Conclusion :
- Les (x,y) des instances `m1..m34` sont des données indispensables (au moins pour la simulation headless).

---

## 2) Ce qui doit être extrait (Switchback)
### 2.1 Table de marqueurs
- `m1..m34` : `x`, `y` (coordonnées dans le repère du clip `_map`)
- (Optionnel mais utile) : taille de la map jouable et origine si décalage (Flash peut avoir un offset visuel)

### 2.2 Collision / zones non constructibles
Le SWF utilise une forme de collision (hitTest) pour empêcher la construction sur le chemin.
Source : `scripts/DefineSprite_146_placer/frame_1/DoAction.as` (placer)
- Validation : `_root._game._map.hit.hitTest(_X, _Y, true)` (et variantes)

À extraire (au choix) :
- Soit une **grille buildable** (bool par cellule 25×25),
- Soit un **masque** (polygone(s) ou bitmap) permettant de reproduire la règle `hitTest` côté Python.

Recommandation (plus simple et robuste pour IA) :
- Construire une **grille buildable** au pas 25 (même pas que le placer).

---

## 3) Où les marqueurs se trouvent dans le SWF (pistes JPEXS)
Dans VectorTD, les marqueurs `m#` sont typiquement des instances placées dans le MovieClip de la map (souvent un sprite `_map`) :

Checklist d’exploration dans JPEXS :
1. Ouvrir le SWF.
2. Trouver le clip/DefineSprite correspondant à `_map` (celui qui contient le décor de la map).
3. Dans ce clip, lister les instances placées (PlaceObject2 / DisplayList).
4. Repérer les instances dont le **nom d’instance** est `m1`, `m2`, …, `m34`.
5. Pour chacune :
   - relever la matrice de transformation (translation) → `x`, `y`
   - noter si les coordonnées sont locales au clip `_map` (le plus probable)

⚠️ Important :
- Le `.as` n’inclut pas les coordonnées : elles résident dans les placements d’instances (DisplayList), donc dans les tags du SWF.

---

## 4) Format de sortie recommandé (JSON)
Créer un fichier data dédié (réutilisable par l’engine et l’IA), par exemple :
- `data/maps/switchback.json`

### 4.1 Schéma minimal
```json
{
  "name": "SWITCHBACK",
  "level_index": 1,
  "spawn_dir": "up",
  "markers": {
    "1": [x1, y1],
    "2": [x2, y2],
    "...": "...",
    "34": [x34, y34]
  },
  "paths": [
    [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33],
    [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34]
  ]
}
