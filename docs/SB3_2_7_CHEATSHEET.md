# SB3 / sb3-contrib 2.7.x – Cheat sheet (MaskablePPO)

## Imports (2.7.x)
- MaskablePPO:
  - `from sb3_contrib import MaskablePPO`

- MaskableEvalCallback:
  - `from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback`

## Env requirements (MaskablePPO)
- Env Gymnasium doit implémenter:
  - `reset(self, *, seed=None, options=None) -> (obs, info)`
  - `step(self, action) -> (obs, reward, terminated, truncated, info)`
- Action masking:
  - Méthode dans l'Env: `action_masks(self) -> np.ndarray`
  - Contrainte: bool array, True = action valide, shape = (action_space.n,)
  - Avec SubprocVecEnv: action_masks() doit être dans l'Env (pas uniquement wrapper)

## Predict avec masques
- Inference / eval:
  - `action, _ = model.predict(obs, action_masks=mask, deterministic=True)`

## VecEnvs SB3
- Train:
  - `from stable_baselines3.common.vec_env import SubprocVecEnv`
- Monitor:
  - `from stable_baselines3.common.monitor import Monitor`

## Eval “best model”
- Utiliser MaskableEvalCallback:
  - `best_model_save_path=...`
  - `log_path=...`
  - `callback_on_new_best=...` (pour déclencher replay)

## Multiprocessing Windows
- Utiliser `if __name__ == "__main__":`
- Utiliser `multiprocessing.freeze_support()`
- Factories picklables (top-level), pas de lambda capturant des objets
