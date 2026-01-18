import inspect
import sys


def main() -> int:
    print("=== Python ===")
    print(sys.version)
    print()

    print("=== Package versions ===")
    try:
        import stable_baselines3 as sb3
        print("stable_baselines3:", sb3.__version__)
    except Exception as e:
        print("stable_baselines3: import failed:", repr(e))

    try:
        import sb3_contrib
        print("sb3_contrib:", sb3_contrib.__version__)
    except Exception as e:
        print("sb3_contrib: import failed:", repr(e))

    try:
        import gymnasium as gym
        print("gymnasium:", gym.__version__)
    except Exception as e:
        print("gymnasium: import failed:", repr(e))

    try:
        import torch
        print("torch:", torch.__version__)
    except Exception as e:
        print("torch: import failed:", repr(e))

    print()

    print("=== Key imports ===")
    try:
        from sb3_contrib import MaskablePPO
        print("MaskablePPO import: OK ->", MaskablePPO)
        print("MaskablePPO.__init__:", inspect.signature(MaskablePPO.__init__))
        print("MaskablePPO.predict:", inspect.signature(MaskablePPO.predict))
    except Exception as e:
        print("MaskablePPO import/signature failed:", repr(e))

    print()

    try:
        from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
        print("MaskableEvalCallback import: OK ->", MaskableEvalCallback)
        print("MaskableEvalCallback.__init__:", inspect.signature(MaskableEvalCallback.__init__))
    except Exception as e:
        print("MaskableEvalCallback import/signature failed:", repr(e))

    print()

    print("=== SB3 VecEnv imports ===")
    try:
        from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
        print("SubprocVecEnv:", SubprocVecEnv)
        print("DummyVecEnv:", DummyVecEnv)
    except Exception as e:
        print("VecEnv imports failed:", repr(e))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
