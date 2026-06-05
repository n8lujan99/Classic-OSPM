# init.py
"""
OSPM initialization.
Defines global physical constants and performs lightweight environment checks.
"""

# -------------------------------
# Optional: environment sanity check
# -------------------------------
def check_environment():
    """
    Light sanity check for required packages.
    Does NOT install anything.
    """
    required = [
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
    ]

    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        raise RuntimeError(
            f"Missing required packages: {missing}\n"
            "Install them before running OSPM."
        )

# Run check only when explicitly requested
if __name__ == "__main__":
    check_environment()
    print("[OSPM init] Environment OK.")
