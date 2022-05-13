import sys

has_seaborn = True
try:
    import seaborn
except:
    has_seaborn = False

missing = []

if not has_seaborn:
    missing.append("seaborn")


if missing:
    print(f"You're missing one or more required packages: {','.join(missing)}")

    command = f"python -m pip install {' '.join(missing)}"
    print(f"Install them with the following command:\n\n\t{command}\n")
    exit(1)

exit(0)
