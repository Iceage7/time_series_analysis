import pkg_resources

packages = [
    "pandas",
    "numpy",
    "scikit-learn",
    "joblib",
    "matplotlib",
    "seaborn",
    "ta",
    "statsmodels",
    "scipy",
    "xgboost",
]

requirements = []

for pkg in packages:
    try:
        version = pkg_resources.get_distribution(pkg).version
        requirements.append(f"{pkg}=={version}")
    except pkg_resources.DistributionNotFound:

        continue

with open("requirements.txt", "w") as f:
    f.write("\n".join(requirements))

print("requirements.txt created with the following packages and versions:")
print("\n".join(requirements))
